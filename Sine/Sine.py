'''
A realization of the abstract MAML class, for the standard Sine meta-learning
benchmark.

The module includes realizations of the standard MAML and of the Risk-Averse RAML.
To enhance the benchmark complexity, we let the sine frequency vary between tasks.
'''

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import MAML
import torch
import BaseLearner
import cross_entropy_method as cem
import utils


def get_sine_model_config(n_mid=40):
    return [
        ('linear', [n_mid, 1]),
        ('relu', [True]),
        ('linear', [n_mid, n_mid]),
        ('relu', [True]),
        ('linear', [1, n_mid]),
    ]

def get_sine_model(config=None, n_mid=40):
    if config is None: config = get_sine_model_config(n_mid)
    return BaseLearner.Learner(config)


class SineMAML(MAML.MAML):
    def __init__(self, *args, ranges=((0.1,5), (0,2*np.pi), (0.3,3)),
                 ranges2=None, normalized_loss=False, **kwargs):
        default_kwargs = dict(device='cpu', meta_train_iterations=10000,
                              train_tasks_per_batch=20, max_test_steps_per_task=5,
                              test_tasks=10000)
        kwargs = utils.update_dict(kwargs, default_kwargs)
        self.normalized_loss = normalized_loss
        super(SineMAML, self).__init__(*args, **kwargs)
        self.ranges = ranges
        if ranges2 is None:
            ranges2 = ranges
        self.ranges2 = ranges2

        self.test_mode = 1
        self.test_res_collection = {}

    def switch_test_mode(self, mode):
        self.test_mode = mode
        if mode in self.test_res_collection:
            self.test_res = self.test_res_collection[mode]

    def test(self, mode=None):
        if mode is not None:
            self.switch_test_mode(mode)
        out = super().test()
        if mode is not None:
            self.test_res_collection[mode] = self.test_res
        return out

    def fun(self, x, amp, phase, freq):
        return amp * torch.sin(freq*x + phase)

    def do_sample_task(self, rng, mode):
        ranges = self.ranges
        if mode == 'test' and self.test_mode == 2:
            ranges = self.ranges2
        return [rng.uniform(*r) for r in ranges]

    def get_task_id(self, task):
        return '_'.join([f'{x:.2f}' for x in task])

    def do_sample_inputs(self, task, n, rng):
        x = rng.uniform(low=0, high=2*np.pi, size=n)
        x = self.tensorize_input(x)
        y = self.fun(x, *task)
        return x, y

    def tensorize_input(self, x):
        return torch.tensor(x, dtype=torch.float).reshape((-1, 1)).to(self.device)

    def run_task(self, model, task, samples, is_evaluation, is_finetuning):
        pred = model(samples[0])
        y = samples[1]
        loss = torch.mean((pred-y)**2)
        if self.normalized_loss:
            loss = loss / task[0]**2
        return loss

    def do_show_task(self, model, task, support_samples, query_samples, step, ax,
                     verbose=1):
        x0 = np.linspace(0, 2 * np.pi, 1000)
        x = self.tensorize_input(x0)
        y0 = self.fun(x, *task).cpu().numpy()
        if not self.plotting_task:
            ax.plot(x0, y0, 'k-')
            y1 = self.fun(support_samples[0], *task).cpu().numpy()
            ax.plot(support_samples[0].cpu().numpy(), y1,
                    'o', label='train samples')
            # y2 = self.fun(query_samples[0], *task)
            # ax.plot(query_samples[0], y2, 'o', label='eval samples')

        y = model(x).detach().cpu().numpy()
        ax.plot(x0, y, '-', label=f'step {step:d}')
        if verbose >= 1:
            print(f'{self.title} ({step}): mse = {((y-y0)**2).mean().item():.2f}')

    def finalize_task_visualization(self, ax):
        ax.set_title(self.title, fontsize=15)
        ax.legend(fontsize=12)


class CEM_Beta(cem.CEM):
    '''CEM for 3D Beta distribution.'''

    def __init__(self, phi0=(0.5,0.5,0.5), *args, ranges, **kwargs):
        phi0 = np.array(phi0)
        super(CEM_Beta, self).__init__(phi0, *args, **kwargs)
        self.default_dist_titles = ('amp_mean','phase_mean','freq_mean')
        self.default_samp_titles = ('amp','phase','freq')
        self.ranges = ranges

    def do_sample(self, phi):
        return [(r[1]-r[0])*np.random.beta(2*p, 2-2*p) + r[0]
                for p,r in zip(phi,self.ranges)]

    def pdf(self, x, phi):
        x = [(xx-r[0])/(r[1]-r[0]) for xx,r in zip(x,self.ranges)]
        return np.prod(stats.beta.pdf(np.clip(x,0.001,0.999), 2*phi, 2-2*phi))

    def scale_back(self, x, i):
        r = self.ranges[i]
        return (x-r[0]) / (r[1]-r[0])

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        return np.array([np.clip(np.mean(w*self.scale_back(s[:,i],i))/np.mean(w),
                                 0.001, 0.999)
                         for i in range(3)])

class SineRAML(SineMAML):
    def __init__(self, *args, alpha=0.05, **kwargs):
        valid_fun = lambda x: np.mean(
            sorted(x,reverse=True)[:int(np.round(alpha*len(x)))])
        super(SineRAML, self).__init__(*args, valid_fun=valid_fun, **kwargs)
        self.alpha = alpha
        self.ce_batch_size = 200
        self.ce_sample_count = self.ce_batch_size - 1
        self.ce_samples = None
        self.ce_scores = []
        self.ce = CEM_Beta(batch_size=self.ce_batch_size, min_batch_update=0.2,
                           soft_update=0.5, n_orig_per_batch=0,
                           ref_alpha=self.alpha, ranges=self.ranges)  # w_clip?

    def do_sample_task_with_weight(self, rng, mode):
        if mode == 'train':
            task, w = self.ce.sample()
            return task, w

        return super().do_sample_task_with_weight(rng, mode)

    def update_sampler(self, loss):
        self.ce.update(-loss.item())

    def ce_summary(self, axs=None, n_smooth=None):
        labs = dict(amp='amplitude', phase='phase', freq='frequency')
        if axs is None: axs = utils.Axes(4, 3, fontsize=14)
        a = 0

        c1, c2 = self.ce.get_data()
        for par, (x1,x2) in zip(labs, self.ranges):
            y = (x2-x1)*c1[f'{par}_mean'] + x1
            if n_smooth:
                y = utils.smooth(y, n_smooth)
            axs[a].axhline(0.5*(x1+x2), color='k', linestyle='--')
            axs[a].axhline(x1, color='k', linestyle='--')
            axs[a].axhline(x2, color='k', linestyle='--')
            axs[a].plot(y)
            # axs[a].set_ylim((x1, x2))
            axs.labs(a, 'CEM iteration', f'mean {labs[par]}')
            a += 1

        self.ce.show_sampled_scores(ax=axs[a])
        axs[a].set_ylim((-25, 0))
        axs.labs(a, 'CEM iteration', 'score (=-loss)')
        a += 1

        plt.tight_layout()
        return axs
