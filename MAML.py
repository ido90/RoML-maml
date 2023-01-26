'''
This module implements MAML in pytorch for an abstract problem.
To use the module, the abstract implementation must be inherited with realization
of the abstract methods (noted below).

This implementation of MAML was written as part of a research on the sampling of
tasks in MAML, and thus focuses on flexible access to the source of tasks.

Module structure (minor calls and dependencies may be missing):
MAML:
    train():
        sample_task():
            do_sample_task()            (ABSTRACT)
        get_task_id()                   (ABSTRACT; optional)
        tune_task():
            sample_inputs():
                do_sample_inputs()      (ABSTRACT)
            run_task()                  (ABSTRACT)
        validate():
            evaluate():
                sample_task()           (see above)
                get_task_id()           (see above)
                tune_task()             (see above)
    test():
        evaluate()                      (see above)
    show_task():
        do_show_task()                  (ABSTRACT; optional)
        finalize_task_visualization()   (ABSTRACT; optional)
    analyze()

Experiment:         wrapper of MAML instances with different configurations.

MultiSeedMAML:      wrapper of MAML instances with different seeds.
SeedsExperiment:    wrapper of MultiSeedMAML instances with different configurations.
'''
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
import copy
import torch
from torch import optim
import utils
import wandb


class MAML:
    def __init__(self, model_constructor=None, model_args=None,
                 optim_constructor=optim.Adam, optim_args=None, grad_clip=None,
                 device='cpu', cvar_obj=1, normalize_cvar_loss=False,
                 seed=0, save_best=True, first_order=False, valid_fun=None,
                 meta_train_iterations=40000, warmup_iters=0, train_steps_per_task=1,
                 average_steps_loss=False, validation_horizon=None, lr_sched=None,
                 support_samples=10, query_samples=10, train_tasks_per_batch=1,
                 train_lr=0.01, log_freq=1000, valid_freq=2000, valid_tasks=100,
                 test_steps_per_task=(0, 1, 2, 3, 4, 5), max_test_steps_per_task=10,
                 test_query_samples=32, test_tasks=2000, only_tune_last=0,
                 title='demo', seed_suffix=False,
                 wandb_usage=False, wandb_entity=None, wandb_project=None,
                 wandb_group=None, wandb_run=None,
                 cvar_alphas=(0.05,)):
        self.seed = seed
        self.title = title
        if seed_suffix:
            self.title = f'{self.title}_{self.seed}'

        self.device = 'cpu'
        if device.startswith('cuda') and torch.cuda.is_available():
            self.device = device

        self.first_order_maml = first_order
        self.cvar_obj = cvar_obj
        if valid_fun is None:
            if cvar_obj == 1:
                valid_fun = np.mean
            else:
                valid_fun = lambda x: np.mean(
                    sorted(x, reverse=True)[:int(np.ceil(alpha * len(x)))])
        self.normalize_cvar_loss = normalize_cvar_loss
        self.valid_fun = valid_fun
        self.save_best = save_best

        self.meta_train_iterations = meta_train_iterations
        if 0<warmup_iters<1:
            warmup_iters = int(warmup_iters * meta_train_iterations)
        self.warmup_iters = warmup_iters
        self.average_steps_loss = average_steps_loss
        self.train_steps_per_task = train_steps_per_task
        if validation_horizon is None:
            validation_horizon = train_steps_per_task
        self.validation_horizon = validation_horizon
        self.support_samples = support_samples
        self.query_samples = query_samples
        self.train_tasks_per_batch = train_tasks_per_batch
        self.train_lr = train_lr
        if lr_sched is None:
            lr_sched = lambda lr, i: lr
        self.lr_sched = lr_sched
        self.only_tune_last = only_tune_last
        self.log_freq = log_freq

        self.valid_freq = valid_freq
        self.valid_tasks = valid_tasks

        if validation_horizon not in test_steps_per_task:
            test_steps_per_task = list(test_steps_per_task) + [validation_horizon]
        self.test_steps_per_task = test_steps_per_task  # K-shot
        self.test_steps_set = set(self.test_steps_per_task)
        self.max_valid_steps = np.max(self.test_steps_per_task)
        if max_test_steps_per_task is None:
            max_test_steps_per_task = self.max_valid_steps
        self.max_test_steps = max_test_steps_per_task
        self.test_query_samples = test_query_samples
        self.test_tasks = test_tasks

        self.valid_seed = 1
        self.test_seed = 2

        self.model = None
        self.model_constructor = model_constructor
        if model_args is None: model_args = dict()
        self.model_args = model_args
        self.grad_clip = grad_clip
        self.optim_constructor = optim_constructor
        if optim_args is None: optim_args = dict(lr=1e-3, weight_decay=0.0)
        self.optim_args = optim_args
        self.optimizer = None
        self.meta_steps_count = 0
        self.curr_tune_steps = 0
        self.train_res = None
        self.valid_res = None
        self.test_res = None
        self.cvar_threshs = []
        self.sample_size = []
        self.eff_sample_size = []
        self.plotting_task = False
        if wandb_usage:
            config_args = copy.copy(self.__dict__)
            config_args.pop("test_steps_set")
            self.wandb_logger = wandb.init(reinit=True,
                                           entity=wandb_entity,
                                           project=wandb_project,
                                           group=wandb_group,
                                           name=wandb_run,
                                           config=config_args)
        else:
            self.wandb_logger = None

        self.cvar_funcs = {}
        for alpha in cvar_alphas:
            self.cvar_funcs[f"cvar_{alpha}"] = \
                lambda x: np.mean(-np.sort(-x)[:int(np.ceil(alpha * len(x)))])

    def log(self, text, **kwargs):
        if self.wandb_logger is not None:
            self.wandb_logger.log(text, **kwargs)

    def reset(self):
        utils.set_all_seeds(self.seed)
        self.model = self.model_constructor(**self.model_args).to(self.device)

    def save_model(self, filename=None, path_base='./models'):
        if filename is None: filename = self.title
        fpath = f'{path_base}/{filename}.mdl'
        torch.save(self.model.state_dict(), fpath)

    def load_model(self, filename=None, path_base='./models'):
        if filename is None: filename = self.title
        fpath = f'{path_base}/{filename}.mdl'
        self.model.load_state_dict(torch.load(fpath))
        self.model.reset_model()

    def get_cvar_loss(self, losses, w=None):
        def sum(losses, weights=None):
            loss = 0
            if weights is None:
                for l in losses:
                    loss = loss + l
            else:
                for l, w in zip(losses, weights):
                    loss = loss + w*l
            return loss

        detached_losses = [l.item() for l in losses]
        if self.cvar_obj == 1:
            q = np.min(detached_losses)
            worst_losses = losses
            kept_weights = None
            sample_size = 1
            eff_sample_size = 1
        else:
            q = utils.quantile(detached_losses, 1 - self.cvar_obj, w)
            worst_losses = [l for l in losses if l >= q]
            kept_weights = np.array(
                [ww for ww,l in zip(w, detached_losses) if l >= q])
            kept_weights = kept_weights / np.mean(kept_weights)
            sample_size = len(worst_losses) / len(losses)
            eff_sample_size = np.sum(kept_weights)**2 / \
                              np.sum(kept_weights**2) / len(losses)

        self.cvar_threshs.append(q)
        self.sample_size.append(sample_size)
        self.eff_sample_size.append(eff_sample_size)

        loss = sum(worst_losses, kept_weights)
        if self.normalize_cvar_loss:
            loss = loss / sample_size

        return loss

    def train(self, do_reset=True):
        if do_reset: self.reset()
        o = self.optimizer
        if o is None:
            o = self.optim_constructor(self.model.parameters(), **self.optim_args)

        train_res = {nm: [] for nm in ('meta_iteration', 'task', 'loss')}
        eval_res = None
        self.meta_steps_count = 0
        best_valid_loss = np.inf
        best_iter = 0
        t0 = time.time()

        for iter in range(self.meta_train_iterations):
            # train
            tune_steps = self.train_steps_per_task
            if self.warmup_iters and iter < self.warmup_iters:
                tune_steps = 0
            o.zero_grad()
            losses = []
            ws = []
            for i_task in range(self.train_tasks_per_batch):
                task, w = self.sample_task(mode='train')
                task_loss = self.tune_task(
                    self.model, task, self.query_samples,
                    tune_steps, eval_steps=None, rng=None)
                self.update_sampler(task_loss)

                train_res['meta_iteration'].append(iter)
                train_res['task'].append(self.get_task_id(task))
                train_res['loss'].append(task_loss.item())

                losses.append(task_loss)
                ws.append(w)

            loss = self.get_cvar_loss(losses, ws)

            # Log using WANDB logger
            self.log({'train/loss': loss.item()}, step=self.meta_steps_count)

            # meta step
            loss = loss / self.train_tasks_per_batch
            loss.backward(retain_graph=True)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            o.step()

            # evaluate
            if (iter % self.valid_freq) == 0:
                valid_loss, eval_res = self.validate(self.model, eval_res, t0)
                if valid_loss <= best_valid_loss:
                    if self.save_best:
                        self.save_model()
                    best_valid_loss = valid_loss
                    best_iter = iter

            # log
            if self.log_freq > 0 and iter > 0 and (iter % self.log_freq) == 0:
                start_idx = len(train_res['loss']) - \
                            self.log_freq * self.train_tasks_per_batch
                recent_losses = train_res['loss'][start_idx:]
                recent_losses = np.mean(recent_losses)
                print(f'[{iter:04d}/{self.meta_train_iterations:04d}] '
                      f'loss={recent_losses:.2f}\t({time.time() - t0:.0f} [s])')

            self.meta_steps_count += 1

        # post evaluation
        valid_loss, eval_res = self.validate(self.model, eval_res, t0)
        if valid_loss <= best_valid_loss:
            if self.save_best:
                self.save_model()
            best_iter = self.meta_train_iterations
            best_valid_loss = valid_loss
        print(f'Best model at iteration {best_iter}/{self.meta_train_iterations}, '
              f'best loss = {best_valid_loss:.2f}.')

        # save results
        if self.save_best:
            # save last model before overriding it by the best one
            self.save_model(f'{self.title:s}_n{self.meta_train_iterations:d}')
            # load the best model
            self.load_model()
        else:
            self.save_model()
        self.train_res = self.eval_to_df(train_res)
        if eval_res is not None:
            self.valid_res = self.eval_to_df(eval_res)

    def validate(self, model, eval_res, t0):
        rng = np.random.default_rng(self.valid_seed)
        eval_res = self.evaluate(model, self.valid_tasks, self.query_samples,
                                 rng, eval_res, mode='valid')

        if 'meta_iteration' not in eval_res:
            eval_res['meta_iteration'] = []
        new_data_len = len(eval_res['loss']) - len(eval_res['meta_iteration'])
        eval_res['meta_iteration'].extend(new_data_len * [self.meta_steps_count])

        tmp_df = pd.DataFrame({k: v[len(v) - new_data_len:]
                               for k, v in eval_res.items()})
        valid_loss = self.valid_fun(
            tmp_df.loss[tmp_df.tune_iters == self.validation_horizon])
        print(f'[{self.meta_steps_count:04d}/{self.meta_train_iterations:04d}] '
              f'valid_loss={valid_loss:.2f}\t({time.time() - t0:.0f} [s])')

        if self.wandb_logger is not None:
            # Log each iteration loss using Wandb
            tune_steps = max(eval_res['tune_iters']) + 1
            for tune_iter in range(tune_steps):
                tune_iter_loss = np.arrday([eval_res['loss'][i]
                                            for i in range(len(eval_res['loss'])) if
                                            eval_res['tune_iters'][i] == tune_iter])
                # tune_iter_loss = self.valid_fun(tune_iter_loss)
                mean_tune_iter_loss = np.mean(tune_iter_loss)
                self.log(
                    {f'validation/iter_{tune_iter}_mean_loss': mean_tune_iter_loss},
                    step=self.meta_steps_count)
                for cvar_name, cvar_func in self.cvar_funcs.items():
                    cvar_loss = cvar_func(tune_iter_loss)
                    self.log(
                        {f'validation/iter_{tune_iter}_{cvar_name}_loss': cvar_loss},
                        step=self.meta_steps_count)

            self.log({f'validation/loss': valid_loss}, step=self.meta_steps_count)
            for cvar_name, cvar_func in self.cvar_funcs.items():
                cvar_loss = cvar_func(tune_iter_loss)
                self.log({f'validation/{cvar_name}_loss': cvar_loss},
                         step=self.meta_steps_count)

        return valid_loss, eval_res

    def test(self):
        rng = np.random.default_rng(self.test_seed)
        res = self.evaluate(self.model, self.test_tasks, self.test_query_samples,
                            rng, save_all=True, mode='test')
        self.test_res = self.eval_to_df(res)

        if self.wandb_logger is not None:
            # Log each iteration loss using Wandb
            test_count = self.meta_steps_count + 1
            tune_steps = max(res['tune_iters']) + 1
            for tune_iter in range(tune_steps):
                self.log({f'test/tune_iter': tune_iter}, step=test_count)
                tune_iter_loss = [res['loss'][i] for i in range(len(res['loss'])) if
                                  res['tune_iters'][i] == tune_iter]
                mean_tune_iter_loss = np.mean(tune_iter_loss)
                self.log({f'test/mean_loss': mean_tune_iter_loss}, step=test_count)
                for cvar_name, cvar_func in self.cvar_funcs.items():
                    cvar_loss = cvar_func(tune_iter_loss)
                    self.log({f'test/{cvar_name}_loss': cvar_loss}, step=test_count)
                test_count += 1

    def evaluate(self, model, n_tasks, n_query_samples, rng=None, eval_res=None,
                 save_all=False, mode='test'):
        steps_set = set(list(range(self.max_test_steps + 1))) \
            if save_all else self.test_steps_set
        max_steps = np.max(list(steps_set))
        for i_task in range(n_tasks):
            task = self.sample_task(rng, mode)[0]
            eval_res = self.tune_task(
                model, task, n_query_samples, max_steps, steps_set,
                rng, eval_res)
        return eval_res

    def tune_task(self, model, task, n_query_samples, total_steps,
                  eval_steps=None, rng=None, eval_res=None, check_nans=False):
        '''
        Fine-tune model on task for total_steps.
        If eval_steps is a list - evaluate model after the specified tuning steps.
        If eval_steps is None - evaluate only once in the end (as in training).
        Support_samples are used for fine-tuning and query_samples for evaluation.
        '''
        is_eval = True
        average_loss = 0
        if eval_steps is None:
            is_eval = False
            if self.average_steps_loss and total_steps>0:
                eval_steps = set(range(1,total_steps+1))
            else:
                eval_steps = set()
        elif eval_res is None:
            eval_res = {nm: [] for nm in ('task', 'tune_iters', 'loss')}

        # reset model's temporary weights
        self.model.reset_model()

        # get task samples
        support_samples = self.sample_inputs(task, self.support_samples, rng)
        query_samples = self.sample_inputs(task, n_query_samples, rng)

        # run task evaluations
        if 0 in eval_steps:
            model.eval()
            eval_loss = self.run_task(model, task, query_samples, is_eval, False)
            if isinstance(eval_loss, torch.Tensor):
                eval_loss = eval_loss.item()
            model.train()
            eval_res['tune_iters'].append(0)
            eval_res['loss'].append(eval_loss)

        self.curr_tune_steps = 0
        for task_iter in range(total_steps):
            # run model
            task_loss = self.run_task(model, task, support_samples, is_eval, True)

            # update task model
            lr = self.lr_sched(self.train_lr, task_iter)
            model.update_current_weights(
                task_loss, lr, self.first_order_maml, self.grad_clip,
                self.only_tune_last)

            # evaluate
            if (task_iter + 1) in eval_steps:
                model.eval()
                eval_loss = self.run_task(
                    model, task, query_samples, is_eval, False)
                model.train()

                if is_eval:
                    if isinstance(eval_loss, torch.Tensor):
                        eval_loss = eval_loss.item()
                    eval_res['tune_iters'].append(task_iter + 1)
                    eval_res['loss'].append(eval_loss)
                else:
                    average_loss = average_loss + eval_loss

            self.curr_tune_steps += 1

        if not is_eval:
            if self.average_steps_loss and total_steps>0:
                average_loss = average_loss / total_steps
                if check_nans and torch.isnan(average_loss):
                    print('averaged loss is nan', self.meta_steps_count, task, total_steps)
                    import pdb
                    pdb.set_trace()
                return average_loss
            # no specified evaluation steps - just return the final loss
            loss = self.run_task(model, task, query_samples, is_eval, False)
            if check_nans and torch.isnan(loss):
                print('loss is nan', self.meta_steps_count, task, total_steps)
                import pdb
                pdb.set_trace()
            return loss

        eval_res['task'].extend(
            (len(eval_res['loss']) - len(eval_res['task'])) * \
            [self.get_task_id(task)])

        return eval_res

    def show_task(self, task=None, n_query_samples=None, total_steps=None,
                  eval_steps=None, rng=None, ax=None):
        if task is None: task = self.sample_task(rng, mode='test')[0]
        if n_query_samples is None: n_query_samples = self.query_samples
        if eval_steps is None: eval_steps = set()
        if total_steps is None:
            total_steps = np.max(eval_steps) if eval_steps else 0
        if ax is None: ax = utils.Axes(1, 1)[0]
        model = self.model
        self.plotting_task = False

        # reset model's temporary weights
        self.model.reset_model()

        # get task samples
        support_samples = self.sample_inputs(task, self.support_samples, rng)
        query_samples = self.sample_inputs(task, n_query_samples, rng)

        # run task evaluations
        if 0 in eval_steps:
            model.eval()
            self.do_show_task(model, task, support_samples, query_samples, 0, ax)
            self.plotting_task = True
            model.train()

        self.curr_tune_steps = 0
        for task_iter in range(total_steps):
            # run model
            task_loss = self.run_task(model, task, support_samples, True, True)

            # update task model
            lr = self.lr_sched(self.train_lr, task_iter)
            model.update_current_weights(
                task_loss, lr, self.first_order_maml, self.grad_clip,
                self.only_tune_last)

            # evaluate
            if (task_iter + 1) in eval_steps:
                model.eval()
                self.do_show_task(model, task, support_samples, query_samples,
                                  task_iter + 1, ax)
                self.plotting_task = True
                model.train()

            self.curr_tune_steps += 1

        if not eval_steps:
            # no specified evaluation steps - just show the final step
            model.eval()
            self.do_show_task(model, task, support_samples, query_samples,
                              total_steps, ax)
            model.train()

        self.finalize_task_visualization(ax)
        self.plotting_task = False

    def sample_task(self, rng=None, mode='test'):
        if rng is None:
            rng = np.random
        return self.do_sample_task_with_weight(rng, mode)

    def do_sample_task_with_weight(self, rng, mode):
        return self.do_sample_task(rng, mode), 1

    def do_sample_task(self, rng, mode):
        raise NotImplementedError()

    def update_sampler(self, loss):
        return

    def get_task_id(self, task):
        return 'unknown'

    def sample_inputs(self, task, n, rng=None):
        if rng is None:
            rng = np.random
        return self.do_sample_inputs(task, n, rng)

    def do_sample_inputs(self, task, n, rng):
        return None

    def run_task(self, model, task, samples, is_evaluation, is_finetuning):
        '''Run self.train_samples_per_task samples and return the task loss.'''
        # Note: both in training and in evaluation, we first fine-tune over
        #       support samples and then apply to query samples.
        #       Thus, is_evaluation and is_finetuning indicate which case
        #       we're currently in.
        raise NotImplementedError()

    def do_show_task(self, model, task, support_samples, query_samples, step, ax):
        '''Visualize task.'''
        return

    def finalize_task_visualization(self, ax):
        return

    def eval_to_df(self, eval_res):
        eval_res['model'] = self.title
        return pd.DataFrame(eval_res)

    def analyze(self, axs=None, axsize=(5, 3.5), train_resolution=250, max_loss=None,
                est='mean', logscale=True, show_early_validations=False,
                show_validation_with_train=True, show_train_ci=False):
        if axs is None: axs = utils.Axes(6, 3, axsize, fontsize=15)
        a = 0

        estimator, loss_lab = process_estimator(est)

        # Train loss vs. meta-iteration
        if self.train_res is not None:
            rr = self.train_res.copy()
            rr['group'] = 'train'
            rr['rolling_loss'] = rr.loss.rolling(train_resolution).mean()
            # add validation results
            if show_validation_with_train:
                rr2 = self.valid_res.copy()[
                    self.valid_res.tune_iters == self.validation_horizon]
                rr2['group'] = 'valid'
                rr2['rolling_loss'] = rr2.loss
                rr = pd.concat((rr, rr2))
                rr.reset_index(drop=True, inplace=True)
            sns.lineplot(data=rr, x='meta_iteration', y='rolling_loss', hue='group',
                         ax=axs[a], ci=95 if show_train_ci else None)
            if logscale: axs[a].set_yscale('log')
            clip_ylim(axs[a], max_loss, None)
            axs.labs(a, 'meta iteration', f'loss ({self.validation_horizon}-shot)',
                     'train results')
        a += 1

        # Validation loss vs. tuning-iteration and meta-iteration
        rr = self.valid_res
        if self.valid_res is not None:
            valid_steps = [i for i in self.test_steps_per_task if i != 0]
            n_obs = len(valid_steps)
            if n_obs > 3:
                valid_steps = np.take(
                    valid_steps, np.round(np.linspace(0, n_obs - 1, 3)).astype(int))
            sns.lineplot(data=rr[rr.tune_iters.isin(valid_steps)],
                         hue='tune_iters', y='loss', x='meta_iteration',
                         estimator=estimator, ax=axs[a])
            # axs[a].set_yscale('log')
            clip_ylim(axs[a], max_loss)
            axs[a].legend(title='tuning iteration', fontsize=13, title_fontsize=13)
            axs.labs(a, 'meta iteration', loss_lab, 'validation results')
        a += 1

        if self.valid_res is not None:
            valid_iterations = pd.unique(self.valid_res.meta_iteration)
            n_obs = len(valid_iterations)
            if n_obs > 3:
                if show_early_validations:
                    valid_iterations = np.take(
                        valid_iterations,
                        np.round(np.linspace(0,n_obs-1,3)).astype(int))
                else:
                    valid_iterations = valid_iterations[-3:]
            sns.lineplot(data=rr[rr.meta_iteration.isin(valid_iterations)],
                         x='tune_iters', y='loss', hue='meta_iteration',
                         estimator=estimator, ax=axs[a])
            # axs[a].set_yscale('log')
            clip_ylim(axs[a], max_loss)
            axs[a].legend(title='meta iteration', fontsize=13, title_fontsize=13)
            axs.labs(a, 'tuning iteration', loss_lab, 'validation results')
        a += 1

        # Test loss vs. tuning-iteration + test loss distribution
        if self.test_res is not None:
            sns.lineplot(data=self.test_res, x='tune_iters', y='loss',
                         estimator=estimator, ax=axs[a])
            clip_ylim(axs[a], max_loss)
            axs.labs(a, 'tuning iteration', loss_lab, 'test results')
            a += 1

            test_res = self.test_res[
                self.test_res.tune_iters == self.validation_horizon]
            utils.plot_quantiles(test_res.loss, ax=axs[a])
            axs.labs(a, 'task quantile [%]',
                     f'loss ({self.validation_horizon}-shot)', 'test results')
            a += 1

            self.show_task(rng=np.random.default_rng(self.valid_seed), ax=axs[a],
                           eval_steps=list(self.test_steps_per_task))
            a += 1

        plt.tight_layout()
        return axs


def clip_ylim(ax, max_y=None, min_y=None):
    if max_y is not None:
        ax.set_ylim((None, min(ax.get_ylim()[1], max_y)))
    if min_y is not None:
        ax.set_ylim((max(ax.get_ylim()[0], min_y), None))


def process_estimator(est_name='mean'):
    if est_name == 'mean':
        est = np.mean
        est_name = f'{est_name} loss'
    elif est_name.startswith('cvar'):
        alpha = float(est_name[len('cvar'):]) / 100
        est = lambda x: np.mean(-np.sort(-x)[:int(np.ceil(alpha * len(x)))])
        est_name = f'CVaR$_{{{alpha:.2f}}}$ loss'
    else:
        raise ValueError(est_name)
    return est, est_name


def analysis_preprocessing(mamls, train_resolution=250):
    train_res = pd.DataFrame()
    for M in mamls:
        if M.train_res is None: continue
        rr = M.train_res.copy()
        rr['group'] = 'train'
        rr['rolling_loss'] = rr.loss.rolling(train_resolution).mean()
        rr['tuning_steps'] = M.train_steps_per_task
        # add validation results
        rr2 = M.valid_res.copy()[
            M.valid_res.tune_iters == M.validation_horizon]
        rr2['group'] = 'valid'
        rr2['rolling_loss'] = rr2.loss
        rr2['tuning_steps'] = M.validation_horizon
        rr = pd.concat((rr, rr2))
        rr['model'] = M.title
        train_res = pd.concat((train_res, rr))
    train_res.reset_index(drop=True, inplace=True)

    valid_res = pd.DataFrame()
    for M in mamls:
        if M.valid_res is None: continue
        rr = M.valid_res.copy()
        rr['model'] = M.title
        valid_res = pd.concat((valid_res, rr))
    valid_res.reset_index(drop=True, inplace=True)

    test_res = pd.DataFrame()
    for M in mamls:
        if M.test_res is None: continue
        rr = M.test_res.copy()
        rr['model'] = M.title
        test_res = pd.concat((test_res, rr))
    test_res.reset_index(drop=True, inplace=True)

    return train_res, valid_res, test_res


def analyze(mamls, axs=None, axsize=(6, 4), train_resolution=250, max_loss=None,
            est='mean', logscale=True, show_early_validations=False,
            show_validation_with_train=True, task_fun=None, test_horizon=None,
            show_train_ci=False):
    analyze_per_task = task_fun is not None
    if axs is None: axs = utils.Axes(6+analyze_per_task, 3, axsize, fontsize=15)
    a = 0

    estimator, loss_lab = process_estimator(est)
    train_res, valid_res, test_res = analysis_preprocessing(mamls, train_resolution)

    # Train loss vs. meta-iteration
    tr = train_res
    if not show_validation_with_train:
        tr = tr[tr.group=='train']
    sns.lineplot(data=tr, x='meta_iteration', y='rolling_loss', hue='model',
                 style='group', ax=axs[a], ci=95 if show_train_ci else None)
    if logscale: axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss, None)
    ylab = 'loss'
    if len(pd.unique(train_res.tuning_steps)) == 1:
        ylab = f'loss ({train_res.tuning_steps.values[0]:d}-shot)'
    axs.labs(a, 'meta iteration', ylab, 'train results')
    a += 1

    # Validation loss vs. meta-iteration
    sns.lineplot(data=train_res[train_res.group == 'valid'], x='meta_iteration',
                 y='rolling_loss', hue='model', ax=axs[a])
    if logscale: axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss, None)
    ylab = 'loss'
    if len(pd.unique(train_res.tuning_steps)) == 1:
        ylab = f'loss ({train_res.tuning_steps.values[0]:d}-shot)'
    axs.labs(a, 'meta iteration', ylab, 'validation results')
    a += 1

    # Validation loss vs. tuning-iteration and meta-iteration
    valid_steps = pd.unique(valid_res.tune_iters)
    n_obs = len(valid_steps)
    if n_obs > 3:
        valid_steps = np.take(
            valid_steps, np.round(np.linspace(0, n_obs - 1, 3)).astype(int))
    sns.lineplot(data=valid_res[valid_res.tune_iters.isin(valid_steps)],
                 style='tune_iters', y='loss', x='meta_iteration', hue='model',
                 estimator=estimator, ci=None, ax=axs[a])
    # axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss)
    axs[a].legend(title='tuning iteration', fontsize=12, title_fontsize=12)
    axs.labs(a, 'meta iteration', loss_lab, 'validation results')
    a += 1

    valid_iterations = sorted(pd.unique(valid_res.meta_iteration))
    n_obs = len(valid_iterations)
    if n_obs > 3:
        if show_early_validations:
            valid_iterations = np.take(
                valid_iterations, np.round(np.linspace(0,n_obs-1,3)).astype(int))
        else:
            valid_iterations = valid_iterations[-3:]
    sns.lineplot(data=valid_res[valid_res.meta_iteration.isin(valid_iterations)],
                 x='tune_iters', y='loss', style='meta_iteration', hue='model',
                 estimator=estimator, ci=None, ax=axs[a])
    # axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss)
    axs[a].legend(title='meta iteration', fontsize=12, title_fontsize=12)
    axs.labs(a, 'tuning iteration', loss_lab, 'validation results')
    a += 1

    # Test loss vs. tuning-iteration
    sns.lineplot(data=test_res, x='tune_iters', y='loss', hue='model',
                 estimator=estimator, ax=axs[a])
    clip_ylim(axs[a], max_loss)
    axs.labs(a, 'tuning iteration', loss_lab, 'test results')
    a += 1

    # # this is irrelevant unless tasks are discrete and resampled over and over
    # # Test task-loss vs. tuning-iteration
    # def agg_tasks(dd):
    #     dd['loss'] = dd.loss.mean()
    #     return dd.iloc[:1,:]
    # task_res = test_res.groupby(['model','tune_iters','task'], sort=False
    #                             ).apply(agg_tasks)
    # task_res.reset_index(drop=True, inplace=True)
    # sns.lineplot(data=task_res, x='tune_iters', y='loss', hue='model',
    #              estimator=estimator, ci=None, ax=axs[a])
    # clip_ylim(axs[a], max_loss)
    # axs.labs(a, 'tuning iteration', loss_lab[:-4]+'task-loss',
    #          'test results')
    # a += 1

    # Test loss distribution
    if test_horizon is None:
        test_horizon = train_res.tuning_steps.max()
    utils.qplot(test_res[test_res.tune_iters==test_horizon], 'loss', 'task',
                'model', axs[a])
    axs.labs(a, None, f'loss ({test_horizon}-shot)')
    a += 1

    # Analyze per task
    if analyze_per_task:
        test_res['task_val'] = [task_fun(t) for t in test_res.task]
        for m in pd.unique(test_res.model):
            tt = test_res[
                (test_res.model==m)&(test_res.tune_iters==test_horizon)]
            axs[a].plot(tt.task_val, tt.loss, '.', label=m)
        axs.labs(a, 'task', f'loss ({test_horizon:d}-shot)', fontsize=15)
        axs[a].legend(fontsize=13)
        a += 1

    plt.tight_layout()
    return axs


def visualize(mamls, axs=None, axsize=(4,3.5), fun='visualize_policy'):
    n = len(mamls)
    if axs is None: axs = utils.Axes(n, 6, axsize=axsize)
    for a, m in enumerate(mamls):
        getattr(m, fun)(ax=axs[a])
        axs[a].set_title(m.title)
    plt.tight_layout()
    return axs


class Experiment:
    def __init__(self, constructors, args, common_args):
        for k in args:
            if 'title' not in args[k]:
                args[k]['title'] = k
        if not isinstance(constructors, (tuple, list, dict)):
            constructors = {k: constructors for k in args}

        self.common_args = common_args
        self.args = args
        self.constructors = constructors

        self.keys = list(self.args.keys())
        self.mamls = {}
        self.construct_models()

    def _get_keys(self, keys=None):
        if keys is None:
            keys = self.keys
        if isinstance(keys, str):
            keys = (keys,)
        return keys

    def construct_models(self, keys=None):
        keys = self._get_keys(keys)
        for k in keys:
            args = utils.update_dict(
                self.args[k], self.common_args, force=False, copy=True)
            self.mamls[k] = self.constructors[k](**args)

    def add_model(self, nm, constructor, args):
        if nm in self.keys:
            warnings.warn(f'Model name {nm} already in keys.')
        if 'title' not in args:
            args['title'] = nm
        self.keys.append(nm)
        self.constructors[nm] = constructor
        self.args[nm] = args
        self.construct_models(nm)

    def train(self, keys=None, **kwargs):
        keys = self._get_keys(keys)
        for k in keys:
            print(f'\nTraining {k}...')
            self.mamls[k].train(**kwargs)

    def test(self, keys=None, **kwargs):
        keys = self._get_keys(keys)
        for k in keys:
            print(f'Testing {k}...', end='')
            t0 = time.time()
            self.mamls[k].test(**kwargs)
            print(f' done ({time.time()-t0:.0f} [s])')

    def analyze_each(self, keys=None, **kwargs):
        keys = self._get_keys(keys)
        for k in keys:
            self.mamls[k].analyze(**kwargs)

    def analyze_all(self, keys=None, **kwargs):
        keys = self._get_keys(keys)
        mamls = [self.mamls[k] for k in keys]
        return analyze(mamls, **kwargs)

    def show_all(self, keys=None, axs=None, **kwargs):
        keys = self._get_keys(keys)
        mamls = [self.mamls[k] for k in keys]
        if axs is None:
            axs = utils.Axes(len(mamls), 4, grid=0)

        for a, m in enumerate(mamls):
            m.show_task(ax=axs[a], **kwargs)
        plt.tight_layout()
        return axs

    def visualize_all(self, keys=None, fun='visualize_policy', **kwargs):
        keys = self._get_keys(keys)
        mamls = [self.mamls[k] for k in keys]
        return visualize(mamls, fun=fun, **kwargs)


class Wandb_Experiment:
    def __init__(self, constructors, args, common_args):
        for k in args:
            if 'title' not in args[k]:
                args[k]['title'] = k
        if not isinstance(constructors, (tuple, list, dict)):
            constructors = {k: constructors for k in args}

        self.common_args = common_args
        self.args = args
        self.constructors = constructors

    def initiate_objects_train_and_test(self, **kwargs):
        for k in self.args:
            maml = self.constructors[k](**self.args[k], **self.common_args)
            print(f'\nTraining {k}...')
            maml.train(**kwargs['train'])
            print(f'Testing {k}...')
            maml.test(**kwargs['test'])

    # def analyze_each(self, **kwargs):
    #     for k, M in self.mamls.items():
    #         M.analyze(**kwargs)
    #
    # def analyze_all(self, **kwargs):
    #     return analyze(list(self.mamls.values()), **kwargs)


################################################################
###############     Analysis over Seeds     ####################
################################################################


class MultiSeedMAML:
    def __init__(self, constructor, n_seeds=5, seeds=None, *args, **kwargs):
        if seeds is None: seeds = list(range(n_seeds))
        self.n_seeds = n_seeds
        self.seeds = seeds
        self.mamls = [constructor(*args, seed=seed, seed_suffix=True, **kwargs)
                      for seed in self.seeds]
        self.res = dict()

    def train(self, *args, **kwargs):
        for M in self.mamls:
            print(f'\nTraining seed {M.seed}...')
            M.train(*args, **kwargs)

    def test(self, *args, **kwargs):
        for M in self.mamls:
            print(f'Testing seed {M.seed}...')
            M.test(*args, **kwargs)

    def summarize_stats_per_seed(self, train_resolution=250, est='mean', verbose=0):
        estimator, loss_lab = process_estimator(est)
        train_res, valid_res, test_res = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        t0 = time.time()

        for seed, M in zip(self.seeds, self.mamls):
            train_r, valid_r, test_r = analysis_preprocessing(
                [M], train_resolution)
            title = M.title[:M.title.rfind('_')]

            if M.train_tasks_per_batch == 1:
                train_r = pd.DataFrame(dict(
                    model=title,
                    tuning_steps=train_r.tuning_steps,
                    group=train_r.group,
                    meta_iteration=train_r.meta_iteration,
                    loss=train_r.loss,
                    rolling_loss=train_r.rolling_loss
                ))
            else:
                train_r = train_r.groupby(['group', 'meta_iteration']).apply(
                    lambda d: pd.DataFrame(dict(
                        model=[title],
                        tuning_steps=d.tuning_steps.values[0],
                        group=d.group.values[0],
                        meta_iteration=d.meta_iteration.values[0],
                        loss=estimator(d.loss.values),
                        rolling_loss=estimator(d.rolling_loss.values)
                    )))

            valid_r = valid_r.groupby(['meta_iteration', 'tune_iters']).apply(
                lambda d: pd.DataFrame(dict(
                    model=[title],
                    meta_iteration=d.meta_iteration.values[0],
                    tune_iters=d.tune_iters.values[0],
                    loss=estimator(d.loss.values)
                )))

            test_r = test_r.groupby(['tune_iters']).apply(
                lambda d: pd.DataFrame(dict(
                    model=[title],
                    tune_iters=d.tune_iters.values[0],
                    loss=estimator(d.loss.values)
                )))

            train_r['seed'] = seed
            valid_r['seed'] = seed
            test_r['seed'] = seed

            train_res = pd.concat((train_res, train_r))
            valid_res = pd.concat((valid_res, valid_r))
            test_res = pd.concat((test_res, test_r))

            if verbose >= 1:
                print(f'{M.title} aggregated.\t({time.time() - t0:.0f}s)')

        train_res.reset_index(drop=True, inplace=True)
        valid_res.reset_index(drop=True, inplace=True)
        test_res.reset_index(drop=True, inplace=True)

        self.res[est] = train_res, valid_res, test_res
        return self.res[est], loss_lab


def analyze_over_seeds(mamls, axs=None, axsize=(6, 4), train_resolution=250,
                       max_loss=None, estimator='mean', force_preprocessing=False,
                       max_test_iters=None, show_early_validations=False, res=None):
    if axs is None: axs = utils.Axes(6, 3, axsize, fontsize=15)
    a = 0

    # Aggregate data
    if res is None:
        train_res, valid_res, test_res = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        loss_lab = 'mean loss'
        for M in mamls:
            if estimator not in M.res or force_preprocessing:
                (train_r, valid_r, test_r), loss_lab = M.summarize_stats_per_seed(
                    train_resolution, estimator)
            else:
                train_r, valid_r, test_r = M.res[estimator]
                loss_lab = process_estimator(estimator)[1]
            train_res = pd.concat((train_res, train_r))
            valid_res = pd.concat((valid_res, valid_r))
            test_res = pd.concat((test_res, test_r))
        train_res.reset_index(drop=True, inplace=True)
        valid_res.reset_index(drop=True, inplace=True)
        test_res.reset_index(drop=True, inplace=True)
        res = train_res, valid_res, test_res, loss_lab
    else:
        train_res, valid_res, test_res, loss_lab = res


    # Train loss vs. meta-iteration
    sns.lineplot(data=train_res, x='meta_iteration', y='rolling_loss', hue='model',
                 style='group', ci=None, ax=axs[a])
    axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss, None)
    ylab = 'loss'
    if len(pd.unique(train_res.tuning_steps)) == 1:
        ylab = f'loss ({train_res.tuning_steps.values[0]:d}-shot)'
    axs.labs(a, 'meta iteration', ylab, 'train results')
    a += 1

    # Valid loss vs. meta-iteration
    sns.lineplot(data=train_res[train_res.group == 'valid'], x='meta_iteration',
                 y='rolling_loss', hue='model', ci=None, ax=axs[a])
    axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss, None)
    ylab = 'loss'
    if len(pd.unique(train_res.tuning_steps)) == 1:
        ylab = f'loss ({train_res.tuning_steps.values[0]:d}-shot)'
    axs.labs(a, 'meta iteration', ylab, 'validation results')
    a += 1

    # Validation loss vs. tuning-iteration and meta-iteration
    valid_steps = [i for i in pd.unique(valid_res.tune_iters) if i != 0]
    n_obs = len(valid_steps)
    if n_obs > 3:
        valid_steps = np.take(
            valid_steps, np.round(np.linspace(0, n_obs - 1, 3)).astype(int))
    sns.lineplot(data=valid_res[valid_res.tune_iters.isin(valid_steps)],
                 style='tune_iters', y='loss', x='meta_iteration', hue='model',
                 ci=None, ax=axs[a])
    # axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss)
    axs[a].legend(title='tuning iteration', fontsize=12, title_fontsize=12)
    axs.labs(a, 'meta iteration', loss_lab, 'validation results')
    a += 1

    valid_iterations = sorted(pd.unique(valid_res.meta_iteration))
    n_obs = len(valid_iterations)
    if n_obs > 3:
        if show_early_validations:
            valid_iterations = np.take(
                valid_iterations, np.round(np.linspace(0,n_obs-1,3)).astype(int))
        else:
            valid_iterations = valid_iterations[-3:]
    sns.lineplot(data=valid_res[valid_res.meta_iteration.isin(valid_iterations)],
                 x='tune_iters', y='loss', style='meta_iteration', hue='model',
                 ci=None, ax=axs[a])
    # axs[a].set_yscale('log')
    clip_ylim(axs[a], max_loss)
    axs[a].legend(title='meta iteration', fontsize=12, title_fontsize=12)
    axs.labs(a, 'tuning iteration', loss_lab, 'validation results')
    a += 1

    # Test loss vs. tuning-iteration
    if max_test_iters is None:
        max_test_iters = test_res.tune_iters.max()
    sns.lineplot(data=test_res[test_res.tune_iters<=max_test_iters],
                 x='tune_iters', y='loss', hue='model', ax=axs[a])
    clip_ylim(axs[a], max_loss)
    axs[a].set_xticks(list(range(0, max_test_iters+1)))
    # axs[a].set_xticklabels()
    axs.labs(a, 'tuning iteration', loss_lab, 'test results')
    a += 1

    # Test loss distribution over seeds
    utils.qplot(test_res[test_res.tune_iters==max_test_iters],
                'loss', 'seed', 'model', axs[a])
    axs.labs(a, 'seed quantile [%]', loss_lab, 'test results')
    a += 1

    plt.tight_layout()
    return axs, res


class SeedsExperiment:
    def __init__(self, args, common_args):
        for k in args:
            if 'title' not in args[k]:
                args[k]['title'] = k

        self.common_args = common_args
        self.args = args

        self.mamls = {k: MultiSeedMAML(**self.args[k], **self.common_args)
                      for k in self.args}
        self.res = {}

    def train(self, **kwargs):
        for k, M in self.mamls.items():
            print(f'\nTraining {k}...')
            M.train(**kwargs)

    def test(self, **kwargs):
        for k, M in self.mamls.items():
            print(f'Testing {k}...')
            M.test(**kwargs)

    def analyze(self, keys=None, estimator='mean', recalculate=False, **kwargs):
        if 'res' not in kwargs:
            kwargs['res'] = None if recalculate else self.res[estimator]
        if 'force_preprocessing' not in kwargs:
            kwargs['force_preprocessing'] = recalculate
        if keys is None:
            keys = list(self.mamls.keys())
        else:
            kwargs['res'] = [r if isinstance(r,str) else r[r.model.isin(keys)]
                             for r in kwargs['res']]
        mamls = [self.mamls[k] for k in keys]

        out = analyze_over_seeds(mamls, estimator=estimator, **kwargs)
        self.res[estimator] = out[1]
        return out
