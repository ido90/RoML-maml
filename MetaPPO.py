'''
To exploit stable-baselines infra as part of our MAML pipeline, we need the following:
1. The MAML pipeline receives the loss of a single "task" (env configuration/objective
   in the RL case) and uses it for learning or meta-learning as needed. Thus, the PPO
   train() method must return the loss rather than updating the weights by itself.
   We override its train() method below accordingly.
2. For MAML, the actor-critic policy itself must rely on a basic meta-learning module
   (which permits forward step with fine-tuned params without modifying the baselines
   network params) - for both actor and critic. We do this by letting MetaPolicy receive
   as arguments the constructors of the actor and the critic.
'''

import warnings
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
import torch
import gym
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.ppo import ppo
from stable_baselines3.common import env_util, policies, buffers
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

import BaseLearner
import MAML
import utils


class MetaPPO(ppo.PPO):
    def __init__(self, *args, n_buffers=2, assume_aligned_episodes=False, **kwargs):
        self.assume_aligned_episodes = assume_aligned_episodes
        self.n_buffers = n_buffers
        self.all_buffers = None
        super().__init__(*args, **kwargs)
        self.init_rollout_buffers()

    def init_rollout_buffers(self):
        buffer_cls = buffers.DictRolloutBuffer if isinstance(
            self.observation_space, gym.spaces.Dict) else buffers.RolloutBuffer

        self.all_buffers = [buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        ) for _ in range(self.n_buffers)]

    def train(self, vf_coef=None):
        """
        Update policy using the currently gathered rollout buffer.
        Unlike the standard PPO class, for MetaPPO we (1) limit the train
        step to a single epoch (multiple epochs require multiple external
        calls); (2) return the loss rather than updating the model.
        """

        if vf_coef is None: vf_coef = self.vf_coef

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        clip_range_vf = 0
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train step
        n_batches = 0
        tot_loss = 0
        # Do a complete pass on the rollout buffer
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the different between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            n_batches += 1
            loss = policy_loss + self.ent_coef * entropy_loss + vf_coef * value_loss
            tot_loss = tot_loss + loss

        tot_loss = tot_loss / n_batches
        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", tot_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        return tot_loss

    def get_rollouts_returns(self):
        '''Get the returns (rather than lossess) of the recent rollout episodes.
           Ignore incomplete prefixes/suffixes from previous rollouts.'''
        buff = self.rollout_buffer
        if len(buff.rewards.shape) == 1:
            # single env
            starts_ids = np.where(buff.episode_starts==1)[0]
            if self.assume_aligned_episodes:
                starts_ids = np.concatenate((starts_ids, [len(buff.episode_starts)]))
            # Note that we exclude rewards[:ids[0]] and rewards[ids[-1]:], which represent
            #  possibly-incomplete episodes.
            return np.array([np.sum(buff.rewards[i1:i2])
                             for i1,i2 in zip(starts_ids[:-1],starts_ids[1:])])
        else:
            # multi env
            rets = []
            for i in range(buff.rewards.shape[1]):
                starts_ids = np.where(buff.episode_starts[:,i]==1)[0]
                if self.assume_aligned_episodes:
                    starts_ids = np.concatenate((starts_ids, [len(buff.episode_starts)]))
                rets.extend([np.sum(buff.rewards[i1:i2, i])
                             for i1,i2 in zip(starts_ids[:-1],starts_ids[1:])])
            return np.array(rets)


###############################################################################

class MetaPolicy(policies.ActorCriticPolicy):
    def __init__(self, *args, actor_args, critic_args,
                 actor_const=BaseLearner.Learner,
                 critic_const=BaseLearner.Learner, **kwargs):
        # Each config should be a tuple (constructor, kwargs).
        # The constructor must be (a variant of) BaseLearning.Learner.
        self.actor_constructor = actor_const
        self.actor_args = actor_args
        self.critic_constructor = critic_const
        self.critic_args = critic_args

        self.deterministic = False
        self.eps = 0.0
        super(MetaPolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.actor = self.actor_constructor(**self.actor_args)
        self.critic = self.critic_constructor(**self.critic_args)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.log_std = torch.nn.Parameter(torch.ones(self.action_dist.action_dim) * \
                                              self.log_std_init, requires_grad=True)
            self.curr_log_std = None

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def reset_model(self):
        self.actor.current_vars = None
        self.critic.current_vars = None
        self.curr_log_std = None

    def update_current_weights(self, loss, lr, first_order=False, grad_clip=None):
        # "First order": the first calls must retain_graph for the last call.
        # The last call should retain_graph only if we use 2nd-order MAML.
        if hasattr(self, "log_std"):
            log_std = self.log_std if self.curr_log_std is None else self.curr_log_std
            self.curr_log_std = BaseLearner.update_weights(
                loss, [log_std], lr, False, grad_clip)[0]
        self.actor.update_current_weights(loss, lr, False, grad_clip)
        self.critic.update_current_weights(loss, lr, first_order, grad_clip)

    def forward(self, obs: torch.Tensor, deterministic=None, eps=None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        if eps is None:
            eps = self.eps
        if deterministic is None:
            deterministic = self.deterministic

        values = self.critic(obs)
        distribution = self._get_action_dist(obs)
        if eps > 0 and np.random.random() < eps:
            # TODO note: we apply the epsilon-randomness uniformly to all envs,
            #  which is probably not the best idea.
            actions = torch.randint(0, distribution.action_dim, [obs.shape[0]])
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist(self, obs: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the observation.

        :param obs: observation
        :return: Action distribution
        """
        mean_actions = self.actor(obs)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std if self.curr_log_std is None else self.curr_log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding for binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions,
                self.log_std if self.curr_log_std is None else self.curr_log_std, obs)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: torch.Tensor, deterministic: bool = False
                 ) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        values = self.critic(obs)
        distribution = self._get_action_dist(obs)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        return self._get_action_dist(obs)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


###############################################################################

class MetaRL(MAML.MAML):
    def __init__(self, ppo_args=None, policy_args=None, env_args=None,
                 env_name=None, n_envs=1, attributes=None, meta=True,
                 updates_per_rollout=10, steps_per_rollout=2048,
                 ignore_critic_on_train_loss=0,
                 multi_rollout_on_eval_tuning=True, is_ce=False, ce_args=None,
                 deterministic_eval=1, eps=0, **kwargs):
        # Modify default MAML's params
        default_kwargs = dict(
            grad_clip=0.5,  # as in stable-baselines
            train_lr=0.1,  # high fine-tuning lr (see MAML/Appendix A2)
            test_steps_per_task=(0,1,2,3),
            max_test_steps_per_task=3,
            log_freq=100,
            valid_freq=100,
            valid_tasks=20,
            test_tasks=100,
        )
        kwargs = utils.update_dict(
            kwargs, dict_to_add=default_kwargs, force=False)

        if env_args is None:
            env_args = {}
        if ppo_args is None:
            ppo_args = {}
        if policy_args is None:
            policy_args = {}
        if ce_args is None:
            ce_args = {}

        super(MetaRL, self).__init__(**kwargs)

        self.n_envs = n_envs
        self.env_name = env_name
        self.env_args = env_args
        self.meta = meta
        self.attributes = attributes
        if isinstance(self.attributes, str):
            self.attributes = (self.attributes,)
        self.init_env()

        self.updates_per_rollout = updates_per_rollout
        self.steps_per_rollout = steps_per_rollout
        self.multi_rollout_on_eval_tuning = multi_rollout_on_eval_tuning
        self.ignore_critic_on_train_loss = ignore_critic_on_train_loss
        self.ppo_args = ppo_args
        self.policy_args = policy_args
        self.losses = None
        self.reset()

        self.deterministic_eval = deterministic_eval
        self.eps = eps

        self.is_ce = is_ce
        self.init_ce(ce_args)

        self.current_train_tasks = self.train_tasks_per_batch * [None]
        self.task_count = 0
        self.train_passes_count = 0
        self.model_updates_count = 0

    def get_policy_fun(self):
        raise NotImplementedError()

    def reset(self):
        utils.set_all_seeds(self.seed)
        self.PPO = MetaPPO(
            self.get_policy_fun(), self.env, device=self.device,
            n_steps=self.steps_per_rollout//self.n_envs, policy_kwargs=self.policy_args,
            n_buffers=self.train_tasks_per_batch+1, **self.ppo_args)
        self.callback = self.PPO._setup_learn(self.PPO.n_steps, None, None, -1)[1]
        self.model = self.PPO.policy
        self.optimizer = self.model.optimizer
        self.current_train_tasks = self.train_tasks_per_batch * [None]
        self.task_count = 0
        self.train_passes_count = 0
        self.model_updates_count = 0
        self.losses = dict(policy=[], critic=[], entropy=[])

    def get_ce_defaults(self):
        return {}

    def init_ce(self, ce_args):
        assign_value = lambda kwargs, key, default_val: \
            kwargs[key] if key in kwargs else default_val
        self.ce_constructor = assign_value(ce_args, 'constructor', None)
        self.ce_phi0 = assign_value(ce_args, 'phi0', None)
        self.ce_batch_size = assign_value(ce_args, 'batch_size', 10)
        self.ce_min_batch_update = assign_value(ce_args, 'min_batch_update', 0.5)
        self.ce_soft_update = assign_value(ce_args, 'soft_update', 0.5)
        self.ce_n_orig_per_batch = assign_value(ce_args, 'n_orig_per_batch', 0)
        self.ce_w_clip = assign_value(ce_args, 'w_clip', 0)
        self.ce_alpha = assign_value(ce_args, 'alpha', 0.1)
        self.ce_kwargs = assign_value(ce_args, 'kwargs', {})
        for k, v in self.get_ce_defaults().items():
            if k not in ce_args:
                setattr(self, f'ce_{k}', v)

        self.ce_sample_count = self.ce_batch_size - 1
        self.ce_samples = None
        self.ce_scores = []

        self.ce = None
        if self.is_ce:
            model_updates_per_ce_batch = self.ce_batch_size * self.updates_per_rollout \
                                         / self.train_tasks_per_batch
            if self.meta_train_iterations % model_updates_per_ce_batch != 0:
                warnings.warn(f'Train iterations {self.meta_train_iterations} are not '
                              f'an integer multiplication of CE batch size '
                              f'{model_updates_per_ce_batch}. CE post analysis might '
                              f'be compromised.')  # TODO fix it in CEM module?

            self.valid_fun = lambda x: np.mean(
                sorted(x, reverse=True)[:int(np.ceil(self.ce_alpha * len(x)))])

            self.ce = self.ce_constructor(
                self.ce_phi0, batch_size=self.ce_batch_size, ref_alpha=self.ce_alpha,
                min_batch_update=self.ce_min_batch_update, soft_update=self.ce_soft_update,
                n_orig_per_batch=self.ce_n_orig_per_batch, w_clip=self.ce_w_clip,
                **self.ce_kwargs)

    def init_env(self):
        self.env = env_util.make_vec_env(
            self.env_name, n_envs=self.n_envs, seed=self.seed,
            env_kwargs=self.env_args)

    def get_actual_env(self, k=0):
        env = self.env.envs[k]
        while hasattr(env, 'env'):
            env = env.env
        return env

    def update_env(self, attr, val):
        # The built-in set_attr() of vec_env does not reach the low-level env.
        # I might have misused it, but for now I just set the attributes directly.
        # self.env.set_attr(attr, val)
        for k in range(self.n_envs):
            setattr(self.get_actual_env(k), attr, val)

    def update_task(self, task):
        if self.attributes is not None:
            if not isinstance(task, (tuple, list, np.ndarray)):
                task = (task,)
            for a, r in zip(self.attributes, task):
                self.update_env(a, r)

    def get_task(self):
        if self.attributes is None:
            return None
        env = self.get_actual_env()
        return [getattr(env, a) for a in self.attributes]

    def update_lr(self):
        # Update optimizer learning rate: this was moved here from
        # PPO.train(), since train() is currently used also for
        # fine-tuning in evaluation mode, where lr should not be
        # modified.
        self.PPO._update_learning_rate(self.model.optimizer)

    def really_sample_task(self, rng, is_evaluation):
        '''Sample and return a task.'''
        '''If self.meta==false, expected to return a constant value.'''
        raise NotImplementedError()

    def do_sample_task(self, rng, is_evaluation):
        if is_evaluation:
            task = self.really_sample_task(rng, is_evaluation)
            self.update_task(task)

        else:
            # training mode
            if self.model_updates_count % self.updates_per_rollout == 0:
                if self.is_ce:
                    task = self.sample_task_using_cem()
                else:
                    task = self.really_sample_task(rng, is_evaluation)
                self.update_task(task)
                self.current_train_tasks[self.task_count] = task
            else:
                task = self.current_train_tasks[self.task_count]

            self.task_count = (self.task_count + 1) % self.train_tasks_per_batch
            if self.task_count == 0:
                self.update_lr()

        return task

    def get_task_id(self, task):
        if isinstance(task, (tuple, list)):
            return '_'.join([f'{x:.2f}' for x in task])
        return f'{task:.2f}'

    def set_deterministic(self, is_evaluation, is_finetuning):
        # Deterministic policy must not be used on fine-tuning, where
        # heterogeneous trajectories must be generated.
        # However, it can be used on test rollouts during training or
        # evaluation.
        if self.deterministic_eval == 0:
            # never use deterministic policy
            is_det = False
        elif self.deterministic_eval == 1:
            # deterministic policy only on evaluation's test rollouts.
            is_det = is_evaluation and not is_finetuning
        elif self.deterministic_eval == 2:
            # deterministic policy only on all test rollouts.
            is_det = not is_finetuning
        else:
            raise ValueError(self.deterministic_eval)

        self.model.deterministic = is_det
        return is_det

    def set_random_eps(self, is_evaluation):
        # Whether to use epsilon-random policy.
        eps = self.eps if not is_evaluation else 0
        self.model.eps = eps

    def choose_rollout_buffer(self, is_evaluation, is_finetuning):
        buff_id = -1 if is_evaluation else (
                self.train_passes_count % self.train_tasks_per_batch)

        if not is_evaluation:
            buff_id = self.train_passes_count % self.train_tasks_per_batch
            # On training, we do rollout (on both fine-tune and test)
            #  once in updates_per_rollout steps.
            rollout_iteration = self.model_updates_count % \
                                self.updates_per_rollout == 0
            first_tune_step = self.curr_tune_steps == 0
            is_rollout = rollout_iteration and first_tune_step
        else:
            # On evaluation, we do rollout every test and at least on
            #  the first fine-tuning iteration.
            if is_finetuning:
                is_rollout = self.multi_rollout_on_eval_tuning or \
                             self.curr_tune_steps == 0
            else:
                is_rollout = True

        return is_rollout, buff_id

    def run_task(self, model, task, samples, is_evaluation, is_finetuning):
        self.set_deterministic(is_evaluation, is_finetuning)
        self.set_random_eps(is_evaluation)

        # Do rollouts
        do_rollout, buff_id = self.choose_rollout_buffer(
            is_evaluation, is_finetuning)
        self.PPO.rollout_buffer = self.PPO.all_buffers[buff_id]
        if do_rollout:
            self.PPO.collect_rollouts(
                self.PPO.env, self.callback, self.PPO.rollout_buffer,
                n_rollout_steps=self.PPO.n_steps)

        # In evaluation, after fine-tuning, return returns rather than loss
        if is_evaluation and not is_finetuning:
            rets = self.PPO.get_rollouts_returns()
            if len(rets) < 1:
                warnings.warn(f'Not a single complete episode in '
                              f'rollout buffer {buff_id:d}.')
            return -np.mean(rets)

        # In the end of a training tuning, update counter
        if not is_evaluation and not is_finetuning:
            self.train_passes_count += 1
            if self.train_passes_count % self.train_tasks_per_batch == 0:
                self.model_updates_count += 1

        # In the end of a training tuning, only measure policy loss
        vf_coef = None
        if self.ignore_critic_on_train_loss and not is_evaluation:
            if self.ignore_critic_on_train_loss in (1,3) and is_finetuning:
                vf_coef = 0
            if self.ignore_critic_on_train_loss in (2,3) and not is_finetuning:
                vf_coef = 0

        loss = self.PPO.train(vf_coef)

        # Save separate losses
        if not is_evaluation and not is_finetuning:
            self.losses['policy'].append(
                self.PPO.logger.name_to_value['train/policy_gradient_loss'])
            self.losses['critic'].append(
                self.PPO.logger.name_to_value['train/value_loss'])
            self.losses['entropy'].append(
                self.PPO.logger.name_to_value['train/entropy_loss'])

        return loss

    def analyze(self, train_resolution=10, logscale=False,
                show_validation_with_train=False, **kwargs):
        return super().analyze(
            train_resolution=train_resolution, logscale=logscale,
            show_validation_with_train=show_validation_with_train, **kwargs)

    ###########    CE-sampling methods    ###########

    def sample_task_using_cem(self):
        self.ce_sample_count += 1
        if self.ce_sample_count >= self.ce_batch_size:
            self.ce_samples = [xw[0] for xw in self.ce.sample_batch()]
            self.ce_sample_count = 0
        return self.ce_samples[self.ce_sample_count]

    def update_sampler(self, loss):
        if not self.is_ce:
            return

        if (self.model_updates_count-1) % self.updates_per_rollout == 0:
            self.ce_scores.append(-loss.item())
            if len(self.ce_scores) >= self.ce_batch_size:
                self.ce.update_batch(self.ce_scores)
                self.ce_scores = []

    def ce_summary(self, axs=None, a=0):
        n = len(self.attributes)
        if axs is None:
            axs = utils.Axes(n+1, 4, fontsize=15)

        c1, c2 = self.ce.get_data()
        for att in self.attributes:
            axs[a].plot(c1.iloc[:,-n+a])
            axs.labs(a, 'iteration', f'E[{att}]')
            a += 1

        self.ce.show_sampled_scores(ax=axs[a])
        a += 1

        plt.tight_layout()
        return axs

def show_train_losses(mamls, axs=None, smooth=100, logscale=True, normalize=True):
    n = len(mamls)
    if axs is None:
        axs = utils.Axes(n, 4, fontsize=15)
    for a, M in enumerate(mamls):
        if not hasattr(M, 'losses'):
            warnings.warn(f'Model {M.title} has no losses recording.')
            continue
        no_neg = True
        for nm, coef in zip(('policy', 'critic', 'entropy'),
                            (1, M.PPO.vf_coef, M.PPO.ent_coef)):
            y = np.array(M.losses[nm])
            if normalize:
                y *= coef
            if smooth > 1 and len(y) > 0:
                y = np.convolve(y, np.ones(smooth)/smooth, mode='same')
            axs[a].plot(1+np.arange(len(y)), y, label=nm)
            if np.any(y<=0):
                no_neg = False
        if logscale and no_neg:
            axs[a].set_yscale('log')
        axs.labs(a, 'iteration', 'loss', M.title)
        axs[a].legend(fontsize=12)
    plt.tight_layout()
    return axs

def analyze_experiment(E, keys=None, analysis_kwargs=None, logscale=False):
    keys = E._get_keys(keys)
    mamls = [E.mamls[k] for k in keys]
    if analysis_kwargs is None: analysis_kwargs = {}

    print('log_std_init vs. log_std:')
    for M in mamls:
        print(M.title, M.model.log_std_init, M.model.log_std.detach())

    axs = E.analyze_all(keys, logscale=logscale, show_validation_with_train=False,
                        **analysis_kwargs)
    axs[0].set_ylabel('train loss', fontsize=15)
    for i in range(1,5):
        axs[i].set_ylabel('negative return', fontsize=15)
    for M in mamls:
        if 'RAML' in M.title:
            E.analyze_all(
                keys, logscale=logscale, show_validation_with_train=False,
                est=f'cvar{int(np.round(100*M.ce_alpha)):02d}',
                **analysis_kwargs)
            break

    for M in mamls:
        if 'RAML' in M.title:
            M.ce_summary()

    show_train_losses(mamls)

    # E.show_all(keys, eval_steps=eval_steps)
    # E.visualize_all(keys)
    # E.visualize_all(keys, fun='visualize_critic')
