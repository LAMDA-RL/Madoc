import numpy as np
import torch as th


class StepSampler(object):

    def __init__(self, env, config_dict):
        self._env = env
        self.max_steps = env.spec.max_episode_steps
        self.model = env.model
        self.config_dict = config_dict

    def sample_para(self, designer_policies, groups, deterministic):
        paras = []
        for i, group in enumerate(groups):
            para = designer_policies[i](deterministic=deterministic)[
                0].detach().reshape(-1)
            paras.append(para)
            for j, para_idx in enumerate(group):
                k, v = list(self.config_dict.items())[para_idx]
                p = para.cpu()[j]
                if 'gravity' in k:
                    # y = (x * (max - min) + max +min) / 2
                    self.model.opt.gravity[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                elif 'body_mass' in k:
                    self.model.body_mass[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                elif 'dof_damping' in k:
                    self.model.dof_damping[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                else:
                    raise ValueError(
                        "No implementation found for {}".format(k))
        paras = th.cat(paras).reshape(1, -1)
        return paras

    def sample(self, policy, n_sample_para, designer_policies, groups, deterministic=False, replay_buffer=None):
        observation = self.env.reset()
        t = 0
        for st in range(self.max_steps):
            if st % (self.max_steps / n_sample_para) == 0:
                paras = self.sample_para(
                    designer_policies, groups, deterministic)

            action = policy(
                np.expand_dims(observation, 0)
            )[0, :]
            next_observation, reward, terminated, info = self.env.step(
                action)

            if replay_buffer is not None:
                if t < self.max_steps - 1:
                    replay_buffer.append(
                        observation, action, reward, next_observation, terminated, paras
                    )
                else:
                    replay_buffer.append(
                        observation, action, reward, next_observation, False, paras
                    )
            observation = next_observation
            t += 1

            if terminated:
                observation = self.env.reset()
                t = 0

    def group_sample(self, policy, n_trajs, n_steps, replay_buffer):
        for i in range(n_trajs):
            paras = []
            for j, (k, v) in enumerate(self.config_dict.items()):
                if i == j:
                    p = np.random.uniform(-1, 1)
                    if p == 0:
                        p += 1e-5
                else:
                    p = 0
                paras.append(p)
                if 'gravity' in k:
                    # y = (x * (max - min) + max +min) / 2
                    self.model.opt.gravity[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                elif 'body_mass' in k:
                    self.model.body_mass[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                elif 'dof_damping' in k:
                    self.model.dof_damping[v['idx']] = (
                        p * (v['max'] - v['min']) + v['max'] + v['min']) / 2
                else:
                    raise ValueError(
                        "No implementation found for {}".format(k))

            paras = th.Tensor(paras).reshape(1, -1)
            observation = self.env.reset()
            t = 0
            for _ in range(n_steps):
                action = policy(
                    np.expand_dims(observation, 0)
                )[0, :]
                next_observation, reward, terminated, info = self.env.step(
                    action)

                if replay_buffer is not None:
                    if t < self.max_steps - 1:
                        replay_buffer.append(
                            observation, action, reward, next_observation, terminated, paras
                        )
                    else:
                        replay_buffer.append(
                            observation, action, reward, next_observation, False, paras
                        )
                observation = next_observation
                t += 1

                if terminated:
                    observation = self.env.reset()
                    t = 0

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            observation = self.env.reset()
            for _ in range(self.max_traj_length):
                action = policy(
                    np.expand_dims(observation, 0), deterministic=True
                )[0, :]
                next_observation, reward, terminated, info = self.env.step(
                    action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated)
                next_observations.append(next_observation)
                observation = next_observation

                if terminated:
                    break

            trajs.append(dict(
                rewards=np.array(rewards, dtype=np.float32),
            ))

        return trajs

    @property
    def env(self):
        return self._env
