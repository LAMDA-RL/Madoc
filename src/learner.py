from ml_collections import ConfigDict
from torch.optim import Adam
from agent import soft_target_update, Scalar
import torch
import torch.nn.functional as F
import numpy
from scipy.cluster.vq import kmeans, vq
import random
from sklearn.decomposition import PCA


class SAC(object):
    @staticmethod
    def get_default_config():
        config = ConfigDict()
        config.batch_size = 256
        config.device = 'cpu'
        config.discount = 0.99
        config.alpha_multiplier = 1
        config.use_automatic_entropy_tuning = True
        config.target_entropy = 0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.d_sa_lr = 1e-3
        config.d_sas_lr = 1e-3
        config.use_dop = True
        config.group_num = 6

        config.noise_std_discriminator = 0.0
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.target_prior_update_period = 10
        config.soft_target_prior_update_rate = 1e-3

        # pM/pM^
        config.clip_log_dynamics_ratio_min = -100
        config.clip_log_dynamics_ratio_max = 100
        return config

    def __init__(self, config, policy, qr_1, qr_2, target_qr_1,
                 target_qr_2, d_sa, d_sas, replay_buffer, config_dict, device):
        self.config = config
        self.policy = policy
        torch.save(self.policy.state_dict(), 'policy.pth')
        self.qr_1 = qr_1
        self.qr_2 = qr_2
        self.target_qr_1 = target_qr_1
        self.target_qr_2 = target_qr_2
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.replay_buffer = replay_buffer
        self.config_dict = config_dict
        self.device = device

        # optimizers
        self.policy_optimizer = Adam(
            self.policy.parameters(), self.config.policy_lr)
        self.qr_optimizer = Adam(
            list(self.qr_1.parameters()) + list(self.qr_2.parameters()), self.config.qf_lr)
        self.d_sa_optimizer = Adam(
            self.d_sa.parameters(), self.config.d_sa_lr)
        self.d_sas_optimizer = Adam(
            self.d_sas.parameters(), self.config.d_sas_lr)

        # whether to use automatic entropy tuning (True in default)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = Adam(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0
        self.designer_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qr_1, self.target_qr_1,
                           soft_target_update_rate)
        soft_target_update(self.qr_2, self.target_qr_2,
                           soft_target_update_rate)

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qr_1, self.qr_2,
                   self.target_qr_1, self.target_qr_2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

    def train_sac(self, batch_size):
        self._total_steps += 1

        sim_batch = self.replay_buffer.sample(batch_size, scope="sim")
        sim_observations = sim_batch['observations']
        sim_actions = sim_batch['actions']
        sim_rewards = sim_batch['rewards']
        sim_next_observations = sim_batch['next_observations']
        sim_dones = sim_batch['dones']

        """ Q function loss """
        if self.config.use_automatic_entropy_tuning:
            alpha = self.log_alpha().detach().exp() * self.config.alpha_multiplier
        else:
            alpha = sim_observations.new_tensor(self.config.alpha_multiplier)

        sim_q1_pred = self.qr_1(sim_observations, sim_actions)
        sim_q2_pred = self.qr_2(sim_observations, sim_actions)
        with torch.no_grad():
            sim_new_next_actions, sim_next_log_pi, _ = self.policy(
                sim_next_observations)
            sim_target_q_values = torch.min(
                self.target_qr_1(sim_next_observations, sim_new_next_actions),
                self.target_qr_2(sim_next_observations, sim_new_next_actions),
            )
            sim_target_q_values = sim_target_q_values - alpha * sim_next_log_pi
            sim_td_target = torch.squeeze(sim_rewards, -1) + (1. - torch.squeeze(
                sim_dones, -1)) * self.config.discount * sim_target_q_values

        sim_qr1_loss = F.mse_loss(sim_q1_pred, sim_td_target)
        sim_qr2_loss = F.mse_loss(sim_q2_pred, sim_td_target)

        qr_loss = sim_qr1_loss + sim_qr2_loss
        self.qr_optimizer.zero_grad()
        qr_loss.backward()
        self.qr_optimizer.step()

        # Policy loss
        sim_new_actions, sim_log_pi, _ = self.policy(sim_observations)
        q_new_actions = torch.min(
            self.qr_1(sim_observations, sim_new_actions),
            self.qr_2(sim_observations, sim_new_actions),
        )
        policy_loss = (alpha * sim_log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (sim_log_pi +
                           self.config.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = sim_observations.new_tensor(0.0)

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        metrics = dict(
            log_pi=sim_log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            sim_qr1_loss=sim_qr1_loss.item(),
            sim_qr2_loss=sim_qr2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_sim_qr1=sim_q1_pred.mean().item(),
            average_sim_qr2=sim_q2_pred.mean().item(),
            average_sim_target_q=sim_target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        return metrics

    def train_bc(self, batch_size):
        real_observations, real_actions = self.replay_buffer.sample(
            batch_size, scope="real", type="sa").values()
        pred_actions, log_pi, _ = self.policy(real_observations)
        bc_loss = F.mse_loss(pred_actions, real_actions)
        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi +
                                               self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().detach().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = real_observations.new_tensor(0.0)
            alpha = real_observations.new_tensor(self.config.alpha_multiplier)
        policy_loss = bc_loss + alpha * log_pi.mean()

        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        return metrics

    def train_discriminator(self):
        self.d_sa.train()
        self.d_sas.train()
        real_obs, real_action, real_next_obs = self.replay_buffer.sample(
            self.config.batch_size, scope="real", type="sas").values()
        sim_obs, sim_action, sim_next_obs = self.replay_buffer.sample(
            self.config.batch_size, scope="sim", type="sas").values()

        # input noise: prevents overfitting
        if self.config.noise_std_discriminator > 0:
            real_obs += torch.randn(real_obs.shape, device=self.config.device) * \
                self.config.noise_std_discriminator
            real_action += torch.randn(real_action.shape, device=self.config.device) * \
                self.config.noise_std_discriminator
            real_next_obs += torch.randn(real_next_obs.shape,
                                         device=self.config.device) * self.config.noise_std_discriminator
            sim_obs += torch.randn(sim_obs.shape, device=self.config.device) * \
                self.config.noise_std_discriminator
            sim_action += torch.randn(sim_action.shape, device=self.config.device) * \
                self.config.noise_std_discriminator
            sim_next_obs += torch.randn(sim_next_obs.shape,
                                        device=self.config.device) * self.config.noise_std_discriminator

        real_sa_logits = self.d_sa(real_obs, real_action)
        real_sa_prob = F.softmax(real_sa_logits, dim=1)
        sim_sa_logits = self.d_sa(sim_obs, sim_action)
        sim_sa_prob = F.softmax(sim_sa_logits, dim=1)

        real_adv_logits = self.d_sas(real_obs, real_action, real_next_obs)
        real_sas_prob = F.softmax(real_adv_logits + real_sa_logits, dim=1)
        sim_adv_logits = self.d_sas(sim_obs, sim_action, sim_next_obs)
        sim_sas_prob = F.softmax(sim_adv_logits + sim_sa_logits, dim=1)

        dsa_loss = (- torch.log(real_sa_prob[:, 0]) -
                    torch.log(sim_sa_prob[:, 1])).mean()
        dsas_loss = (- torch.log(real_sas_prob[:, 0]) -
                     torch.log(sim_sas_prob[:, 1])).mean()

        # Optimize discriminator(s,a) and discriminator(s,a,s')
        self.d_sa_optimizer.zero_grad()
        dsa_loss.backward(retain_graph=True)

        self.d_sas_optimizer.zero_grad()
        dsas_loss.backward()

        self.d_sa_optimizer.step()
        self.d_sas_optimizer.step()

        return dsa_loss.cpu().detach().numpy().item(), dsas_loss.cpu().detach().numpy().item()

    def train_simulator_parameters(self, batch_size, groups):
        self.designer_steps += 1
        dsa_loss, dsas_loss = self.train_discriminator()

        sim_observations, sim_actions, sim_next_observations, sim_paras = self.replay_buffer.sample(
            batch_size, scope="sim", type="sasp").values()
        reward = self.log_real_sim_dynacmis_ratio(
            sim_observations, sim_actions, sim_next_observations).reshape(-1, 1)

        # v_loss
        ind = 0
        v_tot = 0
        for i, group in enumerate(groups):
            p = sim_paras[:, ind: ind + len(group)]
            ind += len(group)
            if self.config.use_dop:
                v_tot += self.designer_vnets[i](p) * \
                    torch.abs(self.dop_weights[i])
            else:
                v_tot += self.designer_vnets[i](p)
        if self.config.use_dop:
            v_tot += self.dop_weights[-1]
        v_loss = F.mse_loss(v_tot, reward)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # designer_loss
        policy_loss = 0
        kl_weight = min(10, numpy.exp(self.designer_steps / 2e5))
        for i in range(len(groups)):
            sample_paras, log_pi, _ = self.designer_policies[i](
                repeat=batch_size)
            v_new_paras = self.designer_vnets[i](sample_paras)
            kl_loss = self.designer_policies[i].KL_loss()
            policy_loss += (0.05 * log_pi - v_new_paras).mean() + \
                kl_loss * kl_weight
        self.designer_policies_optimizer.zero_grad()
        policy_loss.backward()
        self.designer_policies_optimizer.step()

        if self.designer_steps % self.config.target_prior_update_period == 0:
            for i in range(len(groups)):
                self.designer_policies[i].update_prior(
                    self.config.soft_target_prior_update_rate)

        metrics = dict(
            dsa_loss=dsa_loss,
            dsas_loss=dsas_loss,
            reward=reward.mean().item(),
            v_loss=v_loss.item(),
            policy_loss=policy_loss.item()
        )

        for i, group in enumerate(groups):
            para = self.designer_policies[i](deterministic=True)[
                0].detach().reshape(-1)
            for j, para_idx in enumerate(group):
                k, v = list(self.config_dict.items())[para_idx]
                p = para.cpu()[j]
                error = abs(
                    (p * (v['max'] - v['min']) + v['max'] + v['min']) / 2 - v['GT'])
                metrics['error_' + k] = error
        return metrics

    def log_real_sim_dynacmis_ratio(self, observations, actions, next_observations):
        self.d_sa.eval()
        self.d_sas.eval()
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            # clipped pM/pM^
            log_ratio = torch.log(sas_prob[:, 0]) \
                - torch.log(sas_prob[:, 1]) \
                - torch.log(sa_prob[:, 0]) \
                + torch.log(sa_prob[:, 1])

        return torch.clamp(log_ratio, self.config.clip_log_dynamics_ratio_min, self.config.clip_log_dynamics_ratio_max)

    def clear_policy(self):
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha().data.zero_()
            self.alpha_optimizer = Adam(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        self.policy.load_state_dict(torch.load('policy.pth'))
        self.policy_optimizer = Adam(
            self.policy.parameters(), self.config.policy_lr)

    def set_designer(self, designer_policies, designer_vnets):
        self.designer_policies = designer_policies
        self.designer_vnets = designer_vnets
        self.designer_policies_optimizer = Adam(
            self.designer_policies.parameters(), self.config.policy_lr)
        if self.config.use_dop:
            self.dop_weights = torch.ones(
                len(designer_vnets) + 1, requires_grad=True)
            self.v_optimizer = Adam(
                list(self.designer_vnets.parameters()) + [self.dop_weights], self.config.qf_lr)
            self.dop_weights.to(self.device)
        else:
            self.v_optimizer = Adam(
                self.designer_vnets.parameters(), self.config.qf_lr)
        self.designer_policies.to(self.device)
        self.designer_vnets.to(self.device)

    def set_vae(self, vae):
        self.vae = vae
        self.vae_optimizer = Adam(self.vae.parameters(), self.config.policy_lr)
        self.vae.to(self.device)
        self.group_cnt = {}

    def train_group(self, batch_size, epoch):
        sim_observations, sim_actions, sim_next_observations, sim_paras = self.replay_buffer.sample(
            batch_size, scope="sim", type="sasp").values()
        agent_id = torch.where(sim_paras != 0, 1, sim_paras)
        non_zero_indices = torch.nonzero(sim_paras)
        xi = sim_paras[non_zero_indices[:, 0],
                       non_zero_indices[:, 1]].unsqueeze(1)
        vae_loss = self.vae.calculate_loss(
            agent_id, sim_observations, sim_actions, sim_next_observations, xi)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        metrics = dict(
            vae_loss=vae_loss.item(),
        )
        if epoch > 0.95 and random.random() > 0.99:
            key = self.get_groups()
            self.group_cnt[key] = self.group_cnt.get(key, 0) + 1
        return metrics

    def get_groups(self):
        agent_id = torch.eye(len(self.config_dict)).to(self.device)
        z, _ = self.vae.encode(agent_id)
        z = z.detach().cpu().numpy()
        pca = PCA(n_components=3)
        z = pca.fit_transform(z)
        centroids, _ = kmeans(z, self.config.group_num)
        clusters, _ = vq(z, centroids)
        labels = []
        for i in range(self.config.group_num):
            indices = [idx for idx in range(
                len(clusters)) if clusters[idx] == i]
            labels.append(indices)
        labels.sort()
        labels = [tuple(lst) for lst in labels]
        return tuple(labels)
