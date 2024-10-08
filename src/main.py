import absl.app
import absl.flags
from copy import deepcopy
import os
import yaml
import numpy as np
import gym
import torch
from tqdm import trange

from utils import define_flags_with_default, get_user_flags, set_random_seed, prefix_metrics
from sampler import StepSampler, TrajSampler
from replay_buffer import ReplayBuffer
from network import ConcatDiscriminator, VAE
from agent import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy, get_para_dim, get_designer_vnets, get_designer_policies
from learner import SAC

torch.set_num_threads(1)

FLAGS_DEF = define_flags_with_default(
    name_str='ours',
    env_name='HalfCheetah-v2',
    dataset_name='halfcheetah_medium_replay-v2',
    device='cpu',
    use_wandb=False,
    neorl=False,

    offline_size=2e5,
    buffer_size=2e5,
    dis_dropout=True,
    seed=-1,
    batch_size=256,
    clip_action=1.0,

    policy_arch='256-256',
    q_arch='256-256',

    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # train and evaluate policy
    max_traj_length=1000,
    sac_epochs=1000,
    bc_epochs=100,
    group_epochs=200,
    calibrate_epochs=1000,
    n_rollout_steps_per_epoch=1000,
    n_train_steps_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=10,

    config=SAC.get_default_config(),
)


def main(argv):
    # config
    FLAGS = absl.flags.FLAGS
    conf = get_user_flags(FLAGS, FLAGS_DEF)
    if "halfcheetah" in FLAGS.env_name.lower():
        config_name = "halfcheetah"
        FLAGS.config.group_num = 6
    elif "hopper" in FLAGS.env_name.lower():
        config_name = "hopper"
        FLAGS.config.group_num = 4
    elif "walker2d" in FLAGS.env_name.lower():
        config_name = "walker2d"
        FLAGS.config.group_num = 6
    elif "ant" in FLAGS.env_name.lower():
        config_name = "ant"
        FLAGS.config.group_num = 6
    else:
        raise ValueError(
            "No implementation found for {}".format(FLAGS.env_name))
    with open(os.path.join(os.path.dirname(__file__), "config", "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    FLAGS.config.device = FLAGS.device

    # log
    if FLAGS.use_wandb:
        import wandb
        wandb.init(project="xxx", entity='yy',
                   config=conf, mode='online',)
        wandb.run.name = f"{FLAGS.name_str}_{FLAGS.dataset_name}"
    else:
        from torch.utils.tensorboard import SummaryWriter
        logdir = f"results/{FLAGS.name_str}_{FLAGS.dataset_name}"
        if os.path.exists(logdir):
            indices = [int(id) for id in os.listdir(logdir)]
            id = max(indices) + 1
        else:
            id = 0
        writer = SummaryWriter(
            log_dir=f"results/{FLAGS.name_str}_{FLAGS.dataset_name}/{id}")

    # env
    if FLAGS.neorl:
        real_env = gym.make(
            FLAGS.env_name, exclude_current_positions_from_observation=False)
        sim_env = gym.make(
            FLAGS.env_name, exclude_current_positions_from_observation=False)
    else:
        real_env = gym.make(FLAGS.env_name)
        sim_env = gym.make(FLAGS.env_name)

    # seed
    if FLAGS.seed is not -1:
        set_random_seed(FLAGS.seed)
        real_env.seed(FLAGS.seed)
        sim_env.seed(FLAGS.seed)

    # replay buffer
    state_dim = real_env.observation_space.shape[0]
    action_dim = real_env.action_space.shape[0]
    sim_para_dim = get_para_dim(config_dict)
    buffer = ReplayBuffer(FLAGS.neorl, FLAGS.clip_action, state_dim, action_dim, sim_para_dim, FLAGS.dataset_name,  buffer_size=int(
        FLAGS.buffer_size), offline_size=FLAGS.offline_size, device=FLAGS.device)

    # agent
    policy = TanhGaussianPolicy(state_dim, action_dim, arch=FLAGS.policy_arch,
                                log_std_multiplier=FLAGS.policy_log_std_multiplier, log_std_offset=FLAGS.policy_log_std_offset,)

    # q-network
    qr_1 = FullyConnectedQFunction(state_dim, action_dim, arch=FLAGS.q_arch,)
    qr_2 = FullyConnectedQFunction(state_dim, action_dim, arch=FLAGS.q_arch,)
    target_qr_1 = deepcopy(qr_1)
    target_qr_2 = deepcopy(qr_2)

    # discirminators
    d_sa = ConcatDiscriminator(state_dim + action_dim, 256, 2,
                               FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)
    d_sas = ConcatDiscriminator(2 * state_dim + action_dim, 256, 2,
                                FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)

    # learner
    FLAGS.config.target_entropy = -np.prod(real_env.action_space.shape).item()
    sac = SAC(FLAGS.config, policy, qr_1, qr_2, target_qr_1,
              target_qr_2, d_sa, d_sas, buffer, config_dict, FLAGS.device)
    sac.torch_to_device(FLAGS.device)

    # samplers for "simulated" training and "real-world" evaluation
    train_sampler = StepSampler(sim_env.unwrapped, config_dict)
    eval_sampler = TrajSampler(real_env.unwrapped, FLAGS.max_traj_length)
    # sampling policy is always the current policy: \pi
    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    # bc
    for epoch in range(FLAGS.bc_epochs):
        for batch_idx in trange(FLAGS.n_train_steps_per_epoch):
            sac.train_bc(FLAGS.batch_size)
    trajs = eval_sampler.sample(
        sampler_policy, FLAGS.eval_n_trajs
    )
    print("bc return: {}".format(
        np.mean([np.sum(t['rewards']) for t in trajs])))

    # group
    vae = VAE(len(config_dict), state_dim, action_dim, 16, 16)
    sac.set_vae(vae)
    for epoch in range(FLAGS.group_epochs):
        metrics = {}
        n_rollout_trajs = len(config_dict)
        n_rollout_steps_per_traj = FLAGS.n_rollout_steps_per_epoch // n_rollout_trajs
        train_sampler.group_sample(
            sampler_policy,  n_rollout_trajs, n_rollout_steps_per_traj, buffer)
        for batch_idx in trange(FLAGS.n_train_steps_per_epoch):
            if batch_idx + 1 == FLAGS.n_train_steps_per_epoch:
                metrics.update(prefix_metrics(
                    sac.train_group(FLAGS.batch_size, epoch / FLAGS.group_epochs), 'group'))
            else:
                sac.train_group(FLAGS.batch_size, epoch / FLAGS.group_epochs)
        if FLAGS.use_wandb:
            wandb.log(metrics)
        else:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, epoch)

    groups = max(sac.group_cnt, key=sac.group_cnt.get)
    print(sac.group_cnt)
    print("groups: {}".format(groups))

    # simulator designer
    buffer.clear_sim_data()
    sac.clear_policy()
    designer_policies = get_designer_policies(groups)
    designer_vnets = get_designer_vnets(groups)
    sac.set_designer(designer_policies, designer_vnets)

    # calibrate
    for epoch in range(FLAGS.calibrate_epochs):
        metrics = {}
        n_sample_para = 10
        train_sampler.sample(sampler_policy, n_sample_para, designer_policies,
                             groups, deterministic=False, replay_buffer=buffer)
        for batch_idx in trange(FLAGS.n_train_steps_per_epoch):
            if batch_idx + 1 == FLAGS.n_train_steps_per_epoch:
                metrics.update(prefix_metrics(
                    sac.train_bc(FLAGS.batch_size), 'bc'))
                metrics.update(prefix_metrics(
                    sac.train_simulator_parameters(FLAGS.batch_size, groups), 'designer'))
            else:
                sac.train_bc(FLAGS.batch_size)
                sac.train_simulator_parameters(FLAGS.batch_size, groups)
        if FLAGS.use_wandb:
            wandb.log(metrics)
        else:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, epoch)

    # sac
    buffer.clear_sim_data()
    sac.clear_policy()
    for epoch in range(FLAGS.sac_epochs):
        metrics = {}
        # rollout
        train_sampler.sample(sampler_policy, 1, designer_policies,
                             groups, deterministic=True, replay_buffer=buffer)
        # train
        for batch_idx in trange(FLAGS.n_train_steps_per_epoch):
            if batch_idx + 1 == FLAGS.n_train_steps_per_epoch:
                metrics.update(prefix_metrics(
                    sac.train_sac(FLAGS.batch_size), 'sac'))
            else:
                sac.train_sac(FLAGS.batch_size)
        # eval
        if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
            trajs = eval_sampler.sample(
                sampler_policy, FLAGS.eval_n_trajs
            )
            metrics['average_return'] = np.mean(
                [np.sum(t['rewards']) for t in trajs])
            metrics['average_traj_length'] = np.mean(
                [len(t['rewards']) for t in trajs])
        metrics['epoch'] = epoch
        if FLAGS.use_wandb:
            wandb.log(metrics)
        else:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, epoch)

    if not FLAGS.use_wandb:
        writer.close()


if __name__ == '__main__':
    absl.app.run(main)
