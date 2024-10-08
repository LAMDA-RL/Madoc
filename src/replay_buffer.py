import numpy as np
import torch as th
import h5py
from tqdm import tqdm
import os


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


class ReplayBuffer():
    def __init__(self, use_neorl, clip_action, state_dim, action_dim, sim_para_dim, dataset_name, buffer_size, offline_size, device):
        if use_neorl:
            npz_path = os.environ['HOME'] + \
                "/.d4rl/datasets/" + dataset_name + ".npz"
            dataset = np.load(npz_path)
            size = dataset['obs'].shape[0]
            self.offline_size = int(offline_size)
            if self.offline_size > size:
                self.offline_size = size
            print(size, self.offline_size, buffer_size)
            self.total_size = self.offline_size + buffer_size
            self.device = device
            s = th.FloatTensor(dataset['obs'])[:self.offline_size]
            a = th.FloatTensor(dataset['action'])[:self.offline_size]
            r = th.FloatTensor(dataset['reward'])[
                :self.offline_size].reshape(-1, 1)
            s_next = th.FloatTensor(dataset['next_obs'])[:self.offline_size]
            done = th.FloatTensor(dataset['done'])[
                :self.offline_size].reshape(-1, 1)
        else:
            h5path = os.environ['HOME'] + \
                "/.d4rl/datasets/" + dataset_name + ".hdf5"
            dataset = {}
            with h5py.File(h5path, 'r') as dataset_file:
                for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                    try:
                        dataset[k] = dataset_file[k][:]
                    except ValueError as e:
                        dataset[k] = dataset_file[k][()]
            size = dataset['observations'].shape[0]
            self.offline_size = int(offline_size)
            if self.offline_size > size:
                self.offline_size = size
            print(size, self.offline_size, buffer_size)
            self.total_size = self.offline_size + buffer_size
            self.device = device
            idx = np.random.choice(
                range(size), self.offline_size, replace=False)
            s = th.FloatTensor(dataset['observations'])[idx]
            a = th.FloatTensor(dataset['actions'])[idx]
            r = th.FloatTensor(dataset['rewards'])[idx].reshape(-1, 1)
            s_next = th.FloatTensor(dataset['next_observations'])[idx]
            done = th.FloatTensor(dataset['terminals'])[idx].reshape(-1, 1)
        a = th.clamp(a, -clip_action, clip_action)
        self.ptr = self.offline_size
        self.actual_size = self.offline_size
        self.state = th.cat(
            [s, th.zeros(buffer_size, state_dim)], dim=0)
        self.action = th.cat([a, th.zeros(buffer_size, action_dim)], dim=0)
        self.reward = th.cat([r, th.zeros(buffer_size, 1)], dim=0)
        self.next_state = th.cat(
            [s_next, th.zeros(buffer_size, state_dim)], dim=0)
        self.para = th.zeros(self.total_size, sim_para_dim)
        self.done = th.cat([done, th.zeros(buffer_size, 1)], dim=0)

    def sample(self, batch_size, scope=None, type=None):
        if scope == None:
            ind = np.random.randint(0, self.actual_size, size=batch_size)
        elif scope == "real":
            ind = np.random.randint(
                0, self.offline_size, size=batch_size)
        elif scope == "sim":
            ind = np.random.randint(
                self.offline_size, self.actual_size, size=batch_size)
        else:
            raise RuntimeError("Misspecified range for replay buffer sampling")

        if type == None:
            return {
                'observations': self.state[ind].to(self.device),
                'actions': self.action[ind].to(self.device),
                'rewards': self.reward[ind].to(self.device),
                'next_observations': self.next_state[ind].to(self.device),
                'dones': self.done[ind].to(self.device),
                'paras': self.para[ind].to(self.device)
            }
        elif type == "sas":
            return {
                'observations': self.state[ind].to(self.device),
                'actions': self.action[ind].to(self.device),
                'next_observations': self.next_state[ind].to(self.device)
            }
        elif type == "sa":
            return {
                'observations': self.state[ind].to(self.device),
                'actions': self.action[ind].to(self.device)
            }
        elif type == "sasp":
            return {
                'observations': self.state[ind].to(self.device),
                'actions': self.action[ind].to(self.device),
                'next_observations': self.next_state[ind].to(self.device),
                'paras': self.para[ind].to(self.device)
            }
        else:
            raise RuntimeError(
                "Misspecified return data types for replay buffer sampling")

    def append(self, s, a, r, s_, done, para):
        self.state[self.ptr] = th.from_numpy(s)
        self.action[self.ptr] = th.from_numpy(a)
        self.next_state[self.ptr] = th.from_numpy(s_)
        self.reward[self.ptr, 0] = r
        self.done[self.ptr, 0] = done
        self.para[self.ptr] = para

        # fix the offline dataset and shuffle the simulated part
        self.ptr = (self.ptr + 1 - self.offline_size) % (self.total_size -
                                                         self.offline_size) + self.offline_size
        self.actual_size = min(self.actual_size + 1, self.total_size)

    def clear_sim_data(self):
        self.ptr = self.offline_size
        self.actual_size = self.offline_size
