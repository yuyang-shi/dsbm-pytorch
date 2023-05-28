import os, sys, warnings, time
import re
from collections import OrderedDict
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from .data import DBDSB_CacheLoader
from .sde import *
from .runners import *
from .runners.config_getters import get_model, get_optimizer, get_plotter, get_logger
from .runners.ema import EMAHelper
from .trainer_dbdsb import IPF_DBDSB

# from torchdyn.core import NeuralODE
from torchdiffeq import odeint


class IPF_RF(IPF_DBDSB):
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, final_cond_model=None,
                 valid_ds=None, test_ds=None):
        super().__init__(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator, final_cond_model=final_cond_model,
                         valid_ds=valid_ds, test_ds=test_ds)
        self.langevin = DBDSB_VE(0., self.num_steps, self.timesteps, self.shape_x, self.shape_y, first_coupling="ind", ot_sampler=self.args.ot_sampler)

    def build_checkpoints(self):
        self.first_pass = True  # Load and use checkpointed networks during first pass
        self.ckpt_dir = './checkpoints/'
        self.ckpt_prefixes = ["net_b", "sample_net_b", "optimizer_b"]
        self.cache_dir='./cache/'
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)

        if self.args.get('checkpoint_run', False):
            self.resume, self.checkpoint_it, self.checkpoint_pass, self.step = \
                True, self.args.checkpoint_it, self.args.checkpoint_pass, self.args.checkpoint_iter
            print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")

            self.checkpoint_b = hydra.utils.to_absolute_path(self.args.checkpoint_b)
            self.sample_checkpoint_b = hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)
            self.optimizer_checkpoint_b = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)
            
        else:
            self.ckpt_dir_load = os.path.abspath(self.ckpt_dir)
            ckpt_dir_load_list = os.path.normpath(self.ckpt_dir_load).split(os.sep)
            if 'test' in ckpt_dir_load_list:
                self.ckpt_dir_load = os.path.join(*ckpt_dir_load_list[:ckpt_dir_load_list.index('test')], "checkpoints/")
            self.resume, self.checkpoint_it, self.checkpoint_pass, self.step, ckpt_b_suffix = self.find_last_ckpt()

            if self.resume:
                if not self.args.autostart_next_it and self.step == 1 and not (self.checkpoint_it == 1 and self.checkpoint_pass == 'b'): 
                    self.checkpoint_pass, self.checkpoint_it = self.compute_prev_it(self.checkpoint_pass, self.checkpoint_it)
                    self.step = self.compute_max_iter(self.checkpoint_pass, self.checkpoint_it) + 1

                print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")
                self.checkpoint_b, self.sample_checkpoint_b, self.optimizer_checkpoint_b = [os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_{ckpt_b_suffix}.ckpt") for ckpt_prefix in self.ckpt_prefixes[:3]]

    def build_models(self, forward_or_backward=None):
        # running network
        net_b = get_model(self.args)

        if self.first_pass and self.resume:
            if self.resume:
                try:
                    net_b.load_state_dict(torch.load(self.checkpoint_b))
                except:
                    state_dict = torch.load(self.checkpoint_b)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_b.load_state_dict(new_state_dict)

        if forward_or_backward is None:
            net_b = self.accelerator.prepare(net_b)
            self.net = torch.nn.ModuleDict({'b': net_b})
        if forward_or_backward == 'b':
            net_b = self.accelerator.prepare(net_b)
            self.net.update({'b': net_b})

    def build_ema(self):
        if self.args.ema:
            self.ema_helpers = {}

            if self.first_pass and self.resume:
                # sample network
                sample_net_b = get_model(self.args)

                if self.resume:
                    sample_net_b.load_state_dict(
                        torch.load(self.sample_checkpoint_b))
                    sample_net_b = sample_net_b.to(self.device)
                    self.update_ema('b')
                    self.ema_helpers['b'].register(sample_net_b)

    def train(self):
        for n in range(self.checkpoint_it, self.n_ipf + 1):
            self.accelerator.print('RF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            self.ipf_iter('b', n)

    def build_optimizers(self, forward_or_backward=None):
        optimizer_b = get_optimizer(self.net['b'], self.args)

        if self.first_pass and self.resume:
            if self.resume:
                optimizer_b.load_state_dict(torch.load(self.optimizer_checkpoint_b))

        if forward_or_backward is None:
            self.optimizer = {'b': optimizer_b}
        if forward_or_backward == 'b':
            self.optimizer.update({'b': optimizer_b})

    def find_last_ckpt(self):
        existing_ckpts_dict = {}
        for ckpt_prefix in self.ckpt_prefixes:
            existing_ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_**.ckpt")))
            existing_ckpts_dict[ckpt_prefix] = set([os.path.basename(existing_ckpt)[len(ckpt_prefix)+1:-5] for existing_ckpt in existing_ckpts])
        
        existing_ckpts_b = sorted(list(existing_ckpts_dict["net_b"].intersection(existing_ckpts_dict["sample_net_b"], existing_ckpts_dict["optimizer_b"])), reverse=True)

        if len(existing_ckpts_b) == 0:
            return False, 1, 'b', 1, None
        
        def return_valid_ckpt_combi(b_i, b_n):
            # Return is_valid, checkpoint_it, checkpoint_pass, checkpoint_step
            if (b_n == 1 and b_i != self.first_num_iter) or (b_n > 1 and b_i != self.num_iter):  # during b pass
                return True, b_n, 'b', b_i + 1
            else:  
                return True, b_n + 1, 'b', 1

        for existing_ckpt_b in existing_ckpts_b:
            ckpt_b_n, ckpt_b_i = existing_ckpt_b.split("_")
            ckpt_b_n, ckpt_b_i = int(ckpt_b_n), int(ckpt_b_i)
            
            is_valid, checkpoint_it, checkpoint_pass, checkpoint_step = return_valid_ckpt_combi(ckpt_b_i, ckpt_b_n)
            if is_valid:
                break

        if not is_valid:
            return False, 1, 'b', 1, None
        else:
            return True, checkpoint_it, checkpoint_pass, checkpoint_step, existing_ckpt_b

    def apply_net(self, x, y, t, net, fb, return_scale=False):
        out = net.forward(x, y, t)

        if return_scale:
            return out, 1
        else:
            return out

    def compute_prev_it(self, forward_or_backward, n):
        assert forward_or_backward == 'b'
        prev_direction = 'b'
        prev_n = n - 1
        return prev_direction, prev_n

    def compute_next_it(self, forward_or_backward, n):
        assert forward_or_backward == 'b'
        next_direction = 'b'
        next_n = n+1
        return next_direction, next_n