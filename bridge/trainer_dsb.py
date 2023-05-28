import os, sys, warnings, time
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from .data import CacheLoader
from .sde import Langevin
from .runners import *
from .runners.config_getters import get_model, get_optimizer, get_plotter, get_logger
from .runners.ema import EMAHelper


class IPF_DSB:
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, final_cond_model=None,
                 valid_ds=None, test_ds=None):
        self.accelerator = accelerator
        self.device = self.accelerator.device  # local device for each process

        self.args = args
        self.cdsb = self.args.cdsb  # Conditional
        assert self.cdsb is False

        self.init_ds = init_ds
        self.final_ds = final_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.mean_final = mean_final
        self.var_final = var_final
        self.std_final = torch.sqrt(self.var_final) if self.var_final is not None else None

        self.transfer = self.args.transfer

        # training params
        self.n_ipf = self.args.n_ipf
        self.num_steps = self.args.num_steps
        self.batch_size = self.args.batch_size
        self.num_repeat_data = 1
        self.num_iter = self.args.num_iter
        self.grad_clipping = self.args.grad_clipping

        if self.args.symmetric_gamma:
            n = self.num_steps // 2
            if self.args.gamma_space == 'linspace':
                gamma_half = np.linspace(self.args.gamma_min, self.args.gamma_max, n)
            elif self.args.gamma_space == 'geomspace':
                gamma_half = np.geomspace(self.args.gamma_min, self.args.gamma_max, n)
            self.gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        else:
            if self.args.gamma_space == 'linspace':
                self.gammas = np.linspace(self.args.gamma_min, self.args.gamma_max, self.num_steps)
            elif self.args.gamma_space == 'geomspace':
                self.gammas = np.geomspace(self.args.gamma_min, self.args.gamma_max, self.num_steps)
        self.gammas = torch.tensor(self.gammas).to(self.device).float()
        self.T = torch.sum(self.gammas)
        self.accelerator.print("T:", self.T.item())

        # run from checkpoint
        self.resume = self.args.get('checkpoint_run', False)
        if self.resume:
            self.checkpoint_it = self.args.checkpoint_it
            self.checkpoint_pass = self.args.checkpoint_pass
            self.step = self.args.checkpoint_step + 1
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'
            self.step = 1
        
        # get models
        self.first_pass = True  # Load and use checkpointed networks during first pass
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers()

        # get loggers
        self.logger = self.get_logger('train_logs')
        self.save_logger = self.get_logger('test_logs')

        # langevin
        self.time_sampler = None

        # get data
        self.build_dataloaders()

        self.npar = len(init_ds)
        self.cache_npar = self.args.cache_npar if self.args.cache_npar is not None else self.batch_size * self.args.cache_refresh_stride // self.num_repeat_data
        self.cache_epochs = (self.batch_size * self.args.cache_refresh_stride) / (self.cache_npar * self.num_steps)
        self.data_epochs = (self.num_iter * self.cache_npar) / (self.npar * self.args.cache_refresh_stride)
        self.accelerator.print("Cache epochs:", self.cache_epochs)
        self.accelerator.print("Data epochs:", self.data_epochs)

        self.test_num_steps = self.num_steps
        self.plotter = self.get_plotter()
        self.cache_dir='./cache/'

        if self.accelerator.is_main_process:
            ckpt_dir = './checkpoints/'
            os.makedirs(ckpt_dir, exist_ok=True)
            existing_versions = []
            for d in os.listdir(ckpt_dir):
                if os.path.isdir(os.path.join(ckpt_dir, d)) and d.startswith("version_"):
                    existing_versions.append(int(d.split("_")[1]))

            if len(existing_versions) == 0:
                version = 0
            else:
                version = max(existing_versions) + 1

            self.ckpt_dir = os.path.join(ckpt_dir, f"version_{version}")
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.stride = self.args.gif_stride
        self.stride_log = self.args.log_stride

    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def build_models(self, forward_or_backward=None):
        # running network
        net_f, net_b = get_model(self.args), get_model(self.args)

        if self.first_pass and self.resume:
            if self.args.checkpoint_f is not None:
                try:
                    net_f.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_f)))
                except:
                    state_dict = torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_f))
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_f.load_state_dict(new_state_dict)

            if self.args.checkpoint_b is not None:
                try:
                    net_b.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_b)))
                except:
                    state_dict = torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_b))
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_b.load_state_dict(new_state_dict)

        if forward_or_backward is None:
            net_f = self.accelerator.prepare(net_f)
            net_b = self.accelerator.prepare(net_b)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})
        if forward_or_backward == 'f':
            net_f = self.accelerator.prepare(net_f)
            self.net.update({'f': net_f})
        if forward_or_backward == 'b':
            net_b = self.accelerator.prepare(net_b)
            self.net.update({'b': net_b})

    def accelerate(self, forward_or_backward):
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        if self.args.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(mu=self.args.ema_rate, device=self.device)
            self.ema_helpers[forward_or_backward].register(self.accelerator.unwrap_model(self.net[forward_or_backward]))

    def build_ema(self):
        if self.args.ema:
            self.ema_helpers = {}

            if self.first_pass and self.resume:
                # sample network
                sample_net_f, sample_net_b = get_model(self.args), get_model(self.args)

                if self.args.sample_checkpoint_f is not None:
                    sample_net_f.load_state_dict(
                        torch.load(hydra.utils.to_absolute_path(self.args.sample_checkpoint_f)))
                    sample_net_f = sample_net_f.to(self.device)
                    self.update_ema('f')
                    self.ema_helpers['f'].register(sample_net_f)
                if self.args.sample_checkpoint_b is not None:
                    sample_net_b.load_state_dict(
                        torch.load(hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)))
                    sample_net_b = sample_net_b.to(self.device)
                    self.update_ema('b')
                    self.ema_helpers['b'].register(sample_net_b)

    def build_dataloader(self, ds, batch_size, shuffle=True, drop_last=True, repeat=True):
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index * self.args.num_workers)
        dl_kwargs = {"num_workers": self.args.num_workers,
                     "pin_memory": self.args.pin_memory,
                     "worker_init_fn": worker_init_fn}

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **dl_kwargs)
        dl = self.accelerator.prepare(dl)
        if repeat:
            dl = repeater(dl)
        return dl

    def build_dataloaders(self):
        self.plot_npar = min(self.args.plot_npar, len(self.init_ds))
        self.test_npar = min(self.args.test_npar, len(self.init_ds))

        self.cache_batch_size = min(self.args.cache_batch_size * self.accelerator.num_processes, len(self.init_ds))  # Adjust automatically to num_processes
        self.test_batch_size = min(self.args.test_batch_size * self.accelerator.num_processes, len(self.init_ds))    # Adjust automatically to num_processes

        self.init_dl = self.build_dataloader(self.init_ds, batch_size=self.batch_size // self.num_repeat_data)
        self.cache_init_dl = self.build_dataloader(self.init_ds, batch_size=self.cache_batch_size)

        self.save_init_dl = self.build_dataloader(self.init_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
        self.save_dls_dict = {"train": self.save_init_dl}

        if self.valid_ds is not None:
            self.save_valid_dl = self.build_dataloader(self.valid_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
            self.save_dls_dict["valid"] = self.save_valid_dl

        if self.test_ds is not None:
            self.save_test_dl = self.build_dataloader(self.test_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
            self.save_dls_dict["test"] = self.save_test_dl

        if self.final_ds is not None:
            self.final_dl = self.build_dataloader(self.final_ds, batch_size=self.batch_size // self.num_repeat_data)
            self.cache_final_dl = self.build_dataloader(self.final_ds, batch_size=self.cache_batch_size)

            self.save_final_dl_repeat = self.build_dataloader(self.final_ds, batch_size=self.test_batch_size)
            self.save_final_dl = self.build_dataloader(self.final_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
        else:
            self.final_dl = None
            self.cache_final_dl = None

            self.save_final_dl = None
            self.save_final_dl_repeat = None

        batch = next(self.cache_init_dl)
        batch_x = batch[0]
        batch_y = batch[1]
        shape_x = batch_x[0].shape
        shape_y = batch_y[0].shape
        self.shape_x = shape_x
        self.shape_y = shape_y


        self.langevin = Langevin(self.num_steps, shape_x, shape_y, self.gammas, self.time_sampler,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=self.args.mean_match, out_scale=self.args.langevin_scale,
                                 var_final_gamma_scale=self.args.var_final_gamma_scale)

    def get_sample_net(self, fb):
        if self.args.ema:
            sample_net = self.ema_helpers[fb].ema_copy(self.accelerator.unwrap_model(self.net[fb]))
        else:
            sample_net = self.net[fb]
        sample_net = sample_net.to(self.device)
        sample_net.eval()
        return sample_net

    def new_cacheloader(self, forward_or_backward, n):
        if (n == 1) and (forward_or_backward == 'b'):
            sample_net = None
        else:
            sample_direction = 'f' if forward_or_backward == 'b' else 'b'
            sample_net = self.get_sample_net(sample_direction)

        cache_npar = self.cache_npar
        assert cache_npar % self.cache_batch_size == 0
        num_batches = cache_npar // self.cache_batch_size
        
        if forward_or_backward == 'b':
            new_ds = CacheLoader('b',
                                 sample_net,
                                 self.cache_init_dl,
                                 self.cache_final_dl,
                                 num_batches,
                                 self.langevin, self, n)

        else:  # forward
            new_ds = CacheLoader('f',
                                 sample_net,
                                 self.cache_init_dl,
                                 self.cache_final_dl,
                                 num_batches,
                                 self.langevin, self, n)

        assert self.batch_size % self.accelerator.num_processes == 0
        new_dl = DataLoader(new_ds, batch_size=self.batch_size // self.accelerator.num_processes, shuffle=True,
                            drop_last=True, pin_memory=self.args.pin_memory)

        # new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        for n in range(self.checkpoint_it, self.n_ipf + 1):

            self.accelerator.print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_iter('f', n)
            else:
                self.ipf_iter('b', n)
                self.ipf_iter('f', n)

    def sample_batch(self, init_dl, final_dl):
        mean_final = self.mean_final
        std_final = self.std_final

        init_batch = next(init_dl)
        init_batch_x = init_batch[0]
        init_batch_y = init_batch[1]

        if self.final_ds is not None:
            final_batch = next(final_dl)
            final_batch_x = final_batch[0]

        else:
            mean_final = mean_final.to(init_batch_x.device)
            std_final = std_final.to(init_batch_x.device)
            final_batch_x = mean_final + std_final * torch.randn_like(init_batch_x)

        mean_final = mean_final.to(init_batch_x.device)
        std_final = std_final.to(init_batch_x.device)
        var_final = std_final ** 2

        if not self.cdsb:
            init_batch_y = None

        return init_batch_x, init_batch_y, final_batch_x, mean_final, var_final

    def backward_sample(self, final_batch_x, y_c, fix_seed=False, sample_net=None, var_final=None, permute=True, num_steps=None):
        if sample_net is None:
            sample_net = self.get_sample_net('b')
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            final_batch_x = final_batch_x.to(self.device)
            # y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
            x_tot_c, _, _, _ = self.langevin.record_langevin_seq(sample_net, final_batch_x, y_c, 'b', sample=True,
                                                                 var_final=var_final)

            if permute:
                x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot_c, self.num_steps 

    def forward_sample(self, init_batch_x, init_batch_y, fix_seed=False, sample_net=None, permute=True, num_steps=None):
        if sample_net is None:
            sample_net = self.get_sample_net('f')
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            init_batch_x = init_batch_x.to(self.device)
            # init_batch_y = init_batch_y.to(self.device)
            # assert not self.cond_final
            mean_final = self.mean_final.to(self.device)
            var_final = self.var_final.to(self.device)
            # if n == 0:

            #     x_tot, _, _, _ = self.langevin.record_init_langevin(init_batch_x, init_batch_y,
            #                                                         mean_final=mean_final, var_final=var_final)
            # else:
            x_tot, _, _, _ = self.langevin.record_langevin_seq(sample_net, init_batch_x, init_batch_y, 'f', sample=self.transfer,
                                                               var_final=var_final)

        if permute:
            x_tot = x_tot.permute(1, 0, *list(range(2, len(x_tot.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot, self.num_steps 

    def plot_and_test_step(self, i, n, fb):
        sample_net = self.get_sample_net(fb)
        self.set_seed(seed=0 + self.accelerator.process_index)
        test_metrics = self.plotter(i, n, fb)

        if self.accelerator.is_main_process:
            self.save_logger.log_metrics(test_metrics, step=self.compute_current_step(i, n))
        return test_metrics

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        self.accelerator.free_memory()
        torch.cuda.empty_cache()

    def build_optimizers(self, forward_or_backward=None):
        optimizer_f, optimizer_b = get_optimizer(self.net['f'], self.args), get_optimizer(self.net['b'], self.args)

        if self.first_pass and self.resume:
            if self.args.optimizer_checkpoint_f is not None:
                optimizer_f.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_f)))
            if self.args.optimizer_checkpoint_b is not None:
                optimizer_b.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)))

        if forward_or_backward is None:
            self.optimizer = {'f': optimizer_f, 'b': optimizer_b}
        if forward_or_backward == 'f':
            self.optimizer.update({'f': optimizer_f})
        if forward_or_backward == 'b':
            self.optimizer.update({'b': optimizer_b})

    def save_step(self, i, n, fb):
        if (self.first_pass and i == 1) or i % self.stride == 0 or i == self.num_iter:
            if self.accelerator.is_main_process:
                name_net = f'net_{fb}_{n:03}_{i:07}.ckpt'
                name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                torch.save(self.accelerator.unwrap_model(self.net[fb]).state_dict(), name_net_ckpt)
                name_opt = f'optimizer_{fb}_{n:03}_{i:07}.ckpt'
                name_opt_ckpt = os.path.join(self.ckpt_dir, name_opt)
                torch.save(self.optimizer[fb].optimizer.state_dict(), name_opt_ckpt)

                if self.args.ema:
                    sample_net = self.ema_helpers[fb].ema_copy(self.accelerator.unwrap_model(self.net[fb]))
                    name_net = f'sample_net_{fb}_{n:03}_{i:07}.ckpt'
                    name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                    torch.save(sample_net.state_dict(), name_net_ckpt)
                
                ckpt_prefixes = ["net_f", "sample_net_f", "optimizer_f", "net_b", "sample_net_b", "optimizer_b"]
                for ckpt_prefix in ckpt_prefixes:
                    existing_ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, f"{ckpt_prefix}_**.ckpt")))
                    for ckpt_i in range(max(len(existing_ckpts)-2, 0)):
                        os.remove(existing_ckpts[ckpt_i])

            self.plot_and_test_step(i, n, fb)

    def ipf_iter(self, forward_or_backward, n):
        torch.cuda.empty_cache()
        new_dl = self.new_cacheloader(forward_or_backward, n)

        if (not self.first_pass) and (not self.args.use_prev_net):
            self.build_models(forward_or_backward)
            self.build_optimizers(forward_or_backward)

        self.accelerate(forward_or_backward)

        if (forward_or_backward not in self.ema_helpers.keys()) or ((not self.first_pass) and (not self.args.use_prev_net)):
            self.update_ema(forward_or_backward)

        if self.first_pass:
            step = self.step
        else:
            step = 1

        for i in tqdm(range(step, self.num_iter + 1), mininterval=30):
            self.net[forward_or_backward].train()

            self.set_seed(seed=n * self.num_iter + i + self.accelerator.process_index)

            x, out, steps_expanded = next(new_dl)

            x = x.to(self.device)
            # y = y.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)

            eval_steps = self.num_steps - 1 - steps_expanded

            if self.args.mean_match:
                pred = self.net[forward_or_backward](x, None, eval_steps) - x
                loss = F.mse_loss(pred, out)

            else:
                pred = self.net[forward_or_backward](x, None, eval_steps)

                if isinstance(self.args.loss_scale, str):
                    if forward_or_backward == 'f':
                        gamma = self.gammas[eval_steps].view([eval_steps.shape[0]] + [1]*(len(pred.shape)-1))
                    elif forward_or_backward == 'b':
                        gamma = self.gammas[steps_expanded].view([steps_expanded.shape[0]] + [1]*(len(pred.shape)-1))
                    loss_scale = eval(self.args.loss_scale).to(self.device)
                else:
                    loss_scale = self.args.loss_scale

                loss = F.mse_loss(loss_scale*pred, loss_scale*out)

            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = self.accelerator.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.

            if i == 1 or i % self.stride_log == 0 or i == self.num_iter:
                self.logger.log_metrics({'fb': forward_or_backward,
                                         'ipf': n,
                                         'loss': loss,
                                         'grad_norm': total_norm,
                                         "cache_epochs": self.cache_epochs,
                                         "data_epochs": self.data_epochs}, step=self.compute_current_step(i, n))

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad(set_to_none=True)
            if self.args.ema:
                self.ema_helpers[forward_or_backward].update(self.accelerator.unwrap_model(self.net[forward_or_backward]))

            self.save_step(i, n, forward_or_backward)

            if (i % self.args.cache_refresh_stride == 0) and (i != self.num_iter):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(forward_or_backward, n)

        new_dl = None

        self.net[forward_or_backward] = self.accelerator.unwrap_model(self.net[forward_or_backward])
        self.clear()
        self.first_pass = False


    def compute_current_step(self, i, n):
        return i + self.num_iter*max(n-1, 0)