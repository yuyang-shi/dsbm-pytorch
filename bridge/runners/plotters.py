import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torchvision.utils as vutils
import hydra
import glob

from ..data.utils import save_image, to_uint8_tensor, normalize_tensor
from ..data.metrics import PSNR, SSIM, FID  # , LPIPS
from PIL import Image
# matplotlib.use('Agg')


DPI = 200

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)


class Plotter(object):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif'):
        self.ipf = ipf
        self.args = args
        self.plot_level = self.args.plot_level

        self.dataset = self.args.data.dataset
        self.num_steps = self.ipf.test_num_steps

        if self.ipf.accelerator.is_main_process:
            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.metrics_dict = {}

    def __call__(self, i, n, fb, sampler='sde'):
        assert sampler in ['sde', 'ode']
        out = {}
        self.step = self.ipf.compute_current_step(i, n)
        cache_filepath_npy = sorted(glob.glob(os.path.join(self.ipf.cache_dir, f"cache_{fb}_{n:03}.npy")))

        if self.ipf.accelerator.is_main_process:
            out['fb'] = fb
            out['ipf'] = n
            out['T'] = self.ipf.T

        for dl_name, dl in self.ipf.save_dls_dict.items():
            use_cache = ((dl_name == "train") and (sampler == 'sde') and (self.step >= self.ipf.compute_current_step(0, n+1)) and 
                         (len(cache_filepath_npy) > 0) and (self.ipf.cache_num_steps == self.num_steps))

            x_start, y_start, x_tot, x_init, mean_final, var_final, metric_results = \
                self.generate_sequence_joint(dl, i, n, fb, dl_name=dl_name, sampler=sampler)

            if self.ipf.accelerator.is_main_process:
                self.plot_sequence_joint(x_start[:self.args.plot_npar], y_start[:self.args.plot_npar],
                                         x_tot[:, :self.args.plot_npar], x_init[:self.args.plot_npar],
                                         self.dataset, i, n, fb, dl_name=dl_name, sampler=sampler,
                                         mean_final=mean_final, var_final=var_final)

            if use_cache and not self.ipf.cdsb:
                print("Using cached data for training set evaluation")
                fp = np.load(cache_filepath_npy[0], mmap_mode="r")
                all_x = torch.from_numpy(fp[:self.ipf.test_npar])
                if fb == 'f':
                    x_start, x_last = all_x[:, 0], all_x[:, 1]
                else:
                    x_start, x_last = all_x[:, 1], all_x[:, 0]
                y_start, x_init = [], []

            else:
                if fb == 'b' or (self.ipf.transfer and dl_name == 'train'):
                    generate_npar = self.ipf.test_npar
                else:
                    generate_npar = min(self.ipf.test_npar, self.ipf.plot_npar)
                x_start, y_start, x_tot, x_init, mean_final, var_final, metric_results = \
                    self.generate_sequence_joint(dl, i, n, fb, dl_name=dl_name, sampler=sampler, generate_npar=generate_npar, full_traj=False)
                x_last = x_tot[-1]

            x_tot = None

            test_results = self.test_joint(x_start[:self.ipf.test_npar], y_start[:self.ipf.test_npar], 
                                           x_last[:self.ipf.test_npar], x_init[:self.ipf.test_npar],
                                           i, n, fb, dl_name=dl_name, sampler=sampler,
                                           mean_final=mean_final, var_final=var_final)
            
            metric_results = {self.prefix_fn(dl_name, sampler) + k: v for k, v in metric_results.items()}
            test_results = {self.prefix_fn(dl_name, sampler) + k: v for k, v in test_results.items()}

            out.update(metric_results)
            out.update(test_results)

        torch.cuda.empty_cache()
        return out

    def prefix_fn(self, dl_name, sampler):
        assert sampler in ['sde', 'ode']
        if sampler == 'sde':
            return dl_name + '/'
        else:
            return dl_name + '/ode/'

    def generate_sequence_joint(self, dl, i, n, fb, dl_name='train', sampler='sde', generate_npar=None, full_traj=True):
        iter_dl = iter(dl)

        all_batch_x = []
        all_batch_y = []
        all_x_tot = []
        all_init_batch_x = []
        all_mean_final = []
        all_var_final = []
        times = []
        nfes = []
        metric_results = {}

        if generate_npar is None:
            generate_npar = self.ipf.plot_npar
        iters = 0
        while iters * self.ipf.test_batch_size < generate_npar:
            try:
                start = time.time()

                init_batch_x, batch_y, final_batch_x, mean_final, var_final = self.ipf.sample_batch(iter_dl, self.ipf.save_final_dl_repeat)

                with torch.no_grad():
                    if fb == 'f':
                        batch_x = init_batch_x
                        if sampler == 'ode':
                            x_tot, nfe = self.ipf.forward_sample_ode(batch_x, batch_y, permute=False)
                        else:
                            x_tot, nfe = self.ipf.forward_sample(batch_x, batch_y, permute=False, num_steps=self.num_steps)
                        # x_last_true = final_batch_x
                    else:
                        batch_x = final_batch_x
                        if sampler == 'ode':
                            x_tot, nfe = self.ipf.backward_sample_ode(batch_x, batch_y, permute=False)  # var_final=var_final, 
                        else:
                            x_tot, nfe = self.ipf.backward_sample(batch_x, batch_y, permute=False, num_steps=self.num_steps)  # var_final=var_final, 
                        # x_last_true = init_batch_x

                    stop = time.time()
                    times.append(stop - start)
                    nfes.append(nfe)

                    gather_batch_x = self.ipf.accelerator.gather(batch_x)
                    if self.ipf.cdsb:
                        gather_batch_y = self.ipf.accelerator.gather(batch_y)
                    gather_init_batch_x = self.ipf.accelerator.gather(init_batch_x)

                    if not full_traj:
                        x_tot = x_tot[:, -1:].contiguous()
                        gather_x_tot = self.ipf.accelerator.gather(x_tot)
                    else:
                        gather_x_tot = x_tot

                    all_batch_x.append(gather_batch_x.cpu())
                    if self.ipf.cdsb:
                        all_batch_y.append(gather_batch_y.cpu())
                    all_x_tot.append(gather_x_tot.cpu())
                    all_init_batch_x.append(gather_init_batch_x.cpu())

                    iters = iters + 1

            except StopIteration:
                break

        all_batch_x = torch.cat(all_batch_x, dim=0)
        if self.ipf.cdsb:
            all_batch_y = torch.cat(all_batch_y, dim=0)
        all_x_tot = torch.cat(all_x_tot, dim=0)
        all_init_batch_x = torch.cat(all_init_batch_x, dim=0)

        shape_len = len(all_x_tot.shape)
        all_x_tot = all_x_tot.permute(1, 0, *list(range(2, shape_len)))

        all_mean_final = self.ipf.mean_final.cpu()
        all_var_final = self.ipf.var_final.cpu()

        metric_results['nfe'] = np.mean(nfes)
        metric_results['batch_sample_time'] = np.mean(times)

        return all_batch_x, all_batch_y, all_x_tot, all_init_batch_x, all_mean_final, all_var_final, metric_results

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None):
        pass

    def test_joint(self, x_start, y_start, x_last, x_init, i, n, fb, dl_name='train', sampler='sde', mean_final=None, var_final=None):
        out = {}
        metric_results = {}

        x_var_last = torch.var(x_last, dim=0).mean().item()
        x_var_start = torch.var(x_start, dim=0).mean().item()
        x_mean_last = torch.mean(x_last).item()
        x_mean_start = torch.mean(x_start).item()

        x_mse_start_last = torch.mean((x_start - x_last) ** 2).item()

        out = {'x_mean_start': x_mean_start, 'x_var_start': x_var_start,
                'x_mean_last': x_mean_last, 'x_var_last': x_var_last, 
                'x_mse_start_last': x_mse_start_last}

        if mean_final is not None:
            x_mse_last = torch.mean((x_last - mean_final) ** 2).item()
            x_mse_start = torch.mean((x_start - mean_final) ** 2).item()
            out.update({"x_mse_start": x_mse_start, "x_mse_last": x_mse_last})

        if fb == 'b' or (self.ipf.transfer and dl_name == 'train'):
            dl_x_start = self.ipf.build_dataloader(x_start, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_start = iter(dl_x_start)
            if self.ipf.cdsb and len(y_start) > 0:
                dl_y_start = self.ipf.build_dataloader(y_start, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
                dl_y_start = iter(dl_y_start)
            else:
                dl_y_start = None
            dl_x_last = self.ipf.build_dataloader(x_last, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_last = iter(dl_x_last)
            if len(x_init) > 0:
                dl_x_init = self.ipf.build_dataloader(x_init, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
                dl_x_init = iter(dl_x_init)
            else:
                dl_x_init = None
            dl_x_last_true = self.ipf.save_dls_dict[dl_name] if fb == 'b' else self.ipf.save_final_dl
            dl_x_last_true = iter(dl_x_last_true)
            for metric_name, metric in self.metrics_dict.items():
                metric.reset()
                
            iters = 0
            while iters * self.ipf.test_batch_size < self.ipf.test_npar:
                try:
                    x_start, x_last = next(dl_x_start), next(dl_x_last)
                    if dl_y_start is not None:
                        y_start = next(dl_y_start)
                    else:
                        y_start = None
                    if dl_x_init is not None:
                        x_init = next(dl_x_init)
                    else:
                        x_init = None
                    x_last_true, _ = next(dl_x_last_true)
                    self.plot_and_record_batch_joint(x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name=dl_name, sampler=sampler)
                    iters = iters + 1
                
                except StopIteration:
                    break

            if iters > 0:
                for metric_name, metric in self.metrics_dict.items():
                    metric_result = metric.compute()
                    if self.ipf.accelerator.is_main_process:
                        metric_results[metric_name] = metric_result
                    metric.reset()
        
        out.update(metric_results)
        out.update({'test_npar': self.ipf.test_npar})
        return out

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        pass

    def save_image(self, tensor, name, dir, **kwargs):
        return []


class ImPlotter(Plotter):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif'):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir)
        self.num_plots_grid = 100

        self.metrics_dict = {"fid": FID().to(self.ipf.device)}

        if self.dataset == "CIFAR10":
            data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
            root = os.path.join(data_dir, 'cifar10')
            fid_stats = torch.load(os.path.join(root, 'fid_stats.pt'))
            self.metrics_dict["fid"].real_features_sum = fid_stats["real_features_sum"].to(self.ipf.device)
            self.metrics_dict["fid"].real_features_cov_sum = fid_stats["real_features_cov_sum"].to(self.ipf.device)
            self.metrics_dict["fid"].real_features_num_samples = fid_stats["real_features_num_samples"].to(self.ipf.device)
            self.metrics_dict["fid"].reset_real_features = False

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, freq=freq, dl_name=dl_name, sampler=sampler,
                                    mean_final=mean_final, var_final=var_final)
        num_steps = x_tot.shape[0]
        if freq is None:
            freq = num_steps // min(num_steps, 50)

        if self.plot_level >= 1:
            x_tot_grid = x_tot[:, :self.num_plots_grid]
            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, self.prefix_fn(dl_name, sampler))
            gif_dir = os.path.join(self.gif_dir, self.prefix_fn(dl_name, sampler))

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            filename_grid = 'im_grid_start'
            filepath_grid_list = self.save_image(x_start[:self.num_plots_grid], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_last'
            filepath_grid_list = self.save_image(x_tot_grid[-1], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_data_x'
            filepath_grid_list = self.save_image(x_init[:self.num_plots_grid], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            if self.plot_level >= 2:
                plot_paths = []
                x_start_tot_grid = torch.cat([x_start[:self.num_plots_grid].unsqueeze(0), x_tot_grid], dim=0)
                for k in range(num_steps+1):
                    if k % freq == 0 or k == num_steps:
                        # save png
                        filename_grid = 'im_grid_{0}'.format(k)
                        filepath_grid_list = self.save_image(x_start_tot_grid[k], filename_grid, im_dir)
                        plot_paths.append(filepath_grid_list[0])

                make_gif(plot_paths, output_directory=gif_dir, gif_name=name+'_im_grid')

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        if fb == 'b' or self.ipf.transfer:
            uint8_x_last_true = to_uint8_tensor(x_last_true)
            uint8_x_last = to_uint8_tensor(x_last)

            for metric in self.metrics_dict.values():
                metric.update(uint8_x_last, uint8_x_last_true)

            # if self.plot_level >= 3:
            #     name = str(i) + '_' + fb + '_' + str(n)
            #     im_dir = os.path.join(self.im_dir, name, dl_name)
            #     im_dir = os.path.join(im_dir, "im/")
            #     os.makedirs(im_dir, exist_ok=True)

            #     for k in range(x_last.shape[0]):
            #         plt.clf()
            #         file_idx = iters * self.ipf.test_batch_size + self.ipf.accelerator.process_index * self.ipf.test_batch_size // self.ipf.accelerator.num_processes + k
            #         filename_png = os.path.join(im_dir, '{:05}.png'.format(file_idx))
            #         assert not os.path.isfile(filename_png)
            #         save_image(x_last[k], filename_png)

    def save_image(self, tensor, name, dir, **kwargs):
        fp = os.path.join(dir, f'{name}.png')
        save_image(tensor[:self.num_plots_grid], fp, nrow=10)
        return [fp]


class DownscalerPlotter(Plotter):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif'):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir)
        self.num_plots_grid = 16
        assert self.ipf.cdsb

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, freq=freq, dl_name=dl_name, sampler=sampler,
                                    mean_final=mean_final, var_final=var_final)
        num_steps = x_tot.shape[0]
        if freq is None:
            freq = num_steps // min(num_steps, 50)

        if self.plot_level >= 1:
            x_tot_grid = x_tot[:, :self.num_plots_grid]
            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, self.prefix_fn(dl_name, sampler))
            gif_dir = os.path.join(self.gif_dir, self.prefix_fn(dl_name, sampler))

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            filename_grid = 'im_grid_start'
            filepath_grid_list = self.save_image(x_start[:self.num_plots_grid], filename_grid, im_dir, domain=0 if fb=='f' else 1)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_last'
            filepath_grid_list = self.save_image(x_tot_grid[-1], filename_grid, im_dir, domain=1 if fb=='f' else 0)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_data_x'
            filepath_grid_list = self.save_image(x_init[:self.num_plots_grid], filename_grid, im_dir, domain=0)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            # Save y differently (no processing needed)
            filename_grid = 'im_grid_data_y'
            filepath_grid = os.path.join(im_dir, f'{filename_grid}.png')
            save_image(y_start[:self.num_plots_grid], filepath_grid, normalize=True, nrow=4)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, [filepath_grid], step=self.step, fb=fb)

            if self.plot_level >= 2:
                plot_paths = []
                x_start_tot_grid = torch.cat([x_start[:self.num_plots_grid].unsqueeze(0), x_tot_grid], dim=0)
                for k in range(num_steps+1):
                    if k % freq == 0 or k == num_steps:
                        # save png
                        filename_grid = 'im_grid_{0}'.format(k)
                        filepath_grid_list = self.save_image(x_start_tot_grid[k], filename_grid, im_dir, domain=1 if fb=='f' else 0)
                        plot_paths.append(filepath_grid_list)

                for d in [0, 1]:
                    make_gif([plot_path[d] for plot_path in plot_paths], output_directory=gif_dir, gif_name=f'{name}_dim_{d}_im_grid')

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        if fb == 'b' or self.ipf.transfer:            
            if self.plot_level >= 3:
                name = str(i) + '_' + fb + '_' + str(n)
                im_dir = os.path.join(self.im_dir, name, dl_name)
                inner_im_dir = os.path.join(im_dir, "im/")
                os.makedirs(inner_im_dir, exist_ok=True)
                
                file_idx = iters * self.ipf.accelerator.num_processes + self.ipf.accelerator.process_index

                filename = 'im_start'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_start.cpu().numpy())

                filename = 'im_last'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_last.cpu().numpy())

                filename = 'im_data_x'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_init.cpu().numpy())

                filename = 'im_data_y'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, y_start.cpu().numpy())

    def save_image(self, tensor, name, dir, domain=0):
        assert domain in [0, 1]
        fp_list = []
        if domain == 0:
            inverted_tensor, _ = self.ipf.init_ds.invert_preprocessing(tensor)
        else:
            inverted_tensor, _ = self.ipf.final_ds.invert_preprocessing(tensor)
        inverted_tensor = vutils.make_grid(inverted_tensor[:self.num_plots_grid], nrow=4)

        d = 0
        fp = os.path.join(dir, f'dim_{d}_{name}.png')
        plt.imsave(fp, inverted_tensor[0], vmin=-30, vmax=5, cmap='Blues_r')
        fp_list.append(fp)

        d = 1
        fp = os.path.join(dir, f'dim_{d}_{name}.png')
        plt.imsave(fp, inverted_tensor[1], vmin=-25, vmax=25, cmap='bwr_r')
        fp_list.append(fp)

        return fp_list