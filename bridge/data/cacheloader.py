import os, time, shutil
import glob
import numpy as np
from numpy.lib.format import open_memmap
import torch
from torch.utils.data import Dataset, TensorDataset
from bridge.data.utils import save_image


class MemMapTensorDataset(Dataset):
    def __init__(self, npy_file_list) -> None:
        self.npy_file_list = npy_file_list
        self.data_file_list = [np.load(npy_file, mmap_mode='r') for npy_file in self.npy_file_list]

    def __getitem__(self, index):
        out = []
        for data_file in self.data_file_list:
            data = torch.from_numpy(data_file[index])
            out = out + [d for d in data]
        return out

    def __len__(self):
        return len(self.data_file_list[0])


def CacheLoader(fb, sample_net, init_dl, final_dl, num_batches, langevin, ipf, n, device='cpu'):
    start = time.time()
    all_x = []
    # all_y = []
    all_out = []
    all_steps = []

    sample_direction = 'f' if fb == 'b' else 'b'

    for b in range(num_batches):
        init_batch_x, batch_y, final_batch_x, mean_final, var_final = ipf.sample_batch(init_dl, final_dl)
        if sample_direction == "f":
            batch_x = init_batch_x
        else:
            batch_x = final_batch_x

        with torch.no_grad():
            if (n == 1) & (fb == 'b'):
                x, y, out, steps_expanded = langevin.record_init_langevin(batch_x, batch_y,
                                                                          mean_final=mean_final,
                                                                          var_final=var_final)
            else:
                x, y, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch_x, batch_y, sample_direction, var_final=var_final)

            # store x, y, out, steps
            x = x.flatten(start_dim=0, end_dim=1).to(device)
            # y = y.flatten(start_dim=0, end_dim=1).to(device)
            out = out.flatten(start_dim=0, end_dim=1).to(device)
            steps_expanded = steps_expanded.flatten(start_dim=0, end_dim=1).to(device)

            all_x.append(x)
            # all_y.append(y)
            all_out.append(out)
            all_steps.append(steps_expanded)

    all_x = torch.cat(all_x, dim=0)
    # all_y = torch.cat(all_y, dim=0)
    all_out = torch.cat(all_out, dim=0)
    all_steps = torch.cat(all_steps, dim=0)

    stop = time.time()
    ipf.accelerator.print('Cache size: {0}'.format(all_x.shape))
    ipf.accelerator.print("Load time: {0}".format(stop-start))
    ipf.accelerator.print("Out mean: {0}".format(all_out.mean().item()))
    ipf.accelerator.print("Out std: {0}".format(all_out.std().item()))

    return TensorDataset(all_x, all_out, all_steps)


def DBDSB_CacheLoader(sample_direction, sample_fn, init_dl, final_dl, num_batches, langevin, ipf, n, refresh_idx=0, refresh_tot=1, device='cpu'):
    start = time.time()

    # New method, saving as npy
    cache_filename_npy = f'cache_{sample_direction}_{n:03}.npy'
    cache_filepath_npy = os.path.join(ipf.cache_dir, cache_filename_npy)

    cache_filename_txt = f'cache_{sample_direction}_{n:03}.txt'
    cache_filepath_txt = os.path.join(ipf.cache_dir, cache_filename_txt)

    if ipf.cdsb:
        cache_y_filename_npy = f'cache_y_{sample_direction}_{n:03}.npy'
        cache_y_filepath_npy = os.path.join(ipf.cache_dir, cache_y_filename_npy)

    # Temporary cache of each batch
    temp_cache_dir = os.path.join(ipf.cache_dir, f"temp_{sample_direction}_{n:03}_{refresh_idx:03}")
    os.makedirs(temp_cache_dir, exist_ok=True)

    npar = num_batches * ipf.cache_batch_size
    num_batches_dist = num_batches * ipf.accelerator.num_processes  # In distributed mode
    cache_batch_size_dist = ipf.cache_batch_size // ipf.accelerator.num_processes  # In distributed mode

    use_existing_cache = False
    if os.path.isfile(cache_filepath_txt):
        f = open(cache_filepath_txt, 'r')
        input = f.readline()
        f.close()
        input_list = input.split("/")
        if int(input_list[0]) == refresh_idx and int(input_list[1]) == refresh_tot:
            use_existing_cache = True
    
    if not use_existing_cache:
        sample = ((sample_direction == 'b') or ipf.transfer)
        normalize_x1 = ((not sample) and ipf.normalize_x1)

        x1_mean_list, x1_mse_list = [], []

        for b in range(num_batches):
            b_dist = b * ipf.accelerator.num_processes + ipf.accelerator.process_index

            try:
                batch_x0, batch_x1 = torch.load(os.path.join(temp_cache_dir, f"{b_dist}.pt"))
                if ipf.cdsb:
                    batch_y = torch.load(os.path.join(temp_cache_dir, f"{b_dist}_y.pt"))[0]
                assert len(batch_x0) == len(batch_x1) == cache_batch_size_dist
                batch_x0, batch_x1 = batch_x0.to(ipf.device), batch_x1.to(ipf.device)
            except:
                ipf.set_seed(seed=ipf.compute_current_step(0, n+1)*num_batches_dist*refresh_tot + num_batches_dist*refresh_idx + b_dist)

                init_batch_x, init_batch_y, final_batch_x, _, _ = ipf.sample_batch(init_dl, final_dl)

                with torch.no_grad():
                    batch_x0, batch_y, batch_x1 = langevin.generate_new_dataset(init_batch_x, init_batch_y, final_batch_x, sample_fn, sample_direction, sample=sample, num_steps=ipf.cache_num_steps)
                    batch_x0, batch_x1 = batch_x0.contiguous(), batch_x1.contiguous()
                    torch.save([batch_x0, batch_x1], os.path.join(temp_cache_dir, f"{b_dist}.pt"))
                    if ipf.cdsb:
                        torch.save([batch_y], os.path.join(temp_cache_dir, f"{b_dist}_y.pt"))
    
            if normalize_x1:
                x1_mean_list.append(batch_x1.mean(0))
                x1_mse_list.append(batch_x1.square().mean(0))
        
        if normalize_x1:
            x1_mean = torch.stack(x1_mean_list).mean(0)
            x1_mse = torch.stack(x1_mse_list).mean(0)
            reduced_x1_mean = ipf.accelerator.reduce(x1_mean, reduction='mean')
            reduced_x1_mse = ipf.accelerator.reduce(x1_mse, reduction='mean')
            reduced_x1_std = (reduced_x1_mse - reduced_x1_mean.square()).sqrt()
    
        ipf.accelerator.wait_for_everyone() 

        stop = time.time()
        ipf.accelerator.print("Load time: {0}".format(stop-start))

        # Aggregate temporary caches into central cache file
        if ipf.accelerator.is_main_process:
            fp = open_memmap(cache_filepath_npy, dtype='float32', mode='w+', shape=(npar, 2, *batch_x0.shape[1:]))
            if ipf.cdsb:
                fp_y = open_memmap(cache_y_filepath_npy, dtype='float32', mode='w+', shape=(npar, 1, *batch_y.shape[1:]))
            for b_dist in range(num_batches_dist):
                temp_cache_filepath_b_dist = os.path.join(temp_cache_dir, f"{b_dist}.pt")
                loaded = False
                while not loaded:
                    if not os.path.isfile(temp_cache_filepath_b_dist):
                        print(f"Index {ipf.accelerator.process_index} did not find temp cache file {b_dist}, retrying in 5 seconds")
                        time.sleep(5)
                    else:
                        try:
                            batch_x0, batch_x1 = torch.load(temp_cache_filepath_b_dist)
                            batch_x0, batch_x1 = batch_x0.to(ipf.device), batch_x1.to(ipf.device)
                            loaded = True
                        except:
                            print(f"Index {ipf.accelerator.process_index} failed to load cache file {b_dist}, retrying in 5 seconds")
                            time.sleep(5)

                assert len(batch_x0) == len(batch_x1) == cache_batch_size_dist

                if ipf.cdsb:
                    temp_cache_y_filepath_b_dist = os.path.join(temp_cache_dir, f"{b_dist}_y.pt")
                    loaded = False
                    while not loaded:
                        if not os.path.isfile(temp_cache_y_filepath_b_dist):
                            print(f"Index {ipf.accelerator.process_index} did not find temp cache file {b_dist}_y, retrying in 5 seconds")
                            time.sleep(5)
                        else:
                            try:
                                batch_y = torch.load(temp_cache_y_filepath_b_dist)[0]
                                loaded = True
                            except:
                                print(f"Index {ipf.accelerator.process_index} failed to load cache file {b_dist}_y, retrying in 5 seconds")
                                time.sleep(5)
                    assert len(batch_y) == cache_batch_size_dist
                
                if normalize_x1:
                    batch_x1 = (batch_x1 - reduced_x1_mean) / reduced_x1_std
    
                batch = torch.stack([batch_x0, batch_x1], dim=1).float().cpu().numpy()
                fp[b_dist*cache_batch_size_dist:(b_dist+1)*cache_batch_size_dist] = batch
                fp.flush()

                if ipf.cdsb:
                    batch_y = batch_y.unsqueeze(1).float().cpu().numpy()
                    fp_y[b_dist*cache_batch_size_dist:(b_dist+1)*cache_batch_size_dist] = batch_y
                    fp_y.flush()
            
            del fp
            if ipf.cdsb:
                del fp_y
                
            f = open(cache_filepath_txt, 'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
            f.write(f'{refresh_idx}/{refresh_tot}')
            f.close()

            shutil.rmtree(temp_cache_dir)
            
    ipf.accelerator.wait_for_everyone() 

    # All processes check that the cache is accessible
    loaded = False
    while not loaded:
        if not os.path.isfile(cache_filepath_npy):
            print("Index", ipf.accelerator.process_index, "did not find cache file, retrying in 5 seconds")
            time.sleep(5)
        else:
            try:
                fp = np.load(cache_filepath_npy, mmap_mode='r')
                loaded = True
            except:
                print("Index", ipf.accelerator.process_index, "failed to load cache file, retrying in 5 seconds")
                time.sleep(5)
    
    if ipf.cdsb:
        loaded = False
        while not loaded:
            if not os.path.isfile(cache_y_filepath_npy):
                print("Index", ipf.accelerator.process_index, "did not find cache_y file, retrying in 5 seconds")
                time.sleep(5)
            else:
                try:
                    fp_y = np.load(cache_y_filepath_npy, mmap_mode='r')
                    loaded = True
                except:
                    print("Index", ipf.accelerator.process_index, "failed to load cache_y file, retrying in 5 seconds")
                    time.sleep(5)

    ipf.accelerator.wait_for_everyone() 
    ipf.accelerator.print(f'Cache size: {fp.shape}')

    if ipf.accelerator.is_main_process:
        # Visualize first entries
        num_plots_grid = 100
        ipf.plotter.save_image(torch.from_numpy(fp[:num_plots_grid, 0]), f'cache_{sample_direction}_{n:03}_x0', "./", domain=0)
        ipf.plotter.save_image(torch.from_numpy(fp[:num_plots_grid, 1]), f'cache_{sample_direction}_{n:03}_x1', "./", domain=1)

        # Automatically delete old cache files
        for fb in ['f', 'b']:
            existing_cache_files = sorted(glob.glob(os.path.join(ipf.cache_dir, f"cache_{fb}_**.npy")))
            for ckpt_i in range(max(len(existing_cache_files)-1, 0)):
                if not os.path.samefile(existing_cache_files[ckpt_i], cache_filepath_npy):
                    os.remove(existing_cache_files[ckpt_i])

            if ipf.cdsb:
                existing_cache_files = sorted(glob.glob(os.path.join(ipf.cache_dir, f"cache_y_{fb}_**.npy")))
                for ckpt_i in range(max(len(existing_cache_files)-1, 0)):
                    if not os.path.samefile(existing_cache_files[ckpt_i], cache_filepath_npy):
                        os.remove(existing_cache_files[ckpt_i])

    del fp

    if ipf.cdsb:
        del fp_y
        return MemMapTensorDataset([cache_filepath_npy, cache_y_filepath_npy])

    return MemMapTensorDataset([cache_filepath_npy])