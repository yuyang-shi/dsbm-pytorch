import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import copy
import ot as pot

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


device = 'cpu'
dataset_size = 10000
test_dataset_size = 10000
lr = 1e-4
batch_size = 128


class MLP(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super(MLP, self).__init__()
    layers = []
    prev_width = input_dim
    for layer_width in layer_widths:
      layers.append(torch.nn.Linear(prev_width, layer_width))
      prev_width = layer_width
    self.input_dim = input_dim
    self.layer_widths = layer_widths
    self.layers = nn.ModuleList(layers)
    self.activate_final = activate_final
    self.activation_fn = activation_fn
        
  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      x = self.activation_fn(layer(x))
    x = self.layers[-1](x)
    if self.activate_final:
      x = self.activation_fn(x)
    return x


class ScoreNetwork(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super().__init__()
    self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)
    
  def forward(self, x_input, t):
    inputs = torch.cat([x_input, t], dim=1)
    return self.net(inputs)


# Original DSB
class DSB(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=20, sig=0):
    super().__init__()
    self.net_fwd = net_fwd
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
    self.N = num_steps
    self.sig = sig
  
  @torch.no_grad()
  def generate_new_dataset_and_train_tuple(self, x_pairs=None, fb='', first_it=False):
    assert fb in ['f', 'b']

    if fb == 'f':
      prev_fb = 'b'
      zstart = x_pairs[:, 1]
    else:
      prev_fb = 'f'
      zstart = x_pairs[:, 0]

    N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    signal = []
    tlist = []
    z = zstart.detach().clone()
    batchsize = zstart.shape[0]
    dim = zstart.shape[1]
    
    ts = np.arange(N) / N
    tl = np.arange(1, N+1) / N
    if prev_fb == 'b':
      ts = 1 - ts
      tl = 1 - tl
      
    if first_it:
      assert prev_fb == 'f'
      for i in range(N):
        t = torch.ones((batchsize,1), device=device) * ts[i]
        dz = self.sig * torch.randn_like(z) * np.sqrt(dt)
        z = z + dz
        tlist.append(torch.ones((batchsize,1), device=device) * tl[i])
        traj.append(z.detach().clone())
        signal.append(-dz)
    else:
      for i in range(N):
        t = torch.ones((batchsize,1), device=device) * ts[i]
        pred = self.net_dict[prev_fb](z, t)
        z = z.detach().clone() + pred
        dz = self.sig * torch.randn_like(z) * np.sqrt(dt)
        z = z + dz
        tlist.append(torch.ones((batchsize,1), device=device) * tl[i])
        traj.append(z.detach().clone())
        signal.append(- self.net_dict[prev_fb](z, t) - dz)
    
    z_t = torch.stack(traj)
    tlist = torch.stack(tlist)
    target = torch.stack(signal)
    
    randint = torch.randint(N, (1, batchsize, 1), device=device)
    tlist = torch.gather(tlist, 0, randint).squeeze(0)
    z_t = torch.gather(z_t, 0, randint.expand(1, batchsize, dim)).squeeze(0)
    target = torch.gather(target, 0, randint.expand(1, batchsize, dim)).squeeze(0)
    return z_t, tlist, target

  @torch.no_grad()
  def sample_sde(self, zstart=None, fb='', first_it=False, N=None):
    assert fb in ['f', 'b']

    ### NOTE: Use Euler method to sample from the learned flow
    N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)
      z = z.detach().clone() + pred
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
      traj.append(z.detach().clone())
    return traj


def train_dsb_ipf(dsb_ipf, x_pairs, batch_size, inner_iters, fb='', first_it=False, **kwargs):
  assert fb in ['f', 'b']
  dsb_ipf.fb = fb
  optimizer = torch.optim.Adam(dsb_ipf.net_dict[fb].parameters(), lr=lr)
  # optimizer = dsb_ipf.optimizer_dict[fb]
  loss_curve = []
  
  z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)
  dl = iter(DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in tqdm(range(inner_iters)):
    try:
      z_t, t, target = next(dl)
    except StopIteration:
      z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)
      dl = iter(DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z_t, t, target = next(dl)
    
    optimizer.zero_grad()
    pred = dsb_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return dsb_ipf, loss_curve


# DSBM
class DSBM(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
    super().__init__()
    self.net_fwd = net_fwd
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    self.first_coupling = first_coupling
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    if fb == 'f':
      # z1 - z_t / (1-t)
      target = z1 - z0 
      target = target - self.sig * torch.sqrt(t/(1.-t)) * z
    else:
      # z0 - z_t / t
      target = - (z1 - z0)
      target = target - self.sig * torch.sqrt((1.-t)/t) * z
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    assert fb in ['f', 'b']

    if prev_model is None:
      assert first_it
      assert fb == 'b'
      zstart = x_pairs[:, 0]
      if self.first_coupling == "ref":
        # First coupling is x_0, x_0 perturbed
        zend = zstart + torch.randn_like(zstart) * self.sig
      elif self.first_coupling == "ind":
        zend = x_pairs[:, 1].clone()
        zend = zend[torch.randperm(len(zend))]
      else:
        raise NotImplementedError
      z0, z1 = zstart, zend
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:
        zstart = x_pairs[:, 1]
      zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N   
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)
      z = z.detach().clone() + pred * dt
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
      traj.append(z.detach().clone())

    return traj


def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
  assert fb in ['f', 'b']
  dsbm_ipf.fb = fb
  optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)
  # optimizer = dsbm_ipf.optimizer_dict[fb]
  loss_curve = []
  
  dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in tqdm(range(inner_iters)):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)
    
    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = dsbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
    optimizer.zero_grad()
    pred = dsbm_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return dsbm_ipf, loss_curve


# SB-CFM
class SBCFM(nn.Module):
  def __init__(self, net=None, num_steps=1000, sig=0, eps=1e-3):
    super().__init__()
    self.net = net
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    from bridge.sde.optimal_transport import OTPlanSampler
    self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * sig**2)
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, **kwargs):
    x0, x1 = x_pairs[:, 0], x_pairs[:, 1]
    z0, z1 = self.ot_sampler.sample_plan(x0, x1)

    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    target = z1 - z0 
    target = target - self.sig * (torch.sqrt(t)/torch.sqrt(1.-t) - 0.5 / torch.sqrt(t*(1.-t))) * z
    return z_t, t, target
    
  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, **kwargs):
    return x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]

  @torch.no_grad()
  def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    sign = 1 if fb == 'f' else -1
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = sign * self.net(z, t)
      z = z.detach().clone() + pred * dt
      traj.append(z.detach().clone())

    return traj


# Rectified Flow
class RectifiedFlow(nn.Module):
  def __init__(self, net=None, num_steps=1000, sig=0, eps=0):
    super().__init__()
    self.net = net
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
    self.N = num_steps
    self.sig = sig
    self.eps = eps
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', first_it=False):
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]

    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    target = z1 - z0
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    if prev_model is None:
      assert first_it
      z0, z1 = x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:
        zstart = x_pairs[:, 1]
      zend = prev_model.sample_ode(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    sign = 1 if fb == 'f' else -1
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = sign * self.net(z, t)
      z = z.detach().clone() + pred * dt
      traj.append(z.detach().clone())

    return traj


def train_flow_model(flow_model, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
  assert fb in ['f', 'b']
  flow_model.fb = fb
  optimizer = flow_model.optimizer
  loss_curve = []
  
  dl = iter(DataLoader(TensorDataset(*flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in tqdm(range(inner_iters)):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)

    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = flow_model.get_train_tuple(x_pairs=z_pairs, fb=fb, first_it=first_it)

    optimizer.zero_grad()
    pred = flow_model.net(z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return flow_model, loss_curve


@torch.no_grad()
def draw_plot(sample_fn, z0, z1, N=None):
  traj = sample_fn(N=N)
  
  plt.figure(figsize=(4,4))
  plt.xlim(-5,5)
  plt.ylim(-5,5)
    
  plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
  plt.legend()
  plt.title('Distribution')
  plt.tight_layout()

  # traj_particles = torch.stack(traj)
  # plt.figure(figsize=(4,4))
  # plt.xlim(-5,5)
  # plt.ylim(-5,5)
  # plt.axis('equal')
  # for i in range(30):
  #   plt.plot(traj_particles[:, i, 0].cpu(), traj_particles[:, i, 1].cpu())
  # plt.title('Transport Trajectory')
  # plt.tight_layout()


def train(cfg: DictConfig):
  # set seed for random number generators in pytorch, numpy and python.random
  if cfg.get("seed"):
    print(f"Seed: <{cfg.seed}>")
    pl.seed_everything(cfg.seed, workers=True)

  a = cfg.a
  dim = cfg.dim
  initial_model = Normal(-a * torch.ones((dim, )), 1)
  target_model = Normal(a * torch.ones((dim, )), 1)
  
  x0 = initial_model.sample([dataset_size])
  x1 = target_model.sample([dataset_size])
  x_pairs = torch.stack([x0, x1], dim=1).to(device)
  
  x0_test = initial_model.sample([test_dataset_size])
  x1_test = target_model.sample([test_dataset_size])
  x0_test = x0_test.to(device)
  x1_test = x1_test.to(device)

  torch.save({'x0': x0, 'x1': x1, 'x0_test': x0_test, 'x1_test': x1_test}, "data.pt")

  x_test_dict = {'f': x0_test, 'b': x1_test}
  
  net_split = cfg.net_name.split("_")
  if net_split[0] == "mlp":
    if net_split[1] == "small":
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
    else:
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
  else:
    raise NotImplementedError
  
  num_steps = cfg.num_steps
  sigma = cfg.sigma
  inner_iters = cfg.inner_iters
  outer_iters = cfg.outer_iters

  if cfg.model_name == "dsb":
    model = DSB(net_fwd=net_fn().to(device), 
                net_bwd=net_fn().to(device), 
                num_steps=num_steps, sig=sigma)
    train_fn = train_dsb_ipf
    print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "dsbm":
    model = DSBM(net_fwd=net_fn().to(device), 
                  net_bwd=net_fn().to(device), 
                  num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
    train_fn = train_dsbm
    print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "sbcfm":
    model = SBCFM(net=net_fn().to(device), 
                  num_steps=num_steps, sig=sigma)
    train_fn = train_flow_model
    print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "rectifiedflow":
    model = RectifiedFlow(net=net_fn().to(device), 
                          num_steps=num_steps, sig=None)
    train_fn = train_flow_model
    print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
  else:
    raise ValueError("Wrong model_name!")


  # Training loop
  # first_it = True
  model_list = []
  it = 1
  
  # assert outer_iters % len(cfg.fb_sequence) == 0
  while it <= outer_iters:
    for fb in cfg.fb_sequence:
      print(f"Iteration {it}/{outer_iters} {fb}")
      first_it = (it == 1)
      if first_it:
        prev_model = None
      else:
        prev_model = model_list[-1]["model"].eval()
      model, loss_curve = train_fn(model, x_pairs, batch_size, inner_iters, prev_model=prev_model, fb=fb, first_it=first_it)
      model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
    
      if hasattr(model, "sample_sde"):
        draw_plot(partial(model.sample_sde, zstart=x_test_dict[fb], fb=fb, first_it=first_it), z0=x_test_dict['f'], z1=x_test_dict['b'])
        plt.savefig(f"{it}-{fb}.png")
        plt.close()

        # Evaluation
        optimal_result_dict = {'mean': -a, 'var': 1, 'cov': (np.sqrt(5) - 1) / 2}
        result_list = {k: [] for k in optimal_result_dict.keys()}
        for i in range(it):
          traj = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b')
          result_list['mean'].append(traj[-1].mean(0).mean(0).item())
          result_list['var'].append(traj[-1].var(0).mean(0).item())
          result_list['cov'].append(torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
        for i, k in enumerate(result_list.keys()):
          plt.plot(result_list[k], label=f"{cfg.model_name}-{cfg.net_name}")
          plt.plot(np.arange(outer_iters), optimal_result_dict[k] * np.ones(outer_iters), label="optimal", linestyle="--")
          plt.title(k.capitalize())
          if i == 0:
            plt.legend()
          plt.savefig(f"convergence_{k}.png")
          plt.close()
        
        result_list_100 = {k: [] for k in optimal_result_dict.keys()}
        for i in range(it):
          traj_100 = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b', N=100)
          result_list_100['mean'].append(traj_100[-1].mean(0).mean(0).item())
          result_list_100['var'].append(traj_100[-1].var(0).mean(0).item())
          result_list_100['cov'].append(torch.cov(torch.cat([traj_100[0], traj_100[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
      
      if hasattr(model, "sample_ode"):
        draw_plot(partial(model.sample_ode, zstart=x_test_dict[fb], fb=fb, first_it=first_it), z0=x_test_dict['f'], z1=x_test_dict['b'])
        plt.savefig(f"{it}-{fb}-ode.png")
        plt.close()

        # Evaluation
        optimal_result_dict_ode = {'mean': -a, 'var': 1}
        result_list_ode = {k: [] for k in optimal_result_dict_ode.keys()}
        for i in range(it):
          traj_ode = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b')
          result_list_ode['mean'].append(traj_ode[-1].mean(0).mean(0).item())
          result_list_ode['var'].append(traj_ode[-1].var(0).mean(0).item())
        for i, k in enumerate(result_list_ode.keys()):
          plt.plot(result_list_ode[k], label=f"{cfg.model_name}-{cfg.net_name}-ode")
          plt.plot(np.arange(outer_iters), optimal_result_dict_ode[k] * np.ones(outer_iters), label="optimal", linestyle="--")
          plt.title(k.capitalize())
          if i == 0:
            plt.legend()
          plt.savefig(f"convergence_{k}-ode.png")
          plt.close()
        
        result_list_ode_100 = {k: [] for k in optimal_result_dict_ode.keys()}
        for i in range(it):
          traj_ode_100 = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b', N=100)
          result_list_ode_100['mean'].append(traj_ode_100[-1].mean(0).mean(0).item())
          result_list_ode_100['var'].append(traj_ode_100[-1].var(0).mean(0).item())

      # first_it = False
      it += 1

      if it > outer_iters:
        break

  torch.save([{'fb': m['fb'], 'model': m['model'].state_dict()} for m in model_list], "model_list.pt")

  if hasattr(model, "sample_sde"):
    df_result = pd.DataFrame(result_list)
    df_result_100 = pd.DataFrame(result_list_100)
    df_result.to_csv('df_result.csv')
    df_result.to_pickle('df_result.pkl')
    df_result_100.to_csv('df_result_100.csv')
    df_result_100.to_pickle('df_result_100.pkl')

    # Trajectory
    np.save("traj.npy", torch.stack(traj, dim=1).detach().cpu().numpy())
    np.save("traj_100.npy", torch.stack(traj_100, dim=1).detach().cpu().numpy())

  if hasattr(model, "sample_ode"):
    df_result_ode = pd.DataFrame(result_list_ode)
    df_result_ode_100 = pd.DataFrame(result_list_ode_100)
    df_result_ode.to_csv('df_result_ode.csv')
    df_result_ode.to_pickle('df_result_ode.pkl')
    df_result_ode_100.to_csv('df_result_ode_100.csv')
    df_result_ode_100.to_pickle('df_result_ode_100.pkl')

    # Trajectory
    np.save("traj_ode.npy", torch.stack(traj_ode, dim=1).detach().cpu().numpy())
    np.save("traj_ode_100.npy", torch.stack(traj_ode_100, dim=1).detach().cpu().numpy())

  return {}, {}


@hydra.main(config_path="conf", config_name="gaussian.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
