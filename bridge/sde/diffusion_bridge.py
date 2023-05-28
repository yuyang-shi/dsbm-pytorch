import torch
import torch.nn as nn
import numpy as np
from .optimal_transport import OTPlanSampler

class DBDSB_VE:
  def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=False, ot_sampler=None, eps=1e-4, **kwargs):
    self.device = timesteps.device

    self.sig = sig              # total sigma from time 0 and T=1
    self.num_steps = num_steps  # num diffusion steps
    self.timesteps = timesteps  # schedule of timesteps
    assert len(self.timesteps) == self.num_steps
    assert torch.allclose(self.timesteps.sum(), torch.tensor(self.T))  # sum of timesteps is T=1
    assert (self.timesteps > 0).all()
    self.gammas = self.timesteps * self.sig**2  # schedule of variance steps
    
    self.d_x = shape_x  # dimension of object to diffuse
    self.d_y = shape_y  # dimension of conditioning

    self.first_coupling = first_coupling
    self.eps = eps

    self.ot_sampler = None
    if ot_sampler is not None:
      self.ot_sampler = OTPlanSampler(ot_sampler, reg=2*self.sig**2)
    self.mean_match = mean_match

  @property
  def T(self):
    return 1.

  @property
  def alpha(self):
    return 0.

  @torch.no_grad()
  def marginal_prob(self, x, t, fb):
    if fb == "f":
      std = self.sig * torch.sqrt(t)
    else:
      std = self.sig * torch.sqrt(self.T - t)
    mean = x
    return mean, std

  @torch.no_grad()
  def record_langevin_seq(self, net, samples_x, init_samples_y, fb, sample=False, num_steps=None, **kwargs):
    if fb == 'b':
      gammas = torch.flip(self.gammas, (0,))
      timesteps = torch.flip(self.timesteps, (0,))
      t = torch.ones((samples_x.shape[0], 1), device=self.device)
      sign = -1.
    elif fb == 'f':
      gammas = self.gammas
      timesteps = self.timesteps
      t = torch.zeros((samples_x.shape[0], 1), device=self.device)
      sign = 1.

    x = samples_x
    N = x.shape[0]

    if num_steps is None:
      num_steps = self.num_steps
    else:
      timesteps = np.interp(np.arange(1, num_steps+1)/num_steps, np.arange(self.num_steps+1)/self.num_steps, [0, *np.cumsum(timesteps.cpu())])
      timesteps = torch.from_numpy(np.diff(timesteps, prepend=[0])).to(self.device)
      gammas = timesteps * self.sig**2

    x_tot = torch.Tensor(N, num_steps, *self.d_x).to(x.device)
    y_tot = None
    steps_expanded = torch.Tensor(N, num_steps, 1).to(x.device)
    
    drift_fn = self.get_drift_fn_pred(fb)
    
    for k in range(num_steps):
      gamma = gammas[k]
      timestep = timesteps[k]

      pred = net(x, init_samples_y, t)  # Raw prediction of the network

      if sample and (k==num_steps-1) and self.mean_match:
        x = pred
      else:
        drift = drift_fn(t, x, pred)
        x = x + drift * timestep
        if not (sample and (k==num_steps-1)):
          x = x + torch.sqrt(gamma) * torch.randn_like(x)

      x_tot[:, k, :] = x
      # y_tot[:, k, :] = y
      steps_expanded[:, k, :] = t
      t = t + sign * timestep
    
    if fb == 'b':
      assert torch.allclose(t, torch.zeros(1, device=self.device), atol=1e-4, rtol=1e-4), f"{t} != 0"
    else:
      assert torch.allclose(t, torch.ones(1, device=self.device) * self.T, atol=1e-4, rtol=1e-4), f"{t} != 1"

    return x_tot, y_tot, None, steps_expanded

  @torch.no_grad()
  def generate_new_dataset(self, x0, y0, x1, sample_fn, sample_direction, sample=False, num_steps=None):
    if sample_direction == 'f':
      zstart = x0
    else:
      zstart = x1
    zend = self.record_langevin_seq(sample_fn, zstart, y0, sample_direction, sample=sample, num_steps=num_steps)[0][:, -1]
    if sample_direction == 'f':
      z0, z1 = zstart, zend
    else:
      z0, z1 = zend, zstart
    return z0, y0, z1

  @torch.no_grad()
  def probability_flow_ode(self, net_f=None, net_b=None, y=None):
    get_drift_fn_net = self.get_drift_fn_net

    class ODEfunc(nn.Module):
      def __init__(self, net_f=None, net_b=None):
        super().__init__()
        self.net_f = net_f
        self.net_b = net_b
        self.nfe = 0
        if self.net_f is not None:
          self.drift_fn_f = get_drift_fn_net(self.net_f, 'f', y=y)
        self.drift_fn_b = get_drift_fn_net(self.net_b, 'b', y=y)

      def forward(self, t, x):
        self.nfe += 1
        t = torch.ones((x.shape[0], 1), device=x.device) * t.item()
        if self.net_f is None:
          return - self.drift_fn_b(t, x)
        return (self.drift_fn_f(t, x) - self.drift_fn_b(t, x)) / 2

    return ODEfunc(net_f=net_f, net_b=net_b)

  @torch.no_grad()
  def get_train_tuple(self, x0, x1, fb='', first_it=False):
    if first_it and fb == 'b':
      z0 = x0
      if self.first_coupling == "ref":
        # First coupling is x_0, x_0 perturbed
        z1 = z0 + torch.randn_like(z0) * self.sig
      elif self.first_coupling == "ind":
        z1 = x1
      else:
        raise NotImplementedError
    elif first_it and fb == 'f':
      assert self.first_coupling == "ind"
      z0, z1 = x0, x1
    else:
      z0, z1 = x0, x1
    
    if self.ot_sampler is not None:
      assert z0.shape == z1.shape
      original_shape = z0.shape
      z0, z1 = self.ot_sampler.sample_plan(z0.flatten(start_dim=1), z1.flatten(start_dim=1))
      z0, z1 = z0.view(original_shape), z1.view(original_shape)

    t = torch.rand(z1.shape[0], device=self.device) * (1-2*self.eps) + self.eps
    t = t[:, None, None, None]
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    if self.mean_match:
      if fb == 'f':
        target = z1
      else:
        target = z0
    else:
      if fb == 'f':
        # (z1 - z_t) / (1-t)
        # target = z1 - z0 
        # target = target - self.sig * torch.sqrt(t/(1.-t)) * z
        # target = self.A_f(t) * z_t + self.M_f(t) * z1
        drift_f = self.drift_f(t, z_t, z0, z1)
        target = drift_f + self.alpha * z_t
      else:
        # (z0 - z_t) / t
        # target = - (z1 - z0)
        # target = target - self.sig * torch.sqrt((1.-t)/t) * z
        drift_b = self.drift_b(t, z_t, z0, z1)
        target = drift_b - self.alpha * z_t
    return z_t, t, target

  def A_f(self, t):
    return -1./(self.T-t)

  def M_f(self, t):
    return 1./(self.T-t)

  def A_b(self, t):
    return -1./t

  def M_b(self, t):
    return 1./t

  def drift_f(self, t, x, init, final):
    t = t.view(t.shape[0], 1, 1, 1)
    return self.A_f(t) * x + self.M_f(t) * final

  def drift_b(self, t, x, init, final):
    t = t.view(t.shape[0], 1, 1, 1)
    return self.A_b(t) * x + self.M_b(t) * init

  def get_drift_fn_net(self, net, fb, y=None):
    drift_fn_pred = self.get_drift_fn_pred(fb)
    def drift_fn(t, x):
      pred = net(x, y, t)  # Raw prediction of the network
      return drift_fn_pred(t, x, pred)
    return drift_fn

  def get_drift_fn_pred(self, fb):
    def drift_fn(t, x, pred):
      if self.mean_match:
        if fb == 'f':
          drift = self.drift_f(t, x, None, pred)
        else:
          drift = self.drift_b(t, x, pred, None)
      else:
        if fb == 'f':
          drift = pred - self.alpha * x
        else:
          drift = pred + self.alpha * x
      return drift
    return drift_fn


class DBDSB_VP(DBDSB_VE):
  def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=False, ot_sampler=None, eps=1e-4, **kwargs):
    assert ot_sampler is None
    super().__init__(sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=mean_match, ot_sampler=ot_sampler, eps=eps, **kwargs)

  @property
  def alpha(self):
    return 0.5

  @torch.no_grad()
  def marginal_prob(self, x, t, fb):
    if fb == "f":
      mean = torch.exp(-0.5 * t) * x
      std = self.sig * torch.sqrt(1 - torch.exp(-t))
    else:
      raise NotImplementedError
    return mean, std

  def A_f(self, t: float) -> float:
    return -self.alpha / torch.tanh(self.alpha * (self.T - t))

  def M_f(self, t: float) -> float:
    return self.alpha / torch.sinh(self.alpha * (self.T - t))

  def A_b(self, t: float) -> float:
    return -self.alpha / torch.tanh(self.alpha * t)

  def M_b(self, t: float) -> float:
    return self.alpha / torch.sinh(self.alpha * t)
