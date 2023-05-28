import torch

def grad_gauss(x, m, var):
    xout = (m - x) / var
    return xout

# def ornstein_ulhenbeck(x, gradx, gamma):
#     xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
#     return xout

class Langevin:

    def __init__(self, num_steps, shape_x, shape_y, gammas, time_sampler,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]), 
                 mean_match=True, out_scale=1, var_final_gamma_scale=False):
        self.device = gammas.device

        self.mean_match = mean_match
        self.mean_final = mean_final.to(self.device) if mean_final is not None else None
        self.var_final = var_final.to(self.device) if var_final is not None else None
        
        self.num_steps = num_steps # num diffusion steps
        self.d_x = shape_x # dimension of object to diffuse
        self.d_y = shape_y # dimension of conditioning
        self.gammas = gammas # schedule

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device)
        # self.time_sampler = time_sampler
        self.out_scale = out_scale
        self.var_final_gamma_scale = var_final_gamma_scale
            

    def record_init_langevin(self, init_samples_x, init_samples_y, mean_final=None, var_final=None):
        if mean_final is None:
            mean_final = self.mean_final
        if var_final is None:
            var_final = self.var_final
        
        x = init_samples_x
        # y = init_samples_y
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))


        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        # y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        y_tot = None
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        num_iter = self.num_steps
        steps_expanded = steps
        
        for k in range(num_iter):
            gamma = self.gammas[k]

            if self.var_final_gamma_scale:
                var_gamma_ratio = 1 / gamma
                scaled_gamma = gamma * var_final
            else:
                var_gamma_ratio = var_final / gamma
                scaled_gamma = gamma
            
            gradx = grad_gauss(x, mean_final, var_gamma_ratio)
            t_old = x + gradx / 2
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(scaled_gamma)*z
            gradx = grad_gauss(x, mean_final, var_gamma_ratio)
            t_new = x + gradx / 2
            x_tot[:, k, :] = x
            # y_tot[:, k, :] = y
            if self.mean_match:
                out[:, k, :] = (t_old - t_new) #/ (2 * gamma)
            else:
                out_scale = eval(self.out_scale).to(self.device) if isinstance(self.out_scale, str) else self.out_scale
                out[:, k, :] = (t_old - t_new) / out_scale
            
        return x_tot, y_tot, out, steps_expanded

    def record_langevin_seq(self, net, samples_x, init_samples_y, fb, sample=False, var_final=None):
        if var_final is None:
            var_final = self.var_final
        if fb == 'b':
            gammas = torch.flip(self.gammas, (0,))
        elif fb == 'f':
            gammas = self.gammas

        x = samples_x
        # y = init_samples_y
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))

        
        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        # y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        y_tot = None
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        steps_expanded = steps
        num_iter = self.num_steps
        
        if self.mean_match:
            for k in range(num_iter):
                gamma = gammas[k]

                scaled_gamma = gamma
                if self.var_final_gamma_scale:
                    scaled_gamma = scaled_gamma * var_final

                t_old = net(x, None, steps[:, k, :])
                
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(scaled_gamma) * z
                    
                t_new = net(x, None, steps[:, k, :])
                x_tot[:, k, :] = x
                # y_tot[:, k, :] = y
                out[:, k, :] = (t_old - t_new) 
        else:
            for k in range(num_iter):
                gamma = gammas[k]

                scaled_gamma = gamma
                if self.var_final_gamma_scale:
                    scaled_gamma = scaled_gamma * var_final
                out_scale = eval(self.out_scale).to(self.device) if isinstance(self.out_scale, str) else self.out_scale

                t_old = x + out_scale * net(x, None, steps[:, k, :])
                
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(scaled_gamma) * z
                t_new = x + out_scale * net(x, None, steps[:, k, :])
                
                x_tot[:, k, :] = x
                # y_tot[:, k, :] = y
                out[:, k, :] = (t_old - t_new) / out_scale
            

        return x_tot, y_tot, out, steps_expanded
