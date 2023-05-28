import torch
from torch import nn

class EMAHelper(object):
    def __init__(self, mu=0.999, device="cpu"):
        self.mu = mu
        self.shadow = {}
        self.device = device
        
    def register(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.distributed.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.distributed.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data
                
    def ema(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.distributed.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
    
    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.distributed.DistributedDataParallel):
            inner_module = module.module
            locs = inner_module.locals
            module_copy = type(inner_module)(*locs).to(self.device)
            module_copy.load_state_dict(inner_module.state_dict())
            if isinstance(module, nn.DataParallel):
                module_copy = nn.DataParallel(module_copy)
        else:
            locs = module.locals
            module_copy = type(module)(*locs).to(self.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict
