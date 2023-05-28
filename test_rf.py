import torch
import hydra
import os

from bridge.trainer_rf import IPF_RF
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator


def test(args):
    accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    ipf = IPF_RF(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                        final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    test_metrics = ipf.plot_and_test_step(ipf.step, ipf.checkpoint_it, "b", sampler='sde')
    accelerator.print("SDE: ", test_metrics)

    if args.test_ode_sampler:
        test_metrics = ipf.plot_and_test_step(ipf.step, ipf.checkpoint_it, "b", sampler='ode')
        accelerator.print("ODE: ", test_metrics)

