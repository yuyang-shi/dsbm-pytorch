from pytorch_lightning.loggers import CSVLogger as _CSVLogger, WandbLogger as _WandbLogger
import wandb

class Logger:
    def log_metrics(self, metric_dict, step=None):
        pass

    def log_hyperparams(self, params):
        pass

    def log_image(self, key, images, **kwargs):
        pass


class CSVLogger(_CSVLogger):
    def log_image(self, key, images, **kwargs):
        pass


class WandbLogger(_WandbLogger):
    LOGGER_JOIN_CHAR = '/'

    def log_metrics(self, metrics, step=None, fb=None):
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        super().log_metrics(metrics, step=step)

    def log_image(self, key, images, **kwargs):
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        step = kwargs.pop("step", None)
        fb = kwargs.pop("fb", None)
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        if n == 1:
            metrics = {key: wandb.Image(images[0], **kwarg_list[0])}
        else:
            metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step=step, fb=fb)