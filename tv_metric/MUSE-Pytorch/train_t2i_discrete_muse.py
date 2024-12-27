import builtins
import datetime
import os
import time
from torch.utils._pytree import tree_map

import accelerate
import einops
import ml_collections
import torch
from datasets import get_dataset
from loguru import logger
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import taming.models.vqgan
import wandb
from libs.muse import MUSE
from tools.fid_score import calculate_fid_given_paths
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image
import utils

logging = logger


def LSimple(x0, nnet, schedule, **kwargs):
    labels, masked_ids = schedule.sample(x0)
    logits = nnet(masked_ids, **kwargs)
    # b (h w) c, b (h w)
    loss = schedule.loss(logits, labels)
    return loss


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.ConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'imagenet', config=config.to_dict(),
                   job_type='train', mode='online')
        logging.info(config)
    else:
        logging.remove()
        logger.add(sys.stderr, level='ERROR')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      # num_workers=8, pin_memory=True, persistent_workers=True
                                      )

    autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
    autoencoder.to(device)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            yield train_dataset.sample_label(n_samples=config.sample.mini_batch_size, device=device)

    context_generator = get_context_generator()

    muse = MUSE(codebook_size=autoencoder.n_embed, device=device, **config.muse)

    def cfg_nnet(x, context, scale=None):
        _cond = nnet_ema(x, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L ... -> B L ...', B=x.size(0))
        _uncond = nnet_ema(x, context=_empty_context)
        res = _cond + scale * (_cond - _uncond)
        return res

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        model_start_time = time.time()
        _z = _batch[0]
        loss = LSimple(_z, nnet, muse, context=_batch[1])  # currently only support the extracted feature version
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        metric_logger.update(model_time=time.time() - model_start_time)
        metric_logger.update(loss_scaler=accelerator.scaler.get_scale())
        metric_logger.update(grad_norm=utils.get_grad_norm_(nnet.parameters()))

        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                    **{k: v.value for k, v in metric_logger.meters.items()})

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _context = next(context_generator)
            kwargs = dict(context=_context)
            return muse.generate(config, _n_samples, cfg_nnet, decode, **kwargs)

        path = f'{config.workdir}/eval_samples/{train_state.step}_{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
        logging.info(f'Path for FID images: {path}')
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                         dataset.unpreprocess)

        _fid = 0
        if accelerator.is_main_process:
            _fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
            with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
            wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
        _fid = torch.tensor(_fid, device=device)
        _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    metric_logger = utils.MetricLogger()
    if eval_ckpt_path := os.getenv('EVAL_CKPT', ''):
        nnet.eval()
        train_state.resume(eval_ckpt_path)
        logging.info(f'Eval {train_state.step}...')
        fid = eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)
        return
    while train_state.step < config.train.n_steps:
        nnet.train()
        data_time_start = time.time()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metric_logger.update(data_time=time.time() - data_time_start)
        metrics = train_step(batch)

        nnet.eval()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logger.info(f'step: {train_state.step} {metric_logger}')
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            samples = muse.generate(config, 2 * 5, cfg_nnet, decode, context=contexts)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}_{accelerator.process_index}.png'))
            if accelerator.is_main_process:
                wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.fid_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Eval {train_state.step}...')
            fid = eval_step(n_samples=config.eval.n_samples,
                            sample_steps=config.eval.sample_steps)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)


from absl import flags
from absl import app
from ml_collections import config_flags
import sys

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    config.workdir = os.getenv('OUTPUT_DIR')
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


# utils file 
import pickle

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from torch import distributed as dist
from loguru import logger
logging = logger


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit_t2i_vq':
        from libs.uvit_t2i_vq import UViT
        return UViT(**kwargs)
    elif name == 'uvit_vq':
        from libs.uvit_vq import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), map_location='cpu')
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if ckpt_root.endswith('.ckpt'):
            ckpt_path = ckpt_root
        else:
            if step is None:
                ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
                if not ckpts:
                    return
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
            ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, dist=True):
    if path:
        os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes if dist else mini_batch_size

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        if dist:
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

from collections import defaultdict, deque
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    from torch._six import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (8, 16, 16)

    config.autoencoder = d(
        config_file='vq-f16-jax.yaml',
    )

    config.train = d(
        n_steps=99999999,
        batch_size=2048,
        log_interval=10,
        eval_interval=5000,
        save_interval=5000,
        fid_interval=50000,
    )

    config.eval = d(
        n_samples=10000,
        sample_steps=12,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0004,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_vq',
        img_size=16,
        codebook_size=1024,
        in_chans=256,
        patch_size=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        num_classes=1001,
        use_checkpoint=False,
        skip=True,
    )

    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )

    config.dataset = d(
        name='imagenet256_features',
        path='assets/datasets/imagenet256_vq_features/vq-f16-jax',
        cfg=True,
        p_uncond=0.15,
    )

    config.sample = d(
        sample_steps=12,
        n_samples=50000,
        mini_batch_size=50,
        cfg=True,
        linear_inc_scale=True,
        scale=3.,
        path=''
    )
    return config




config = get_config()

device = 'cuda'

text = 'the parthenon'
nnnet_ckpt_path = '/data5/home/rajivporana/tv_metric/MUSE-Pytorch/pre_trained_models/nnet.pth'
vqgan_ckpt_path = '/data5/home/rajivporana/tv_metric/MUSE-Pytorch/pre_trained_models/vqgan_jax_strongaug.ckpt'

nnet_model =utils.get_nnet('uvit_t2i_vq', config.nnet)
nnet_model.load_state_dict(torch.load(nnnet_ckpt_path))

# define nnet model and load the checkpoint
vqgan_model = taming.models.vqgan.VQModel.load_from_checkpoint(vqgan_ckpt_path)
nnet_model.load_state_dict(torch.load(nnnet_ckpt_path))

nnet_model = nnet_model.to(device)
vqgan_model = vqgan_model.to(device)

nnet_model.eval()
vqgan_model.eval()



def cfg_nnet(x, context, scale=None, nnet_model=nnet_model):
    _cond = nnet_model(x, context=context)

    # write empty context vector using torch we dont have dataset object context of shape L ...
    _empty_context = torch.zeros(context.shape[0], context.shape[1], device=device)
    _empty_context = einops.repeat(_empty_context, 'L ... -> B L ...', B=x.size(0))
    _uncond = nnet_model(x, context=_empty_context)
    res = _cond + scale * (_cond - _uncond)
    return res

muse_model = MUSE(codebook_size=vqgan_model.quantize.n_embed, device=device, ignore_ind=-1, smoothing=0.1, gen_temp=4.5)
muse_model = muse_model.to(device)


vqgan_model.eval()

vqgan_model = vqgan_model.to(device)

autoencoder = taming.models.vqgan.get_model(config.autoencoder)
autoencoder.to(device)

def encode(_batch):
    return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

def decode(_batch):
    return autoencoder.decode_code(_batch)






with torch.no_grad():
    context = muse_model.encode_text(text)
    img = muse_model.generate(config, 10, nnet_model, decode, context=context)
    img.save('output.png')

# with vqgan get the latent code from the image
img = Image.open('output.png')
img = TF.to_tensor(img).unsqueeze(0).to(device)
z, *_ = vqgan_model.encode_image(img)
z = vqgan_model.quantize.get_codebook_indices(z)
# replace 20% of z with random values
z = utils.replace_indices(z, 0.2)
# decode the image
img = vqgan_model.decode(z)
img = TF.to_pil_image(img[0].cpu())
img.save('corrupt_output.png')





    





