# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import numpy as np
import torch

# The torch.distributed package provides PyTorch support
# and communication primitives for multiprocess parallelism
# across several computation nodes running on one or more machines.
import torch.distributed as dist

import torch.nn as nn

# converts the dataclass obj to a dict
from dataclasses import asdict

# This container provides data parallelism by synchronizing gradients across each model replica (multiple GPU support).
from torch.nn.parallel import DistributedDataParallel

# DistributedSampler: Sampler that restricts data loading to a subset of the dataset.
#                     It is especially useful in conjunction with nn.parallel.DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# SummaryWriter: The SummaryWriter class provides a high-level API to create an event file in a given directory
#                and add summaries and events to it. The class updates the file contents asynchronously.
#                This allows a training program to call methods to add data to the file directly from the training loop,
#                without slowing down training.
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import Dict

from .config_torchwavenet import Config
from .torchdataset import RFMixtureDatasetBase, get_train_val_dataset
from .torchwavenet import Wave

SEED = 0


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def view_as_complex(x):
    """
    Combines inserted tensor x from two real dimensions to one complex dimension.
    """
    x = x[:, 0, ...] + 1j * x[:, 1, ...]
    return x


class WaveLearner:
    def __init__(self, cfg: Config, model: nn.Module, rank: int):
        self.cfg = cfg

        # Store some import variables <- priori comment
        self.model_dir = cfg.model_dir
        self.distributed = cfg.distributed.distributed
        self.world_size = cfg.distributed.world_size
        self.rank = rank
        self.log_every = cfg.trainer.log_every
        self.validate_every = cfg.trainer.validate_every
        self.save_every = cfg.trainer.save_every
        self.max_steps = cfg.trainer.max_steps
        self.epochs = cfg.trainer.epochs  # added epochs as originally steps were the training measure
        self.build_dataloaders()

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.trainer.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min",
        )
        self.autocast = torch.amp.autocast(device_type='cuda', enabled=cfg.trainer.fp16)
        self.scaler = torch.amp.GradScaler(enabled=cfg.trainer.fp16)
        self.step = 0

        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter(self.model_dir)

    @property
    def is_master(self):
        return self.rank == 0

    def build_dataloaders(self):
        """
        Initializes the PyTorch DataLoaders from the Datasets created from "torchdataset.py",
        where the data is taken from the "root_dir" from the Config class object.
        """
        self.dataset = RFMixtureDatasetBase(
            root_dir=self.cfg.data.root_dir,
        )
        self.train_dataset, self.val_dataset = get_train_val_dataset(
            self.dataset, self.cfg.data.train_fraction)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=not self.distributed,
            num_workers=self.cfg.data.num_workers if self.distributed else 0,
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank) if self.distributed else None,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size * 4,
            shuffle=not self.distributed,
            num_workers=self.cfg.data.num_workers if self.distributed else 0,
            sampler=DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank) if self.distributed else None,
            pin_memory=True,
        )

    def state_dict(self):
        """
        PyTorch RFC WaveNet custom state dictionary to save.
        """
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor)
            else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor)
            else v for k, v in self.optimizer.state_dict().items()},
            'cfg': asdict(self.cfg),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Load PyTorch RFC WaveNet custom state dictionary.
        """
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        """
        Gives the latest saved model state_dict a link (shortcut)
        in order to load it generally without the saved step number.
        """
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        """
        Restore model state dict from last saved checkpoint.
        """
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, RuntimeError):  #fix
            return False

    def train(self):
        """
        PyTorch standard model training function.
        """
        device = next(self.model.parameters()).device

        #while True:
        max_step = 'end' if self.max_steps <= 0 else self.max_steps
        for epoch in range(self.epochs):
            for i, features in enumerate(
                    tqdm(self.train_dataloader,
                         desc=f"Training (step: {self.step} / {max_step}), epoch: ({epoch + 1}/{self.epochs})")):
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at step {self.step}.')

                if self.is_master:
                    if self.step % self.log_every == 0:
                        self.writer.add_scalar('train/loss', loss, self.step)
                        self.writer.add_scalar(
                            'train/grad_norm', self.grad_norm, self.step)
                    if self.step % self.save_every == 0:
                        self.save_to_checkpoint()

                if self.step % self.validate_every == 0:
                    val_loss = self.validate()
                    # Update the learning rate if it plateus
                    self.lr_scheduler.step(val_loss)

                if self.distributed:
                    dist.barrier()

                self.step += 1

                if self.max_steps > 0:
                    if self.step == self.max_steps:
                        if self.is_master and self.distributed:
                            self.save_to_checkpoint()
                            print("Ending training...")
                        if self.distributed:
                            dist.barrier()  #fixed for 1 gpu
                        exit(0)
        else:
            if self.is_master and self.distributed:
                self.save_to_checkpoint()
                print("Ending training...")
            if self.distributed:
                dist.barrier()  #fixed for 1 gpu
            exit(0)

    def train_step(self, features: Dict[str, torch.Tensor]):
        for param in self.model.parameters():
            param.grad = None

        sample_mix = features["sample_mix"]
        sample_soi = features["sample_soi"]

        N, _, _ = sample_mix.shape

        with self.autocast:
            predicted = self.model(sample_mix)
            loss = self.loss_fn(predicted, sample_soi)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.trainer.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        loss = 0
        for features in tqdm(
                self.val_dataloader,
                desc=f"Running validation after step {self.step}"
        ):
            features = _nested_map(features, lambda x: x.to(
                device) if isinstance(x, torch.Tensor) else x)
            sample_mix = features["sample_mix"]
            sample_soi = features["sample_soi"]
            N, _, _ = sample_mix.shape

            with self.autocast:
                predicted = self.model(sample_mix)
                loss += torch.sum(
                    (predicted - sample_soi) ** 2, (0, 1, 2)
                ) / len(self.val_dataset) / np.prod(sample_soi.shape[1:])
        if self.distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        self.writer.add_scalar('val/loss', loss, self.step)
        self.model.train()

        return loss


def _train_impl(rank: int, model: nn.Module, cfg: Config):
    torch.backends.cudnn.benchmark = True

    learner = WaveLearner(cfg, model, rank)
    learner.restore_from_checkpoint()
    learner.train()


def train(cfg: Config):
    """Training on a single GPU."""
    torch.manual_seed(SEED)  # i added
    model = Wave(cfg.model).cuda()
    _train_impl(0, model, cfg)


def init_distributed(rank: int, world_size: int, port: str):
    """Initialize distributed training on multiple GPUs."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size)


def train_distributed(rank: int, world_size: int, port, cfg: Config):
    """Training on multiple GPUs."""
    torch.manual_seed(SEED)  # i added
    init_distributed(rank, world_size, port)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    model = Wave(cfg.model).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    _train_impl(rank, model, cfg)
