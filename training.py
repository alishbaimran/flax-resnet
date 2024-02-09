import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import flax
from flax.training import dynamic_scale as dynamic_scale_lib
import flax.linen as nn
from flax.training import train_state
from flax.training import common_utils
from flax.training import checkpoints
from flax.training import lr_schedule
import optax
import numpy as np
import dataclasses
import functools
from tqdm import tqdm
from typing import Any
import argparse
import wandb
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import flaxmodels as fm

def cross_entropy_loss(logits, labels):
    """
    Computes the cross entropy loss.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return -jnp.sum(common_utils.onehot(labels, num_classes=logits.shape[1]) * logits) / labels.shape[0]


def compute_metrics(logits, labels):
    """
    Computes the cross entropy loss and accuracy.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (dict): Dictionary containing the cross entropy loss and accuracy.
    """
    loss = cross_entropy_loss(logits, labels)
    top1_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    top5_accuracy = jnp.mean(jnp.sum(jnp.argsort(logits, axis=-1)[:, -5:] == jnp.expand_dims(labels, axis=-1), axis=-1))
    metrics = {'loss': loss, 'top1_accuracy': top1_accuracy, 'top5_accuracy': top5_accuracy}
    return metrics

class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.

    Attributes:
        batch_stats (Any): Collection used to store an exponential moving
                           average of the batch statistics.
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
    """
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    epoch: int


def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.

    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.

    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    return checkpoints.restore_checkpoint(path, state)


def save_checkpoint(state, step_or_metric, path):
    """
    Saves a checkpoint from the given state.

    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.

    """
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(path, state, step_or_metric, keep=3)

def sync_batch_stats(state):
    """
    Sync the batch statistics across devices.

    Args:
        state (train_state.TrainState): Training state.
    
    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def configure_dataloader(ds, prerocess, num_devices, batch_size):
    # https://www.tensorflow.org/tutorials/load_data/images
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(lambda x, y: (prerocess(x), y), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=num_devices * batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def train_step(state, batch):

    def loss_fn(params):
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                 batch['image'],
                                                 mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_model_state, logits)

    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = jax.lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])

    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                  new_state.opt_state,
                                                                  state.opt_state),
                                      params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                               new_state.params,
                                                               state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics

def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'])

def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            lambda x: x.permute(1, 2, 0)
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            lambda x: x.permute(1, 2, 0)
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    sample_train_image = next(iter(train_loader))[0][0]
    sample_val_image = next(iter(val_loader))[0][0]
    print("Shape of the first training image:", sample_train_image.shape)
    print("Shape of the first validation image:", sample_val_image.shape)

    return train_loader, val_loader

def train_and_evaluate(config):
    #wandb.login(key="647e50adb12fd19103925b4d7f4528d96e6b1c6d")
    #wandb.init(project="unet-foveal")
    num_devices = jax.device_count()

    #--------------------------------------
    # Data
    #--------------------------------------
    ds_train, ds_val = data_loader(config.data_dir, batch_size=config.batch_size * num_devices)

    print("Training dataset size:", len(ds_train.dataset))
    print("Validation dataset size:", len(ds_val.dataset))

    dataset_size = len(ds_train.dataset)

    #--------------------------------------
    # Seeding, Devices, and Precision
    #--------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None


    #--------------------------------------
    # Initialize Models
    #--------------------------------------
    rng, init_rng = jax.random.split(rng)
    
    if config.arch == 'resnet18':
        model = fm.ResNet18(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet34':
        model = fm.ResNet34(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet50':
        model = fm.ResNet50(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet101':
        model = fm.ResNet101(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet152':
        model = fm.ResNet152(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)

    variables = model.init(init_rng, jnp.ones((1, config.img_size, config.img_size, config.img_channels), dtype=dtype))
    params, batch_stats = variables['params'], variables['batch_stats']
    
    #--------------------------------------
    # Initialize Optimizer
    #--------------------------------------
    steps_per_epoch = dataset_size // config.batch_size

    learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(config.learning_rate,
                                                                        steps_per_epoch,
                                                                        config.num_epochs - config.warmup_epochs,
                                                                        config.warmup_epochs)

    tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=dynamic_scale,
                              epoch=0)
    
    step = 0
    epoch_offset = 0
    if config.resume:
        ckpt_path = checkpoints.latest_checkpoint(config.ckpt_dir)
        state = restore_checkpoint(state, ckpt_path)
        step = jax.device_get(state.step)
        epoch_offset = jax.device_get(state.epoch)
    
    state = flax.jax_utils.replicate(state)
    
    #--------------------------------------
    # Create train and eval steps
    #--------------------------------------
    p_train_step = jax.pmap(functools.partial(train_step), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')


    #--------------------------------------
    # Training 
    #--------------------------------------

    best_top1_acc = 0.0

    for epoch in range(epoch_offset, config.num_epochs):
        print("epoch running!!!")
        pbar = tqdm(total=dataset_size)

        top1_accuracy = 0.0
        top5_accuracy = 0.0

        train_loss_accumulator = 0.0
        val_loss_accumulator = 0.0

        n = 0
        for batch in ds_train:
            print("train batch!!!")
            print(images.shape)
            print(labels.shape)
            images, labels = batch
            pbar.update(num_devices * config.batch_size)
            images = images.numpy().astype(dtype)
            labels = labels.numpy().astype(dtype)

            if images.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            images = jnp.reshape(images, (num_devices, -1) + images.shape[1:])
            labels = jnp.reshape(labels, (num_devices, -1) + labels.shape[1:])

            state, metrics = p_train_step(state, {'image': images, 'label': labels})
            top1_accuracy += metrics['top1_accuracy']  # Accumulate top-1 accuracy
            top5_accuracy += metrics['top5_accuracy']
            train_loss_accumulator += metrics['loss']
            
            n += 1

        if config.wandb:
            if 'scale' in metrics:
                wandb.log({'training/scale': jnp.mean(metrics['scale']).item()}, step=step, commit=False)
            wandb.log({'training/top1_accuracy': jnp.mean(metrics['top1_accuracy']).item()}, step=step)
            wandb.log({'training/top5_accuracy': jnp.mean(metrics['top5_accuracy']).item()}, step=step)
            wandb.log({'training/loss':  jnp.mean(metrics['loss'].item())}, step=step)
        step += 1

        pbar.close()
        top5_accuracy /= n
        top1_accuracy /= n
        train_loss_accumulator /= n

        print(f'Epoch: {epoch}')
        print('Training top-1 accuracy:', jnp.mean(top1_accuracy))
        print('Training top-5 accuracy:', jnp.mean(top5_accuracy))
        print(f'Average Training Loss:', jnp.mean(train_loss_accumulator))

        #--------------------------------------
        # Validation 
        #--------------------------------------
        # Sync batch stats
        state = sync_batch_stats(state)

        top1_accuracy = 0.0
        top5_accuracy = 0.0
        n = 0

        for batch in ds_val:
            print("val batch!!!")
            images, labels = batch
            images = images.numpy().astype(dtype)
            labels = labels.numpy().astype(dtype)
            if images.shape[0] % num_devices != 0:
                continue
            
            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            images = jnp.reshape(images, (num_devices, -1) + images.shape[1:])
            labels = jnp.reshape(labels, (num_devices, -1) + labels.shape[1:])
            # metrics gets updated here 
            metrics = p_eval_step(state, {'image': images, 'label': labels})
            val_loss_accumulator += metrics['loss']
            
            top1_accuracy += metrics['top1_accuracy']  
            top5_accuracy += metrics['top5_accuracy']
            n += 1

        top1_accuracy /= n 
        top5_accuracy /= n 
        val_loss_accumulator /= n

        print(f'Epoch: {epoch}')
        print('Top 1 validation accuracy:', jnp.mean(top1_accuracy))
        print('Top 5 validation accuracy:', jnp.mean(top5_accuracy))
        print(f'Average Validation Loss:', jnp.mean(val_loss_accumulator))

        top1_accuracy = jnp.mean(top1_accuracy).item()
        top5_accuracy = jnp.mean(top5_accuracy).item()

        #if top1_accuracy > best_top1_acc:
            #best_top1_acc = top1_accuracy
            #state_top1 = dataclasses.replace(state, **{'step': flax.jax_utils.replicate(step), 'epoch': flax.jax_utils.replicate(epoch)})
            #save_checkpoint(state_top1, jnp.mean(top1_accuracy).item(), config.ckpt_dir)
        # Calculate average training and validation loss for the epoch

        if config.wandb:
            wandb.log({'validation/loss': jnp.mean(metrics['loss']).item()}, step=step)
            wandb.log({'validation/top1_accuracy': jnp.mean(top1_accuracy).item()}, step=step)
            wandb.log({'validation/top5_accuracy': jnp.mean(top5_accuracy).item()}, step=step)

