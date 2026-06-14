import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from train.distributed import is_master
from train.precision import get_autocast
from model import get_input_dtype


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if len(batch) == 3:
            texts, images, extra_texts = batch
        else:
            texts, images = batch
            extra_texts = None
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        
        if extra_texts is not None:
            extra_texts = extra_texts.to(device=device, dtype=input_dtype, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            model_out = model(images, texts, extra_texts)
            logit_scale = model_out["logit_scale"]
            losses = loss(**model_out, output_dict=True)
            total_loss = losses['contrastive_loss']

        backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"Logit Bias: {model_out['logit_bias'].item():.3f} "
                + loss_log

            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "scale": logit_scale_scalar,
                "logit_bias": model_out['logit_bias'].item(),
                "lr": optimizer.param_groups[0]["lr"],
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
                
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def evaluate(model, data, loss, epoch, args):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                texts, images = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                batch_size = len(images)
                with autocast():
                    model_out = model(images, texts)
                    all_image_features.append(model_out["image_features"])
                    all_text_features.append(model_out["text_features"])
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    total_loss = loss(**model_out, output_dict=True)['contrastive_loss']

                cumulative_loss += total_loss * batch_size * batch_size
                num_samples += batch_size * batch_size

            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Loss: {cumulative_loss / num_samples:.6f}\t")
            with autocast():
                val_metrics = get_siglip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=model_out["logit_scale"],
                    logit_bias=model_out["logit_bias"]
                )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        # if 'train' in data:
        #     dataloader = data['train'].dataloader
        #     num_batches_per_epoch = dataloader.num_batches // args.accum_freq
        #     step = num_batches_per_epoch * epoch
        # else:
        #     step = None
        log_data['epoch'] = epoch
        wandb.log(log_data)

    return metrics

def get_siglip_metrics(image_features, text_features, logit_scale, logit_bias):
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    metrics = {}
    logits_per_image = (image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}

    # retrieval metrics
    ground_truth = torch.arange(len(text_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    # binary classification metrics, the percentage of diagonal logits >0 , and rest <0
    n = logits_per_image.shape[0]
    diagonal_positive = torch.sum(torch.diag(logits_per_image) > 0).cpu()
    positive_accuracy = diagonal_positive.float() / n
    off_diagonal = logits_per_image[~torch.eye(n, dtype=bool)].cpu()
    negative_correct = torch.sum(off_diagonal < 0)
    total_negative = n * (n - 1)
    negative_accuracy = negative_correct / total_negative
    metrics["positive_accuracy"] = positive_accuracy.item()
    metrics["negative_accuracy"] = negative_accuracy.item()

    return metrics
