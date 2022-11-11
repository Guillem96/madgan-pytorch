import random
from typing import Callable, Dict, Iterator

import numpy as np
import torch
import torch.nn as nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(generator: nn.Module,
                    discriminator: nn.Module,
                    loss_fn: LossFn,
                    real_dataloader: Iterator[torch.Tensor],
                    latent_dataloader: Iterator[torch.Tensor],
                    discriminator_optimizer: torch.optim.Optimizer,
                    generator_optimizer: torch.optim.Optimizer,
                    normal_label: int = 0,
                    anomaly_label: int = 1,
                    epoch: int = 0,
                    log_every: int = 30) -> None:
    """Trains a GAN for a single epoch.

    Args:
        generator (nn.Module): Torch module implementing the GAN generator.
        discriminator (nn.Module): Torch module implementing the GAN 
            discriminator.
        loss_fn (LossFn): Loss function, should return a reduced value.
        real_dataloader (Iterator[torch.Tensor]): Iterator to go over real data 
            samples.
        latent_dataloader (Iterator[torch.Tensor]): Iterator to go through 
            generated samples from the latent space.
        discriminator_optimizer (torch.optim.Optimizer): Optimizer for the 
            discrimninator.
        generator_optimizer (torch.optim.Optimizer): Oprimizer for the generator 
            module.
        normal_label (int): Label for samples with normal behaviour 
            (real or non-anomaly). Defaults to 0.
        anomaly_label (int): Label that identifies generate samples 
            (anomalies when running inference). Defaults to 1.
        epoch (int, optional): Current epoch (just for logging purposes). 
            Defaults to 0.
        log_every (int, optional): Log the training progess every n steps. 
            Defaults to 30.
    """
    generator.train()
    discriminator.train()

    for i, (real, z) in enumerate(zip(real_dataloader, latent_dataloader)):
        bs = real.size(0)
        real_labels = torch.full((bs, ), normal_label).float().to(real.device)
        fake_labels = torch.full((bs, ), anomaly_label).float().to(real.device)
        all_labels = torch.cat([real_labels, fake_labels])

        # Generate fake samples with the generator
        fake = generator(z)

        # Update discriminator
        discriminator_optimizer.zero_grad()
        discriminator.train()
        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        d_logits = torch.cat([real_logits, fake_logits])

        # Discriminator tries to identify the true nature of each sample
        # (anomaly or normal)
        d_real_loss = loss_fn(real_logits.view(-1), real_labels)
        d_fake_loss = loss_fn(fake_logits.view(-1), fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        discriminator_optimizer.step()

        # Update generator
        generator.zero_grad()
        discriminator.eval()

        g_logits = discriminator(fake)
        # Generator will improve so it can cheat the discriminator
        cheat_loss = loss_fn(g_logits, real_labels)
        cheat_loss.backward()
        generator_optimizer.step()

        if (i + 1) % log_every == 0:
            discriminator_acc = ((d_logits.detach() >
                                  .5) == all_labels).float()
            discriminator_acc = discriminator_acc.sum().div(bs)

            generator_acc = (g_logits.detach() > .5 == real_labels).float()
            generator_acc = generator_acc.sum().div(bs)

            log = {
                "generator_loss": cheat_loss.item(),
                "discriminator_loss": d_loss.item(),
                "discriminator_acc": discriminator_acc.item(),
                "generator_acc": generator_acc.item(),
            }

            header = f"Epoch [{epoch}] Step [{i}]"
            log_metrics = " ".join(
                f"{metric_name}:{metric_value:.4f}"
                for metric_name, metric_value in log.items())
            print(header, log_metrics)

    discriminator.zero_grad()
    generator.zero_grad()


@torch.no_grad()
def evaluate(generator: nn.Module,
             discriminator: nn.Module,
             loss_fn: LossFn,
             real_dataloader: Iterator[torch.Tensor],
             latent_dataloader: Iterator[torch.Tensor],
             normal_label: int = 0,
             anomaly_label: int = 1) -> Dict[str, float]:
    """Evaluates a trained GAN.

    Reports the real and fake losses for the discriminator, as well as the
    accuracies.

    Args:
        generator (nn.Module): Torch module implementing the GAN generator.
        discriminator (nn.Module): Torch module implementing the GAN 
            discriminator.
        loss_fn (LossFn): Loss function, should return a reduced value.
        real_dataloader (Iterator[torch.Tensor]): Iterator to go over real data 
            samples.
        latent_dataloader (Iterator[torch.Tensor]): Iterator to go through 
            generated samples from the latent space.
        normal_label (int): Label for samples with normal behaviour 
            (real or non-anomaly). Defaults to 0.
        anomaly_label (int): Label that identifies generate samples 
            (anomalies when running inference). Defaults to 1.

    Returns:
        Dict[str, float]: Aggregated metrics.
    """
    generator.eval()
    discriminator.eval()

    agg_metrics: Dict[str, float] = {}
    for real, z in zip(real_dataloader, latent_dataloader):
        bs = real.size(0)
        real_labels = torch.full((bs, ), normal_label).float().to(real.device)
        fake_labels = torch.full((bs, ), anomaly_label).float().to(real.device)
        all_labels = torch.cat([real_labels, fake_labels])

        # Generate fake samples with the generator
        fake = generator(z)

        # Try to classify the real and generated samples
        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        d_logits = torch.cat([real_logits, fake_logits])

        # Discriminator tries to identify the true nature of each sample
        # (anomaly or normal)
        d_real_loss = loss_fn(real_logits.view(-1), real_labels)
        d_fake_loss = loss_fn(fake_logits.view(-1), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        discriminator_acc = ((d_logits > .5) == all_labels).float()
        discriminator_acc = discriminator_acc.sum().div(bs)

        generator_acc = (fake_logits > .5 == real_labels).float()
        generator_acc = generator_acc.sum().div(bs)

        log = {
            "discriminator_real_loss": d_real_loss.item(),
            "discriminator_fake_loss": d_fake_loss.item(),
            "discriminator_loss": d_loss.item(),
            "discriminator_acc": discriminator_acc.item(),
            "generator_acc": generator_acc.item(),
        }

        if not agg_metrics:
            agg_metrics = log
        else:
            agg_metrics = {
                metric_name: (agg_metrics[metric_name] + metric_value) / 2.
                for metric_name, metric_value in log.items()
            }

    log_metrics = " ".join(
        f"{metric_name}:{metric_value:.4f}"
        for metric_name, metric_value in agg_metrics.items())
    print("Evaluation metrics:", log_metrics)
    return agg_metrics
