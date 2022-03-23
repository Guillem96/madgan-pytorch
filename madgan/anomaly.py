import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(object):

    def __init__(self,
                 *,
                 discriminator: nn.Module,
                 generator: nn.Module,
                 latent_space_dim: int,
                 res_weight: float = .2,
                 anomaly_threshold: float = 1.0) -> None:
        self.discriminator = discriminator
        self.generator = generator
        self.threshold = anomaly_threshold
        self.latent_space_dim = latent_space_dim
        self.res_weight = res_weight

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(tensor) > self.anomaly_threshold).int()

    def predict_proba(self, tensor: torch.Tensor) -> torch.Tensor:
        discriminator_score = self.compute_anomaly_score(tensor)
        discriminator_score *= 1 - self.res_weight
        reconstruction_loss = self.compute_reconstruction_loss(tensor)
        reconstruction_loss *= self.res_weight
        return reconstruction_loss + discriminator_score

    def compute_anomaly_score(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            discriminator_score = self.discriminator(tensor)
        return discriminator_score

    def compute_reconstruction_loss(self,
                                    tensor: torch.Tensor) -> torch.Tensor:
        best_reconstruct = self._generate_best_reconstruction(tensor)
        return (best_reconstruct - tensor).abs().sum(dim=(1, 2))

    def _generate_reconstruction(self, tensor: torch.Tensor) -> None:
        # The goal of this function is to find the corresponding latent space for the given
        # input and then generate the best possible reconstruction.
        max_iters = 10

        Z = torch.empty(
            (tensor.size(0), tensor.size(1), self.latent_space_dim),
            requires_grad=True)
        nn.init.normal_(Z, std=0.05)

        optimizer = torch.optim.RMSprop(params=[Z], lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")
        normalized_target = F.normalize(tensor, dim=1, p=2)

        for _ in range(max_iters):
            optimizer.zero_grad()
            generated_samples = self.generator(Z)
            normalized_input = F.normalize(generated_samples, dim=1, p=2)
            reconstruction_error = loss_fn(normalized_input,
                                           normalized_target).sum(dim=(1, 2))
            reconstruction_error.backward()
            optimizer.step()

        with torch.no_grad():
            best_reconstruct = self.generator(Z)
        return best_reconstruct
