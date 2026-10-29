"""Per-person behavioural objectives and representation regularizers.

Sequence padding and missing measurements never contribute to an objective.
Survey weights are applied by the trainer after per-person aggregation.
"""

import numpy as np

from .runtime import torch, F


class MatchingLoss:
    """Stateless loss components, independently usable in training and validation."""

    @staticmethod
    def class_weights(targets, sample_weight, count):
        valid = targets >= 0
        weights = np.broadcast_to(sample_weight[:, None], targets.shape)[valid]
        frequencies = np.bincount(targets[valid], weights=weights, minlength=count)
        # Square-root weighting avoids inverse-frequency explosions for rare purposes.
        result = np.sqrt(frequencies.sum() / np.maximum(frequencies, 1))
        result[frequencies == 0] = 0
        result /= max(result[result > 0].mean(), 1e-8)
        return result.astype(np.float32)

    @staticmethod
    def categorical(logits, target, class_weight=None):
        valid = target >= 0
        loss = F.cross_entropy(logits.transpose(1, 2), target, weight=class_weight,
                               ignore_index=-100, reduction="none")
        return loss.sum(1) / valid.sum(1).clamp(min=1)

    @staticmethod
    def masked_regression(prediction, target, mask):
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        return (error * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def trip_count(prediction, target):
        return F.smooth_l1_loss(prediction, target, reduction="none")

    @staticmethod
    def reconstruction(prediction, encoder_input):
        # Prevent the projections from shrinking the reconstruction target.
        return ((prediction - encoder_input.detach()) ** 2).mean(1)

    @staticmethod
    def latent_norm(latent):
        return (latent ** 2).mean(1)

    @staticmethod
    def latent_spread(latent):
        return F.relu(0.5 - latent.var(dim=0, unbiased=False).mean())

    @classmethod
    def per_person(cls, output, targets, indices, device, weights, class_weights):
        """Return the weighted objective and its unweighted per-person components."""
        losses = {}
        for name, key in (("activity", "act"), ("mode", "mode")):
            target = torch.as_tensor(targets[key + "_tgt"][indices], dtype=torch.long, device=device)
            losses[name] = cls.categorical(output[name], target, class_weights.get(name))
        for name, key in (("departure", "dep"), ("distance", "dist"),
                          ("duration", "duration"), ("activity_duration", "activity_duration")):
            target = torch.as_tensor(targets[key + "_tgt"][indices], device=device)
            mask = torch.as_tensor(targets[key + "_mask"][indices], device=device)
            losses[name] = cls.masked_regression(output[name], target, mask)
        count = torch.as_tensor(targets["trip_count_tgt"][indices], device=device)
        losses["trip_count"] = cls.trip_count(output["trip_count"], count)
        losses["reconstruction"] = cls.reconstruction(output["reconstruction"], output["encoder_input"])
        losses["latent_norm"] = cls.latent_norm(output["latent"])
        return sum(float(weights.get(name, 0)) * value for name, value in losses.items()), losses
