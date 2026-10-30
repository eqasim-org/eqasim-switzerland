"""Independent person-fold training for epoch selection, followed by a full refit."""
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def train_cross_validated(model, tensors, fold_indices, validation_groups, score_samples, *,
                          epochs, batch_size, lr, weight_decay, lr_step_size, lr_gamma,
                          grad_clip, patience, min_delta, min_epochs, seed, logger,
                          path, history_settings, predict_utilities=None):
    """Keep each fold model's held-out people permanently out of its training data."""

    # Share prepared tensors across independent model/optimizer states.

    person_static, person_dynamic, candidate_static, candidate_dynamic, mask, weights = tensors
    device = model._device if hasattr(model, '_device') else next(model.parameters()).device
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    all_rows = torch.arange(len(weights), device=device)
    models = [deepcopy(model) for _ in fold_indices]
    optimizers = [torch.optim.AdamW(fold.parameters(), lr=lr, weight_decay=weight_decay) for fold in models]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
                  for optimizer in optimizers]
    train_rows = [torch.cat([indices for other, indices in enumerate(fold_indices) if other != fold])
                  for fold in range(len(fold_indices))]
    audit = []
    for train, valid in zip(train_rows, fold_indices):
        train_ids = np.unique(validation_groups[train.cpu().numpy()])
        valid_ids = np.unique(validation_groups[valid.cpu().numpy()])
        if np.intersect1d(train_ids, valid_ids).size:
            raise ValueError('Cross-validation people must be disjoint within every fold')
        audit.append(dict(train_people=train_ids.tolist(), validation_people=valid_ids.tolist(),
                          train_rows=len(train), validation_rows=len(valid)))

    def utilities_for(network, indices):
        if predict_utilities is not None:
            return predict_utilities(network, indices)
        static = candidate_static if candidate_static.shape[0] == 1 else candidate_static[indices]
        return network(person_static[indices], person_dynamic[indices], static,
                       candidate_dynamic[indices]).masked_fill(~mask[indices], -1e9)

    def fit_epoch(network, optimizer, indices, epoch_seed):
        torch.manual_seed(epoch_seed)
        network.train()
        order = indices[torch.randperm(len(indices), device=device)]
        numerator, denominator = 0., 0.
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            values = score_samples(rows, utilities_for(network, rows))
            weighted = (values * weights[rows]).sum()
            total = weights[rows].sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            (weighted / total).backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), float(grad_clip))
            optimizer.step()
            numerator += float(weighted.detach())
            denominator += float(total)
        return numerator, denominator

    def evaluate(network, indices):
        network.eval()
        numerator, denominator = 0., 0.
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                values = score_samples(rows, utilities_for(network, rows))
                numerator += float((values * weights[rows]).sum())
                denominator += float(weights[rows].sum())
        return numerator, denominator

    # Evaluate all untouched folds at every epoch, never rotating people through one model.

    train_curve, validation_curve, fold_curves = [], [], []
    best_loss, best_epoch, bad_epochs = np.inf, None, 0
    stopped = False
    logger.info('Independent %s-fold person CV; minimum training=%s epochs; validation each epoch',
                len(models), min_epochs)
    for epoch in range(1, epochs + 1):
        train_sum = train_weight = validation_sum = validation_weight = 0.
        values = []
        for fold, (network, optimizer, scheduler, train, valid) in enumerate(
                zip(models, optimizers, schedulers, train_rows, fold_indices)):
            numerator, denominator = fit_epoch(network, optimizer, train,
                                                seed + fold * 1000003 + epoch)
            train_sum += numerator
            train_weight += denominator
            numerator, denominator = evaluate(network, valid)
            validation_sum += numerator
            validation_weight += denominator
            values.append(numerator / denominator)
            scheduler.step()
        loss = validation_sum / validation_weight
        if not np.isfinite(loss):
            raise ValueError('Cross-validation objective became nonfinite')
        train_curve.append(train_sum / train_weight)
        validation_curve.append(loss)
        fold_curves.append(values)
        improved = loss < best_loss - min_delta
        if improved:
            best_loss, best_epoch, bad_epochs = loss, epoch, 0
        elif epoch > min_epochs:
            bad_epochs += 1
        else:
            bad_epochs = 0
        logger.info('CV epoch %s/%s | Train %.4f | Validation %.4f | Fold losses %s%s',
                    epoch, epochs, train_curve[-1], loss,
                    ', '.join(f'{value:.4f}' for value in values), ' | best' if improved else '')
        if patience is not None and epoch >= min_epochs and bad_epochs >= patience:
            stopped = True
            break

    # Select duration from CV, then restart one model on every supplied training row.

    del models, optimizers, schedulers
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    logger.info('CV selected epoch %s; refitting on all %s rows', best_epoch, len(weights))
    for epoch in range(1, best_epoch + 1):
        fit_epoch(model, optimizer, all_rows, seed + epoch)
        scheduler.step()
    model.eval()

    # Keep fold membership and curves available for auditing and report interpretation.

    history = dict(history_settings, train_losses=train_curve, val_losses=validation_curve,
            val_epochs=list(range(1, len(validation_curve) + 1)),
            validation_mode='independent_kfold', validation_grouped=True,
            validation_folds=len(fold_indices), validation_every=1,
            validation_fraction=1. / len(fold_indices), fold_val_losses=fold_curves,
            best_epoch=best_epoch, best_val_loss=best_loss,
            early_stopping_min_epochs=min_epochs, early_stopping_patience=patience,
            early_stopping_min_delta=min_delta, stopped_early=stopped,
            epochs_completed=len(train_curve), refit_full=True, refit_rows=len(weights),
            refit_epochs=best_epoch, cross_validation_seed=seed)
    if path is not None:
        directory = Path(path)
        (directory / 'loss_history.json').write_text(json.dumps(history, indent=2))
        audit_path = directory / 'cross_validation_folds.json'
        audit_path.write_text(json.dumps(audit, indent=2))
        audit_path.chmod(0o600)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['val_epochs'], train_curve, label='Mean fold training loss')
        for fold in np.asarray(fold_curves).T:
            ax.plot(history['val_epochs'], fold, color='#bbb', alpha=.5, linewidth=.7)
        ax.plot(history['val_epochs'], validation_curve, label='Weighted held-out CV loss', color='#D55E00')
        ax.axvline(best_epoch, color='#009E73', linestyle='--', label=f'Selected epoch {best_epoch}')
        if min_epochs:
            ax.axvline(min_epochs, color='#777', linestyle=':', label='Minimum exploration period')
        ax.set(xlabel='Epoch', ylabel='Weighted training objective', title='Independent person-fold cross-validation')
        ax.legend()
        fig.tight_layout()
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(directory / ('lossevolution.' + extension), dpi=600)
        plt.close(fig)
    return history
