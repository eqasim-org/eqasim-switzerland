"""Person-grouped training schedules and checkpoint selection."""
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


def rotating_holdout_split(groups, folds, seed, fraction):
    """Reserve people before training; partition only the remaining people into folds."""
    groups = np.asarray(groups)
    people = np.random.default_rng(seed).permutation(np.unique(groups))
    if not 0 < fraction < .5 or folds < 2:
        raise ValueError('Reserved holdout requires 0 < fraction < .5 and at least two folds')
    count = max(1, int(np.ceil(len(people) * fraction)))
    if len(people) - count < folds:
        raise ValueError('Not enough development people for the requested rotation folds')
    held = np.flatnonzero(np.isin(groups, people[:count]))
    rotating = [np.flatnonzero(np.isin(groups, part))
                for part in np.array_split(people[count:], folds)]
    return held, rotating


def train_rotating_holdout(model, tensors, fold_indices, validation_groups, score_samples, *,
                           epochs, batch_size, lr, weight_decay, lr_step_size, lr_gamma,
                           grad_clip, patience, min_delta, min_epochs, seed, logger,
                           path, history_settings, predict_utilities=None,
                           rotation_epochs=3, holdout_fraction=.1):
    """Train one continuous model, then stop on people untouched by the warm-up."""

    # Reserve validation people before any gradient update and partition the rest.

    if min_epochs < 0 or epochs <= min_epochs or rotation_epochs < 1:
        raise ValueError('Rotating holdout requires epochs > minimum warm-up epochs >= 0 and a positive rotation interval')
    person_static, person_dynamic, candidate_static, candidate_dynamic, mask, weights = tensors
    device = next(model.parameters()).device
    held, rotating = rotating_holdout_split(validation_groups, len(fold_indices), seed, holdout_fraction)
    holdout = torch.as_tensor(held, device=device)
    folds = [torch.as_tensor(rows, device=device) for rows in rotating]
    development = torch.cat(folds)
    train_folds = [torch.cat([rows for number, rows in enumerate(folds) if number != omitted])
                   for omitted in range(len(folds))]
    rng = np.random.default_rng(seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Keep network and optimizer state throughout both phases.

    def utilities(rows):
        if predict_utilities is not None:
            return predict_utilities(model, rows)
        static = candidate_static if candidate_static.shape[0] == 1 else candidate_static[rows]
        return model(person_static[rows], person_dynamic[rows], static,
                     candidate_dynamic[rows]).masked_fill(~mask[rows], -1e9)

    def run_epoch(rows, training):
        model.train(training)
        order = rows[torch.randperm(len(rows), device=device)] if training else rows
        numerator, denominator = 0., 0.
        with torch.set_grad_enabled(training):
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                values = score_samples(batch, utilities(batch))
                total = weights[batch].sum().clamp_min(1e-12)
                weighted = (values * weights[batch]).sum()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    (weighted / total).backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                numerator += float(weighted.detach())
                denominator += float(total)
        return numerator / denominator

    # Rotate diagnostic validation folds during warm-up; stop only on the reserved set.

    train_curve, validation_curve, epoch_folds = [], [], []
    best, best_epoch, best_state, bad_epochs = np.inf, None, None, 0
    previous_fold, stopped = None, False
    logger.info('One continuous model: %s warm-up epochs, rotating %s folds every %s epochs; %.0f%% reserved people; no refit',
                min_epochs, len(folds), rotation_epochs, 100 * holdout_fraction)
    for epoch in range(1, epochs + 1):
        torch.manual_seed(seed + epoch)
        warmup = epoch <= min_epochs
        if warmup:
            if (epoch - 1) % rotation_epochs == 0:
                choices = [number for number in range(len(folds)) if number != previous_fold]
                previous_fold = int(rng.choice(choices))
            train, valid, fold = train_folds[previous_fold], folds[previous_fold], previous_fold
        else:
            train, valid, fold = development, holdout, -1
        training_loss = run_epoch(train, True)
        validation_loss = run_epoch(valid, False)
        if not np.isfinite(training_loss) or not np.isfinite(validation_loss):
            raise ValueError('Rotating-holdout objective became nonfinite')
        train_curve.append(training_loss)
        validation_curve.append(validation_loss)
        epoch_folds.append(fold)
        improved = False
        if not warmup:
            improved = validation_loss < best - min_delta
            if improved:
                best, best_epoch, bad_epochs = validation_loss, epoch, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                bad_epochs += 1
        scheduler.step()
        logger.info('Epoch %s/%s | %s | Train %.4f | Validation %.4f%s', epoch, epochs,
                    f'rotation fold {fold}' if warmup else 'reserved holdout', training_loss,
                    validation_loss, ' | best' if improved else '')
        if not warmup and patience is not None and bad_epochs >= patience:
            stopped = True
            break
    model.load_state_dict(best_state)
    model.eval()

    # Save phase-aware history and person membership for leakage checks.

    history = dict(history_settings, train_losses=train_curve, val_losses=validation_curve,
                   val_epochs=list(range(1, len(train_curve) + 1)), validation_mode='rotating_holdout',
                   validation_grouped=True, validation_folds=len(folds), validation_every=1,
                   validation_fraction=holdout_fraction, validation_rotation_epochs=rotation_epochs,
                   restore_best_weights=True,
                   validation_fold_by_epoch=epoch_folds, fixed_holdout_start_epoch=min_epochs + 1,
                   best_epoch=best_epoch, best_val_loss=best, early_stopping_min_epochs=min_epochs,
                   early_stopping_patience=patience, early_stopping_min_delta=min_delta,
                   stopped_early=stopped, epochs_completed=len(train_curve), refit_full=False,
                   refit_rows=0, refit_epochs=0, cross_validation_seed=seed,
                   training_rows=len(development), reserved_validation_rows=len(holdout))
    if path is not None:
        directory = Path(path)
        (directory / 'loss_history.json').write_text(json.dumps(history, indent=2))
        groups = np.asarray(validation_groups)
        audit = dict(reserved_people=np.unique(groups[held]).tolist(),
                     rotation_person_folds=[np.unique(groups[rows]).tolist() for rows in rotating],
                     validation_fold_by_epoch=epoch_folds)
        audit_path = directory / 'cross_validation_folds.json'
        audit_path.write_text(json.dumps(audit, indent=2))
        audit_path.chmod(0o600)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['val_epochs'], train_curve, label='Training objective')
        ax.plot(history['val_epochs'], validation_curve, label='Validation (rotating, then reserved)', color='#D55E00')
        ax.axvline(min_epochs + .5, color='#777', linestyle=':', label='Start reserved validation')
        ax.axvline(best_epoch, color='#009E73', linestyle='--', label=f'Best reserved checkpoint {best_epoch}')
        ax.set(xlabel='Epoch', ylabel='Weighted training objective', title='Continuous training with a reserved person holdout')
        ax.legend()
        fig.tight_layout()
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(directory / ('lossevolution.' + extension), dpi=600)
        plt.close(fig)
    return history
