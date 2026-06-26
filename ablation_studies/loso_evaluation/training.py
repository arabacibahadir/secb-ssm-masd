from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_train_epoch(model, loader, optimizer, criterion, device, scaler, amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb, _, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * xb.shape[0]
        total += xb.shape[0]
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device) -> dict:
    model.eval()
    y_true_parts = []
    y_pred_parts = []
    y_proba_parts = []
    record_parts = []
    start_parts = []
    total_loss = 0.0
    total = 0
    for xb, yb, record_ids, starts in loader:
        xb = xb.to(device, non_blocking=True)
        yb_device = yb.to(device, non_blocking=True)
        logits = model(xb)
        probabilities = F.softmax(logits, dim=-1)
        total_loss += float(F.cross_entropy(logits, yb_device).item()) * xb.shape[0]
        total += xb.shape[0]
        y_true_parts.append(yb.numpy())
        y_pred_parts.append(logits.argmax(dim=-1).cpu().numpy())
        y_proba_parts.append(probabilities.cpu().numpy())
        record_parts.append(np.asarray(record_ids))
        start_parts.append(np.asarray(starts))
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_proba = np.concatenate(y_proba_parts)
    result = {
        "loss": total_loss / max(total, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "record_ids": np.concatenate(record_parts),
        "starts": np.concatenate(start_parts),
    }
    result["metrics"] = compute_metrics(y_true, y_pred, y_proba)
    return result


def train_model(
    model,
    train_loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_amp: bool,
    log_prefix: str = "",
):
    amp = bool(use_amp and device.type == "cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 1
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp,
        )
        validation = evaluate_model(model, validation_loader, device)
        scheduler.step()
        validation_score = validation["metrics"]["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["metrics"]["accuracy"],
                "validation_balanced_accuracy": validation["metrics"]["balanced_accuracy"],
                "validation_macro_f1": validation_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"{log_prefix} epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_macro_f1={validation_score:.4f}",
            flush=True,
        )
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch
