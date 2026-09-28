from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from fim.training.optimizer import build_optimizer, build_scheduler, clip_gradients


@dataclass
class TrainCheckpoint:
    epoch: int
    best_val_loss: float
    history: Dict[str, List[float]]


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: torch.nn.Module) -> None:
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    def to(self, device: torch.device) -> "EMA":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        self.backup = {k: v.to(device) for k, v in self.backup.items()}
        return self


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        for x_key in ("x", "inputs", "input", "source"):
            if x_key in batch:
                x = batch[x_key]
                break
        else:
            raise KeyError("Batch dict must contain one of: x, inputs, input, source")

        for y_key in ("y", "targets", "target", "label", "labels"):
            if y_key in batch:
                y = batch[y_key]
                break
        else:
            raise KeyError("Batch dict must contain one of: y, targets, target, label, labels")
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError("Unsupported batch format")


def _prepare_single_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1).unsqueeze(1)
    if x.ndim == 3:
        if x.shape[-1] <= 512:
            return x[:, 0].unsqueeze(1).unsqueeze(1)
        return x.unsqueeze(1)
    return x


def _align_target_to_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape == target.shape:
        return target

    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch size mismatch between prediction {tuple(pred.shape)} and target {tuple(target.shape)}"
        )

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    if pred_flat.shape[1] == target_flat.shape[1]:
        return target.reshape_as(pred)

    raise ValueError(
        f"Cannot align prediction shape {tuple(pred.shape)} with target shape {tuple(target.shape)}"
    )


def _compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        total += float(grad.norm(2).item() ** 2)
    return float(total ** 0.5)


def _forward(model: torch.nn.Module, x: torch.Tensor, update_memory: bool) -> Tuple[torch.Tensor, Optional[Any]]:
    if hasattr(model, "step"):
        try:
            out = model.step(x, store_traces=update_memory, retrieve=update_memory)
        except TypeError:
            out = model.step(x)
    else:
        out = model(x)

    if hasattr(out, "prediction"):
        return out.prediction, out

    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None

    return out, None


def _reset_model_state(model: torch.nn.Module) -> None:
    for name in ("reset_state", "clear_memory", "detach_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _is_sequence_batch(x: torch.Tensor, y: torch.Tensor, rollout_steps: int) -> bool:
    # A four-dimensional B,C,H,W field batch is a single target. Treating its
    # channel axis as time slices the target to B,H,W and breaks spatial models.
    return (
        x.ndim >= 3
        and x.ndim != 4
        and y.ndim >= 3
        and y.ndim != 4
        and x.shape[0] == y.shape[0]
        and x.shape[1] >= 1
        and y.shape[1] >= 1
    )


def _sequence_targets(x: torch.Tensor, y: torch.Tensor, rollout_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _is_sequence_batch(x, y, rollout_steps):
        return _prepare_single_input(x), y

    steps = min(rollout_steps, y.shape[1])
    seed = x[:, 0]
    targets = y[:, :steps]
    return _prepare_single_input(seed), targets


def _rollout_weights(steps: int, decay: float, device: torch.device) -> torch.Tensor:
    if steps <= 0:
        return torch.empty(0, device=device)
    if decay == 1.0:
        w = torch.ones(steps, device=device)
    else:
        powers = torch.arange(steps, device=device, dtype=torch.float32)
        w = decay ** powers
    return w / w.sum().clamp_min(1e-8)


def _rollout_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rollout_steps: int,
    teacher_forcing_ratio: float,
    horizon_decay: float,
    update_memory: bool,
) -> Tuple[torch.Tensor, Optional[Any], Dict[str, float]]:
    if not _is_sequence_batch(x, y, rollout_steps):
        x = _prepare_single_input(x)
        pred, out = _forward(model, x, update_memory=update_memory)
        target = _align_target_to_prediction(pred, y)
        loss = F.mse_loss(pred, target)
        return loss, out, {"rollout_loss": float(loss.detach().item()), "horizon": 1.0}

    current, targets = _sequence_targets(x, y, rollout_steps)
    steps = targets.shape[1]
    weights = _rollout_weights(steps, horizon_decay, device=targets.device)

    losses = []
    last_out = None
    last_pred = None

    for t in range(steps):
        pred, out = _forward(model, current, update_memory=update_memory)
        last_out = out
        last_pred = pred

        target_t = _align_target_to_prediction(pred, targets[:, t])
        step_loss = F.mse_loss(pred, target_t)
        losses.append(step_loss * weights[t])

        if t + 1 < steps:
            if teacher_forcing_ratio >= 1.0:
                current = _prepare_single_input(targets[:, t])
            elif teacher_forcing_ratio <= 0.0:
                current = pred
            else:
                use_teacher = torch.rand((), device=pred.device) < teacher_forcing_ratio
                teacher = _prepare_single_input(targets[:, t])
                # Avoid host sync on the Bernoulli draw; blend on-device.
                if teacher.shape == pred.shape:
                    current = torch.where(use_teacher, teacher, pred)
                else:
                    current = teacher if bool(use_teacher.item()) else pred

    total = torch.stack(losses).sum()
    stats = {
        "rollout_loss": float(total.detach().item()),
        "horizon": float(steps),
        "last_step_loss": float(F.mse_loss(last_pred, _align_target_to_prediction(last_pred, targets[:, -1])).detach().item()),
    }
    return total, last_out, stats


def _run_epoch_from_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_clip: float,
    scaler: Optional[Any],
    autocast_ctx,
    use_amp: bool,
    rollout_steps: int,
    teacher_forcing_ratio: float,
    horizon_decay: float,
    ema: Optional[EMA],
    scheduler_per_batch: bool,
) -> Dict[str, float]:
    model.train()
    losses = []
    grad_norms = []
    rollout_losses = []

    for batch in loader:
        _reset_model_state(model)
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast_ctx():
                loss, _, stats = _rollout_loss(
                    model=model,
                    x=x,
                    y=y,
                    rollout_steps=rollout_steps,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    horizon_decay=horizon_decay,
                    update_memory=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                clip_gradients(model, grad_clip)
            grad_norms.append(_compute_grad_norm(model.parameters()))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, stats = _rollout_loss(
                model=model,
                x=x,
                y=y,
                rollout_steps=rollout_steps,
                teacher_forcing_ratio=teacher_forcing_ratio,
                horizon_decay=horizon_decay,
                update_memory=True,
            )
            loss.backward()
            if grad_clip > 0:
                clip_gradients(model, grad_clip)
            grad_norms.append(_compute_grad_norm(model.parameters()))
            optimizer.step()

        if scheduler_per_batch and scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        losses.append(float(loss.detach().item()))
        rollout_losses.append(stats["rollout_loss"])

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "rollout_loss": sum(rollout_losses) / max(len(rollout_losses), 1),
        "grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
    }


def _run_epoch_from_benchmark(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_clip: float,
    scaler: Optional[Any],
    autocast_ctx,
    use_amp: bool,
    batch_size: int,
    steps_per_epoch: int,
    rollout_steps: int,
    teacher_forcing_ratio: float,
    horizon_decay: float,
    ema: Optional[EMA],
    scheduler_per_batch: bool,
) -> Dict[str, float]:
    model.train()
    losses = []
    grad_norms = []
    rollout_losses = []

    for _ in range(steps_per_epoch):
        _reset_model_state(model)
        x, y = benchmark.generate_batch(batch_size=batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast_ctx():
                loss, _, stats = _rollout_loss(
                    model=model,
                    x=x,
                    y=y,
                    rollout_steps=rollout_steps,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    horizon_decay=horizon_decay,
                    update_memory=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                clip_gradients(model, grad_clip)
            grad_norms.append(_compute_grad_norm(model.parameters()))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, stats = _rollout_loss(
                model=model,
                x=x,
                y=y,
                rollout_steps=rollout_steps,
                teacher_forcing_ratio=teacher_forcing_ratio,
                horizon_decay=horizon_decay,
                update_memory=True,
            )
            loss.backward()
            if grad_clip > 0:
                clip_gradients(model, grad_clip)
            grad_norms.append(_compute_grad_norm(model.parameters()))
            optimizer.step()

        if scheduler_per_batch and scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        losses.append(float(loss.detach().item()))
        rollout_losses.append(stats["rollout_loss"])

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "rollout_loss": sum(rollout_losses) / max(len(rollout_losses), 1),
        "grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
    }


@torch.no_grad()
def _validate_from_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    rollout_steps: int,
    horizon_decay: float,
) -> Dict[str, float]:
    model.eval()
    losses = []
    rollout_losses = []

    for batch in loader:
        _reset_model_state(model)
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss, _, stats = _rollout_loss(
            model=model,
            x=x,
            y=y,
            rollout_steps=rollout_steps,
            teacher_forcing_ratio=0.0,
            horizon_decay=horizon_decay,
            update_memory=False,
        )
        losses.append(float(loss.item()))
        rollout_losses.append(stats["rollout_loss"])

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "rollout_loss": sum(rollout_losses) / max(len(rollout_losses), 1),
    }


@torch.no_grad()
def _validate_from_benchmark(
    model: torch.nn.Module,
    benchmark: Any,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_steps: int,
    horizon_decay: float,
) -> Dict[str, float]:
    model.eval()
    losses = []
    rollout_losses = []

    for _ in range(batches):
        _reset_model_state(model)
        x, y = benchmark.generate_batch(batch_size=batch_size, device=device)
        loss, _, stats = _rollout_loss(
            model=model,
            x=x,
            y=y,
            rollout_steps=rollout_steps,
            teacher_forcing_ratio=0.0,
            horizon_decay=horizon_decay,
            update_memory=False,
        )
        losses.append(float(loss.item()))
        rollout_losses.append(stats["rollout_loss"])

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "rollout_loss": sum(rollout_losses) / max(len(rollout_losses), 1),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    history: Dict[str, List[float]],
    scaler: Optional[Any] = None,
    ema: Optional[EMA] = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Optional[Any] = None,
    ema: Optional[EMA] = None,
    map_location: str | torch.device = "cpu",
) -> TrainCheckpoint:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])

    return TrainCheckpoint(
        epoch=int(ckpt.get("epoch", 0)),
        best_val_loss=float(ckpt.get("best_val_loss", float("inf"))),
        history=ckpt.get("history", {"train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}),
    )


def train(
    model: torch.nn.Module,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    benchmark: Optional[Any] = None,
    epochs: int = 10,
    device: str | torch.device = "cpu",
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    batch_size: int = 32,
    steps_per_epoch: int = 100,
    val_batches: int = 20,
    rollout_steps: int = 4,
    teacher_forcing_ratio: float = 0.5,
    horizon_decay: float = 1.0,
    ema_decay: float = 0.999,
    use_ema: bool = True,
    evaluate_ema: bool = True,
    scheduler_per_batch: bool = True,
    checkpoint_dir: Optional[str | Path] = None,
    resume_from: Optional[str | Path] = None,
    save_best: bool = True,
    save_last: bool = True,
    checkpoint_name: str = "fim_checkpoint.pt",
    log_every: int = 10,
) -> Dict[str, List[float]]:
    device = torch.device(device)
    model = model.to(device)
    use_amp = bool(use_amp and device.type == "cuda")

    optimizer = build_optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if train_loader is not None:
        total_train_steps = max(1, epochs * max(1, len(train_loader)))
    else:
        total_train_steps = max(1, epochs * max(1, steps_per_epoch))

    scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_train_steps)),
        num_training_steps=total_train_steps,
    )

    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
            autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        scaler = None
        autocast_ctx = nullcontext

    ema = EMA(model, decay=ema_decay) if use_ema else None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_rollout_loss": [],
        "val_loss": [],
        "val_rollout_loss": [],
        "lr": [],
        "grad_norm": [],
    }

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from is not None:
        state = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            map_location=device,
        )
        start_epoch = state.epoch + 1
        best_val_loss = state.best_val_loss
        for key, values in state.history.items():
            if key in history and isinstance(values, list):
                history[key].extend(values)

    if train_loader is None and benchmark is None:
        raise ValueError("train() requires either train_loader or benchmark")

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        if train_loader is not None:
            train_stats = _run_epoch_from_loader(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                use_amp=use_amp,
                rollout_steps=rollout_steps,
                teacher_forcing_ratio=teacher_forcing_ratio,
                horizon_decay=horizon_decay,
                ema=ema,
                scheduler_per_batch=scheduler_per_batch,
            )
        else:
            train_stats = _run_epoch_from_benchmark(
                model=model,
                benchmark=benchmark,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                use_amp=use_amp,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                rollout_steps=rollout_steps,
                teacher_forcing_ratio=teacher_forcing_ratio,
                horizon_decay=horizon_decay,
                ema=ema,
                scheduler_per_batch=scheduler_per_batch,
            )

        history["train_loss"].append(train_stats["loss"])
        history["train_rollout_loss"].append(train_stats["rollout_loss"])
        history["grad_norm"].append(train_stats["grad_norm"])

        if scheduler is not None and not scheduler_per_batch and hasattr(scheduler, "step"):
            scheduler.step()

        if val_loader is not None:
            if evaluate_ema and ema is not None:
                ema.apply(model)
                try:
                    val_stats = _validate_from_loader(model, val_loader, device, rollout_steps, horizon_decay)
                finally:
                    ema.restore(model)
            else:
                val_stats = _validate_from_loader(model, val_loader, device, rollout_steps, horizon_decay)

            history["val_loss"].append(val_stats["loss"])
            history["val_rollout_loss"].append(val_stats["rollout_loss"])
            best_val_loss = min(best_val_loss, val_stats["loss"])
            if epoch % max(1, log_every) == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}: train={train_stats['loss']:.4f}, "
                    f"train_rollout={train_stats['rollout_loss']:.4f}, "
                    f"val={val_stats['loss']:.4f}, val_rollout={val_stats['rollout_loss']:.4f}"
                )
        elif benchmark is not None:
            if evaluate_ema and ema is not None:
                ema.apply(model)
                try:
                    val_stats = _validate_from_benchmark(
                        model=model,
                        benchmark=benchmark,
                        device=device,
                        batch_size=batch_size,
                        batches=max(1, min(val_batches, steps_per_epoch)),
                        rollout_steps=rollout_steps,
                        horizon_decay=horizon_decay,
                    )
                finally:
                    ema.restore(model)
            else:
                val_stats = _validate_from_benchmark(
                    model=model,
                    benchmark=benchmark,
                    device=device,
                    batch_size=batch_size,
                    batches=max(1, min(val_batches, steps_per_epoch)),
                    rollout_steps=rollout_steps,
                    horizon_decay=horizon_decay,
                )

            history["val_loss"].append(val_stats["loss"])
            history["val_rollout_loss"].append(val_stats["rollout_loss"])
            best_val_loss = min(best_val_loss, val_stats["loss"])
            if epoch % max(1, log_every) == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}: train={train_stats['loss']:.4f}, "
                    f"train_rollout={train_stats['rollout_loss']:.4f}, "
                    f"val={val_stats['loss']:.4f}, val_rollout={val_stats['rollout_loss']:.4f}"
                )
        else:
            if epoch % max(1, log_every) == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}: train={train_stats['loss']:.4f}, "
                    f"train_rollout={train_stats['rollout_loss']:.4f}"
                )

        lr_value = optimizer.param_groups[0]["lr"]
        history["lr"].append(float(lr_value))

        if checkpoint_dir is not None and save_last:
            save_checkpoint(
                checkpoint_dir / checkpoint_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                scaler=scaler,
                ema=ema,
            )

        if checkpoint_dir is not None and save_best:
            current_val = history["val_loss"][-1] if history["val_loss"] else float("inf")
            if current_val <= best_val_loss:
                save_checkpoint(
                    checkpoint_dir / f"best_{checkpoint_name}",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=current_val,
                    history=history,
                    scaler=scaler,
                    ema=ema,
                )

        if hasattr(model, "detach_state"):
            try:
                model.detach_state()
            except Exception:
                pass

    return history
