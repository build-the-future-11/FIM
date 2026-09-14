from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from fim.training.losses import fim_total_loss
from fim.training.optimizer import build_optimizer, clip_gradients, EMA
from fim.training.scheduler import build_scheduler


@dataclass
class TrainStats:
    loss: float
    pred_loss: float
    mem_loss: float
    stab_loss: float
    retr_loss: float
    cons_loss: float


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        device: str | torch.device = "cpu",
        grad_clip: float = 1.0,
        use_amp: bool = True,
        use_ema: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.grad_clip = float(grad_clip)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.autocast = torch.cuda.amp.autocast if self.use_amp else torch.cpu.amp.autocast

        self.optimizer = build_optimizer(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.ema = EMA(self.model) if use_ema else None

    def set_scheduler(
        self,
        total_steps: int,
        warmup_steps: int = 0,
    ) -> None:
        self.scheduler = build_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _safe_retrieval(self, out) -> torch.Tensor:
        if getattr(out, "retrieved", None) is None:
            return torch.zeros(
                out.state.tensor.shape[0],
                out.state.tensor.shape[1],
                device=out.state.tensor.device,
            )
        return out.retrieved

    def _mask_from_salience(self, out) -> torch.Tensor:
        if hasattr(out, "salience"):
            return (out.salience > 0.5).float().view(-1)
        return torch.zeros(out.state.tensor.shape[0], device=out.state.tensor.device)

    def train_step(self, batch) -> TrainStats:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        with self.autocast(enabled=self.use_amp):
            out = self.model.step(x)

            pred = out.prediction

            retrieval = self._safe_retrieval(out)
            mask = self._mask_from_salience(out)

            loss_out = fim_total_loss(
                pred=pred,
                target=y,
                state=out.state.tensor,
                stored_mask=mask,
                retrieval_context=retrieval,
                pre_retrieval_state=out.state.tensor,
                post_retrieval_state=out.state.tensor,
                w_pred=1.0,
                w_mem=0.05,
                w_stab=0.05,
                w_sparse=0.05,
                w_retr=0.1,
                w_cons=0.05,
            )

        self.scaler.scale(loss_out.total).backward()

        self.scaler.unscale_(self.optimizer)
        clip_gradients(self.model, self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model)

        return TrainStats(
            loss=float(loss_out.total.item()),
            pred_loss=float(loss_out.prediction.item()),
            mem_loss=float(loss_out.memory.item()),
            stab_loss=float(loss_out.stability.item()),
            retr_loss=float(loss_out.retrieval.item()),
            cons_loss=float(loss_out.consistency.item()),
        )

    @torch.no_grad()
    def eval_step(self, batch) -> TrainStats:
        self.model.eval()

        if self.ema is not None:
            self.ema.apply_to(self.model)

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model.step(x)

        retrieval = self._safe_retrieval(out)
        mask = self._mask_from_salience(out)

        loss_out = fim_total_loss(
            pred=out.prediction,
            target=y,
            state=out.state.tensor,
            stored_mask=mask,
            retrieval_context=retrieval,
            pre_retrieval_state=out.state.tensor,
            post_retrieval_state=out.state.tensor,
            w_pred=1.0,
            w_mem=0.05,
            w_stab=0.05,
            w_sparse=0.05,
            w_retr=0.1,
            w_cons=0.05,
        )

        return TrainStats(
            loss=float(loss_out.total.item()),
            pred_loss=float(loss_out.prediction.item()),
            mem_loss=float(loss_out.memory.item()),
            stab_loss=float(loss_out.stability.item()),
            retr_loss=float(loss_out.retrieval.item()),
            cons_loss=float(loss_out.consistency.item()),
        )