from __future__ import annotations

from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def autoregressive_rollout(
    model,
    x0: torch.Tensor,
    steps: int,
    *,
    control_sequence: Optional[torch.Tensor] = None,
    teacher_forcing: Optional[torch.Tensor] = None,
    update_memory: bool = False,
    return_outputs: bool = True,
) -> Tuple[torch.Tensor, Optional[List]]:
    if steps <= 0:
        raise ValueError("steps must be > 0")

    B = x0.shape[0]

    preds = torch.zeros(
        B,
        steps,
        *x0.shape[1:],
        device=x0.device,
        dtype=x0.dtype,
    )

    outputs: Optional[List] = [] if return_outputs else None

    x = x0

    for t in range(steps):
        if control_sequence is not None:
            if control_sequence.dim() < 2 or t >= control_sequence.shape[1]:
                raise ValueError("control_sequence shorter than steps or invalid shape")
            control = control_sequence[:, t]
        else:
            control = None

        result = model.step(
            x,
            control=control,
            update_memory=update_memory,
        )

        if isinstance(result, tuple):
            y, out = result
        else:
            y = result
            out = None

        if y.shape != x.shape:
            raise RuntimeError(
                f"Shape mismatch: predicted {y.shape} vs input {x.shape}"
            )

        preds[:, t] = y

        if outputs is not None:
            outputs.append(out)

        if teacher_forcing is not None:
            if teacher_forcing.dim() < 2 or t >= teacher_forcing.shape[1]:
                raise ValueError("teacher_forcing shorter than steps or invalid shape")
            x = teacher_forcing[:, t]
        else:
            x = y

    return preds, outputs