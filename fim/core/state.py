from __future__ import annotations

from typing import Any, Optional

import torch


class FabricState:
    """
    Container for a fabric tensor and metadata.

    Supports:
        FabricState(batch_size=..., channels=..., height=..., width=...)
        FabricState(B, C, H, W)
        FabricState(tensor)
    """

    def __init__(
        self,
        *args: Any,
        tensor: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        step: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if tensor is not None:
            self.tensor = tensor
        elif len(args) == 1 and isinstance(args[0], torch.Tensor):
            self.tensor = args[0]
        else:
            if len(args) == 4 and all(isinstance(a, int) for a in args):
                batch_size, channels, height, width = args
            elif all(v is not None for v in (batch_size, channels, height, width)):
                batch_size = int(batch_size)  # type: ignore[arg-type]
                channels = int(channels)      # type: ignore[arg-type]
                height = int(height)          # type: ignore[arg-type]
                width = int(width)            # type: ignore[arg-type]
            else:
                raise TypeError(
                    "FabricState requires either a tensor, four positional dims, "
                    "or keyword dims batch_size/channels/height/width."
                )

            self.tensor = torch.zeros(
                batch_size,  # type: ignore[arg-type]
                channels,    # type: ignore[arg-type]
                height,      # type: ignore[arg-type]
                width,       # type: ignore[arg-type]
                device=device,
                dtype=dtype,
            )

        self.step = int(step)
        self.metadata = dict(metadata or {})

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def clone(self) -> "FabricState":
        return FabricState(
            tensor=self.tensor.clone(),
            step=self.step,
            metadata=dict(self.metadata),
        )

    def detach(self) -> "FabricState":
        return FabricState(
            tensor=self.tensor.detach(),
            step=self.step,
            metadata=dict(self.metadata),
        )

    def to(self, *args: Any, **kwargs: Any) -> "FabricState":
        return FabricState(
            tensor=self.tensor.to(*args, **kwargs),
            step=self.step,
            metadata=dict(self.metadata),
        )

    def increment(self, n: int = 1) -> "FabricState":
        self.step += int(n)
        return self

    def with_metadata(self, **kwargs: Any) -> "FabricState":
        meta = dict(self.metadata)
        meta.update(kwargs)
        return FabricState(
            tensor=self.tensor,
            step=self.step,
            metadata=meta,
        )