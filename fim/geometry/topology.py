from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch


def _coords(height: int, width: int, device=None):
    yy = torch.arange(height, device=device)[:, None].expand(height, width)
    xx = torch.arange(width, device=device)[None, :].expand(height, width)
    return yy, xx


def neighborhood_mask(
    height: int,
    width: int,
    center: Tuple[int, int],
    radius: int = 1,
    *,
    metric: Literal["linf", "l1", "l2"] = "linf",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    cy, cx = center
    if not (0 <= cy < height and 0 <= cx < width):
        raise ValueError("center must lie within the grid")

    yy, xx = _coords(height, width, device=device)

    if metric == "linf":
        mask = (yy - cy).abs().le(radius) & (xx - cx).abs().le(radius)
    elif metric == "l1":
        mask = (yy - cy).abs() + (xx - cx).abs() <= radius
    elif metric == "l2":
        mask = ((yy - cy).float() ** 2 + (xx - cx).float() ** 2).le(radius**2)
    else:
        raise ValueError("metric must be one of: linf, l1, l2")

    return mask.to(dtype)


def radial_kernel(
    height: int,
    width: int,
    center: Tuple[int, int],
    *,
    sigma: float = 1.0,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    cy, cx = center
    yy, xx = _coords(height, width, device=device)

    dist2 = (yy - cy).float() ** 2 + (xx - cx).float() ** 2
    kernel = torch.exp(-dist2 / (2 * sigma**2))

    if normalize:
        kernel = kernel / kernel.sum().clamp_min(1e-8)

    return kernel


def multi_center_mask(
    height: int,
    width: int,
    centers: torch.Tensor,
    radius: int,
    *,
    metric: Literal["linf", "l1", "l2"] = "linf",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be (N, 2)")

    mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    for cy, cx in centers.tolist():
        mask |= neighborhood_mask(
            height,
            width,
            (int(cy), int(cx)),
            radius,
            metric=metric,
            device=device,
        )

    return mask


def adjacency_from_grid(
    height: int,
    width: int,
    *,
    connectivity: Literal[4, 8] = 4,
    device: torch.device | str | None = None,
    self_loops: bool = False,
    normalized: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    n = height * width

    def idx(i: int, j: int) -> int:
        return i * width + j

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    rows = []
    cols = []

    for i in range(height):
        for j in range(width):
            u = idx(i, j)
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    v = idx(ni, nj)
                    rows.append(u)
                    cols.append(v)

    if self_loops:
        for k in range(n):
            rows.append(k)
            cols.append(k)

    indices = torch.tensor([rows, cols], device=device, dtype=torch.long)
    values = torch.ones(indices.shape[1], device=device)

    if sparse:
        adj = torch.sparse_coo_tensor(indices, values, (n, n))
    else:
        adj = torch.zeros((n, n), device=device, dtype=torch.float32)
        adj[indices[0], indices[1]] = 1.0

    if normalized:
        if sparse:
            deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1e-6)
            inv_deg = 1.0 / deg
            values = values * inv_deg[indices[0]]
            adj = torch.sparse_coo_tensor(indices, values, (n, n))
        else:
            deg = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
            adj = adj / deg

    return adj


def grid_laplacian(
    height: int,
    width: int,
    *,
    connectivity: Literal[4, 8] = 4,
    device: torch.device | str | None = None,
    normalized: bool = True,
) -> torch.Tensor:
    adj = adjacency_from_grid(
        height,
        width,
        connectivity=connectivity,
        device=device,
        self_loops=False,
        normalized=False,
        sparse=False,
    )

    deg = adj.sum(dim=1)
    D = torch.diag(deg)

    L = D - adj

    if normalized:
        deg_inv_sqrt = deg.clamp_min(1e-6).pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt

    return L


def neighborhood_indices(
    height: int,
    width: int,
    center: Tuple[int, int],
    radius: int,
    *,
    metric: Literal["linf", "l1", "l2"] = "linf",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    mask = neighborhood_mask(
        height,
        width,
        center,
        radius,
        metric=metric,
        device=device,
    )
    idx = torch.nonzero(mask, as_tuple=False)
    return idx