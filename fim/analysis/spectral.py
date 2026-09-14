from __future__ import annotations

import torch


def fft_energy(tensor: torch.Tensor) -> torch.Tensor:
    """Total Fourier energy of a spatial tensor."""
    freq = torch.fft.fftn(tensor, dim=tuple(range(-2, 0)))
    return torch.sum(torch.abs(freq) ** 2)


def radial_spectrum(tensor: torch.Tensor) -> torch.Tensor:
    """Compute a crude radial spectrum for a 2D field.

    Returns mean energy per radial shell.
    """
    if tensor.ndim < 2:
        raise ValueError("tensor must be at least 2D")
    spatial = tensor
    if tensor.ndim > 2:
        spatial = tensor.flatten(start_dim=0, end_dim=-3) if tensor.ndim > 3 else tensor
    # use last two dims
    h, w = tensor.shape[-2], tensor.shape[-1]
    freq = torch.fft.fftn(tensor, dim=(-2, -1))
    energy = torch.abs(freq) ** 2

    yy = torch.fft.fftfreq(h, d=1.0, device=tensor.device)[:, None]
    xx = torch.fft.fftfreq(w, d=1.0, device=tensor.device)[None, :]
    rr = torch.sqrt(yy**2 + xx**2)

    bins = torch.linspace(0.0, rr.max().item() + 1e-8, steps=min(h, w) // 2 + 1, device=tensor.device)
    shell_vals = []
    flat_energy = energy.mean(dim=tuple(range(energy.ndim - 2))) if energy.ndim > 2 else energy
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        if mask.any():
            shell_vals.append(flat_energy[mask].mean())
        else:
            shell_vals.append(torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    return torch.stack(shell_vals)


def spectral_centroid(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    spectrum = radial_spectrum(tensor)
    freqs = torch.arange(spectrum.numel(), device=tensor.device, dtype=tensor.dtype)
    return torch.sum(freqs * spectrum) / torch.sum(spectrum).clamp_min(eps)


def spectral_entropy(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    spectrum = radial_spectrum(tensor)
    p = spectrum / spectrum.sum().clamp_min(eps)
    return -torch.sum(p * torch.log(p.clamp_min(eps)))
