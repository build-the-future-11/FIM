from .diffusion_equation import DiffusionBenchmark, DiffusionConfig
from .wave_equation import WaveBenchmark, WaveConfig
from .navier_stokes import Lorenz96Benchmark, Lorenz96Config
from .fractional_pde import FractionalDiffusionBenchmark, FractionalConfig

__all__ = [
    "DiffusionBenchmark",
    "DiffusionConfig",
    "WaveBenchmark",
    "WaveConfig",
    "Lorenz96Benchmark",
    "Lorenz96Config",
    "FractionalDiffusionBenchmark",
    "FractionalConfig",
]