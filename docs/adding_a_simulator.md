# Adding a Real Ray-Tracing Simulator

The `mock_simulate()` function in `src/parameter_sweep/generate_grid.py` is a
placeholder that generates toy spectra. Replace it with your actual simulator.

## Option A: Subprocess wrapper (Fortran/C binary)

```python
import subprocess, tempfile, os, numpy as np

def mock_simulate(spin, r_bp, beta, phi, inclination) -> np.ndarray:
    """Wrapper for external ray-tracing binary."""
    out_file = tempfile.mktemp(suffix=".npy")
    try:
        subprocess.run(
            [
                "/path/to/raytrace_binary",
                f"--spin={spin:.4f}",
                f"--r_bp={r_bp:.4f}",
                f"--beta={beta:.4f}",
                f"--phi={phi:.4f}",
                f"--incl={inclination:.4f}",
                f"--n_photons=1000000",
                f"--output={out_file}",
            ],
            check=True,
            capture_output=True,
        )
        spectrum = np.load(out_file)   # expect shape (N_energy, 3)
    finally:
        if os.path.exists(out_file):
            os.remove(out_file)
    return spectrum
```

## Option B: Python-wrapped Krawczynski code

If the ray-tracing code has a Python interface:

```python
from raytrace import RayTracer  # your module

_tracer = RayTracer(n_photons=3_500_000)

def mock_simulate(spin, r_bp, beta, phi, inclination) -> np.ndarray:
    result = _tracer.run(
        spin=spin, r_bp=r_bp, beta=beta,
        phi=phi, inclination=inclination
    )
    return result.as_numpy()   # (N_energy, 3)
```

## Option C: Parallelised sweep

For large sweeps, wrap the call with `concurrent.futures`:

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def run_sweep_parallel(params_array, param_names, output_path, n_workers=8):
    def job(row):
        return mock_simulate(**dict(zip(param_names, row)))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        spectra = list(ex.map(job, params_array))

    X = np.array(spectra)
    y = np.array(params_array)
    save_dataset_hdf5(X, y, output_path, param_names=param_names)
```

## Output contract

Whatever simulator you use, `mock_simulate()` **must return** a numpy array
of shape `(N_energy, 3)` with columns `[energy_keV, pol_fraction_pct, pol_angle_deg]`
and `N_energy == len(ENERGY_BINS)` (50 by default, set in `src/utils/physics.py`).
