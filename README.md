# GRAND Decoder: GPU-Accelerated Decoding for Quantum Error Correction

A CUDA implementation of GRAND (Guessing Random Additive Noise Decoding) for
decoding quantum error-correcting codes. The library exposes a C API,
compiled into a shared library, that can be called from C/C++ or from Python
(via `ctypes`).

It supports two input representations:

- **CSS-style parity-check matrices** (`Hx` / `Hz`) — for stabilizer/CSS codes.
- **Detector-error-model (DEM) style matrices** (`H_dem` / `L_dem`) — the
  representation used by tools such as [Stim](https://github.com/quantumlib/Stim).

Decoding runs as a batch over `num_shots` syndromes/detector outcomes per
call, with the combinatorial search parallelized across CUDA threads.

## Features

- Exhaustive combinatorial GRAND search up to a configurable maximum error
  weight (`GRAND_MAX_W`).
- CPU, GPU, and hybrid CPU/GPU search modes, selectable per call via
  `GrandDecoderConfig.mode_of_search`.
- Optional probability-weighted search order, using per-position/per-mechanism
  error probabilities loaded from file.
- Optional maximum-likelihood (LLR-based) selection among matching error
  patterns.
- Per-shot timing and search statistics (`GrandDecoderResult`), useful for
  benchmarking.

## Requirements

- NVIDIA CUDA Toolkit (tested with `nvcc`, C++17 support — adjust the
  `-std=` flag in the `makefile` if your toolkit is older).
- A CUDA-capable GPU.
- Python 3 (only if you intend to use the library from Python, as in the
  `Examples/` folder). No extra Python packages are required beyond the
  standard library (`ctypes`) unless a given example imports something else —
  check the header of each example script.

## Repository structure

```
.
├── grand_decoder_final.cu   # Implementation (CUDA/C++)
├── grand_decoder_final.h    # Public C API (documented with Doxygen-style comments)
├── makefile                 # Build script
├── Examples/                 # Usage examples for all 5 public functions
├── H_Matrix/                 # Sample CSS parity-check matrices (Hx/Hz)
├── H_Matrix_DEM/              # Sample DEM parity-check matrices
├── L_Matrix_DEM/              # Sample logical-observable matrices
├── Syndromes/                 # Sample syndrome data (CSS input format)
├── Syndromes_DEM/              # Sample detector data (DEM input format)
├── Create_Files/               # Scripts to generate matrices/syndromes
├── Errors/                     # Sample/generated error data
├── CITATION.cff
└── LICENSE
```

## Building

```bash
make
```

This produces `libgranddecoder.so`, a shared library exposing the C API
declared in `grand_decoder_final.h`.

To remove build artifacts:

```bash
make clean
```

## API overview

The library exposes five public functions, declared in
`grand_decoder_final.h` (see that file for full parameter documentation):

| Function | Input format | Search order |
|---|---|---|
| `grand_decode_cc` | CSS (`Hx`/`Hz`) | uniform |
| `grand_decode_prob_cc` | CSS (`Hx`/`Hz`) | probability-weighted |
| `grand_decode_cln` | DEM (`H_dem`/`L_dem`) | uniform |
| `grand_decode_prob_cln` | DEM (`H_dem`/`L_dem`) | probability-weighted |
| `grand_decode_ML_cln` | DEM (`H_dem`/`L_dem`) | maximum-likelihood (LLR-based) selection |

All functions share the same overall calling convention: a parity-check
matrix (and, for the `cln` family, a logical-observable matrix), a batch of
input syndromes/detectors, a `GrandDecoderConfig` describing the search
strategy, and output buffers for the decoded errors/logicals and per-shot
`GrandDecoderResult` statistics.

## Usage example (Python via ctypes)

The snippet below sketches how the shared library is loaded and called from
Python. See `Examples/` for complete, runnable scripts covering all five
functions.

```python
import ctypes

lib = ctypes.CDLL("./libgranddecoder.so")

config = GrandDecoderConfig(
    max_error_weight=4,
    threads_per_block=256,
    use_gpu_sum=1,
    max_size_active_positions=64,
    threshold_for_hybrid=100000,
    mode_of_search=1,  # hybrid
)

# ... load Hx, Hz (or H_dem, L_dem) and the syndromes/detectors as ctypes arrays ...

lib.grand_decode_cc(Hx, Hz, n, m, errors, syndromes, num_shots,
                     ctypes.byref(config), results, ord('x'))
```

Refer to `Examples/README.md` for a description of each example script.

## Citing

If you use this software in academic work, please cite it using the metadata
in [`CITATION.cff`](./CITATION.cff).

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Author

Maria Carolina Rodrigues Boavida Fernandes — University of Coimbra
