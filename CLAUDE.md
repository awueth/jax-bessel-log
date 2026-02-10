# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUSF (CUDA Special Functions) is a math library providing GPU-accelerated implementations of mathematical special functions. The library has two main backends:
- **cusf/cusf/**: Standalone CUDA C++ library (requires Linux, CUDA >= 11.0)
- **cusf/torch/**: PyTorch extension wrapping the CUDA kernels
- **cusf/autograd/**: Pure Python autograd implementations using PyTorch for automatic differentiation

## Build Commands

### Building the CUDA library (Linux only)
```bash
cd cusf/cusf
make lib                    # Build libcusf.a static library
make tests                  # Build and run CUDA tests
make TESTS=*bessel_iv.cu tests  # Run specific test
```

### Building the PyTorch extension
```bash
cd cusf/torch
make lib                    # Build and install cusf Python package
make tests                  # Run Python tests
```

### Full build from root
```bash
cd cusf
make all                    # Build cusf library then torch extension
make tests                  # Build and run all tests
```

## Architecture

### CUDA Kernel Structure (`cusf/cusf/`)
- `src/bessel/` - Bessel functions: iv, jv, kv, yv (and variants like iv_log, kv_log)
- `src/hypergeometric/` - Hypergeometric functions: hyp1f1, hyp2f1
- `src/gamma/` - Gamma and digamma functions
- `src/math/` - Error function (erf)
- `src/faddeeva/` - Faddeeva function
- `include/` - Header files mirroring src/ structure
- `tests/` - CUDA test files (test_*.cu)

### PyTorch Bindings (`cusf/torch/`)
- `src/` - C++ wrapper files that call CUSF kernels and expose to Python
- `include/` - Header files for the wrappers
- `setup.py` - PyTorch CUDAExtension build configuration
- `tests/` - Python test files

### Autograd Implementations (`cusf/autograd/`)
- `autograd/bessel/` - PyTorch autograd.Function classes for Bessel functions with custom backward passes
- `test_*.py` - Tests comparing autograd implementations against scipy

## Testing

```bash
# CUDA tests (in cusf/cusf/)
make TESTS=*bessel_iv.cu run     # Single test
make TESTTOL=1e-5 tests          # Set tolerance

# Python tests (in cusf/torch/tests/)
python test_bessel.py
python test_iv_log.py
```

## Key Dependencies

- CUDA >= 11.0 with CUDA_HOME environment variable set
- PyTorch with CUDA support
- GSL (GNU Scientific Library) for CUDA tests
- scipy for reference implementations in Python tests
