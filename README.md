# GPU Performance Estimator

[![CI](https://github.com/cvlt-ai/gpu-performance-estimator/actions/workflows/ci.yml/badge.svg)](https://github.com/cvlt-ai/gpu-performance-estimator/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/gpu-performance-estimator.svg)](https://pypi.org/project/gpu-performance-estimator/)

A sophisticated tool for measuring GPU performance across different precision types, including separate measurements for raw CUDA core performance and tensor core performance.

## Features

- **Dual Performance Measurement**: Measures both raw CUDA core performance (without tensor cores) and tensor core-accelerated performance
- **Extended Precision Support**: Test with fp64, fp32, fp16, bf16, int8, and int4 precision types
- **Multi-GPU Support**: Test multiple GPUs simultaneously
- **Accurate FLOPs Calculation**: Proper accounting for different operation types and precision modes
- **Hardware Detection**: Automatically detects GPU compute capability, CUDA core count, tensor core count, and clock rate
- **Configurable Tests**: Customize tensor sizes and test durations
- **Cross-Platform**: Works on both Linux and Windows

## File Structure

```
.
├── README.md
├── requirements.txt
└── src
    └── gpu_performance_estimator.py
```

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA support

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Basic Usage

Test a single GPU with default settings (fp32 precision, 1024x1024x1024 tensors):

```
python src/gpu_performance_estimator.py --gpus 0
```

### Test Multiple GPUs

```
python src/gpu_performance_estimator.py --gpus 0 1 2
```

### Test Specific Precisions

Test with floating point precisions (shows both raw and tensor core performance):

```
python src/gpu_performance_estimator.py --gpus 0 --precisions fp32 fp16 bf16
```

Test with quantized precisions:

```
python src/gpu_performance_estimator.py --gpus 0 --precisions int8 int4
```

### Custom Tensor Size

Adjust tensor dimensions for different workload characteristics:

```
python src/gpu_performance_estimator.py --gpus 0 --tensor-size 2048 2048 2048
```

### Longer Test Duration

Increase test duration for more stable measurements:

```
python src/gpu_performance_estimator.py --gpus 0 --min-time 5.0
```

## Understanding Results

The tool outputs performance in GFLOPS (Giga Floating Point Operations Per Second). For floating point types, you'll see two separate measurements:

- **Raw (CUDA cores)**: Performance using traditional CUDA cores without tensor core acceleration
- **Tensor cores**: Performance with tensor core acceleration (when available and enabled)

For quantized types (int8, int4), only a single measurement is provided as these use different hardware pathways.

### Example Output

```
=== Testing GPU 0 ===
  Name: NVIDIA RTX 3090
  Compute Capability: 8.6
  SM Count: 82
  Estimated CUDA Cores: 10496
  Tensor Cores: 10482
  Total Memory: 23.77 GB

  Testing fp32 (raw CUDA cores)... 1250.45 GFLOPS
  Testing fp32 (tensor cores)... 18500.32 GFLOPS
  Testing fp16 (raw CUDA cores)... 1100.21 GFLOPS
  Testing fp16 (tensor cores)... 22000.15 GFLOPS
  Testing int8... 8500.42 GFLOPS

=== Performance Summary ===
GPU_0:
  fp32:
    Raw (CUDA cores): 1250.45 GFLOPS
    Tensor cores: 18500.32 GFLOPS
  fp16:
    Raw (CUDA cores): 1100.21 GFLOPS
    Tensor cores: 22000.15 GFLOPS
  int8: 8500.42 GFLOPS
```

## Technical Details

### Tensor Core Measurement

- **Raw CUDA Core Mode**: TF32 is disabled for fp32 tests, ensuring measurements use only traditional CUDA cores
- **Tensor Core Mode**: Autocast is enabled with appropriate precision settings to maximize tensor core utilization on supported architectures (Volta+, Turing, Ampere, Hopper)

### Precision Support

- **fp64**: 64-bit floating point (double precision)
- **fp32**: 32-bit floating point (single precision)
- **fp16**: 16-bit floating point (half precision)
- **bf16**: 16-bit brain floating point
- **int8**: 8-bit integer (simulated using PyTorch operations)
- **int4**: 4-bit integer (simulated using packed int8 operations)

> **Note**: int4 and int8 measurements are approximations as they use simulated operations rather than dedicated quantized kernels. For production quantized inference benchmarks, use dedicated tools like NVIDIA's TensorRT.

### FLOPs Calculation

For matrix multiplication operations (M×K × K×N):

```
Total FLOPs = 2 × M × K × N × iterations
```

For quantized types, operation counts are adjusted to reflect the effective computational density of the precision.

### Hardware Detection

The tool automatically detects:

- GPU name and compute capability
- Number of SMs (Streaming Multiprocessors)
- Estimated CUDA core count (based on architecture)
- Estimated tensor core count (based on architecture)
- Clock rate (when available)
- Total memory

## Performance Considerations

- **Warm-up Runs**: The script performs warm-up iterations to stabilize GPU clock speeds and ensure consistent results
- **Dynamic Iteration Count**: Automatically adjusts iteration count to achieve a minimum test duration for accurate timing
- **Memory Bandwidth vs Compute**: Performance is influenced by both compute capability and memory bandwidth; use larger tensor sizes to stress compute more than memory
- **Architecture Differences**: Tensor core performance varies significantly between architectures (Volta, Turing, Ampere, Hopper)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce tensor size with `--tensor-size`
2. **Unsupported precision**: Ensure your PyTorch version supports the requested precision
3. **Slow performance**: Check GPU utilization with `nvidia-smi`; background processes may affect results
4. **No tensor core acceleration**: Verify your GPU architecture supports tensor cores (compute capability 7.0+)

### Verification

To verify correct operation, check that:

- Raw CUDA core performance is lower than tensor core performance for fp16/bf16 on supported GPUs
- Results are consistent across multiple runs
- GPU memory usage stays within limits

## Limitations

- **int4/int8**: These are simulated using element-wise operations, not true matrix multiplication with quantized kernels
- **Dynamic Clocks**: GPU boost clocks may vary based on temperature and power limits
- **System Load**: Other processes may affect timing measurements
- **Driver Version**: Performance can vary with different CUDA driver versions

## Future Enhancements

Potential improvements:

- Use actual quantized matmul operations with PyTorch's quantized module
- Add support for additional operations (convolution, attention, etc.)
- Memory bandwidth measurement
- Export results to CSV/JSON
- Performance visualization
- Support for more GPU architectures

## Version History

- **v2.0.0**: Separated raw vs tensor core performance, added int8/int4 support, enhanced hardware detection
- **v1.2.0**: Added BF16 support and improved documentation
- **v1.1.0**: Multi-GPU support and enhanced error handling
- **v1.0.0**: Initial release

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the repository.

## Acknowledgements

This tool was developed to help researchers and developers evaluate GPU performance across different precision types.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CUDA Performance Best Practices](https://developer.nvidia.com/blog/cuda-best-practices/)
- [NVIDIA Tensor Cores](https://developer.nvidia.com/tensor-cores)

## Support

This tool is maintained and tested on:

- Ubuntu 20.04 LTS
- Windows 10/11
- NVIDIA RTX 30xx series GPUs
- NVIDIA A100 and H100 GPUs
- PyTorch 1.8+

---

**Disclaimer**: The performance measurements provided are estimates and may vary based on system configuration, driver versions, and other factors. This tool should be used as a reference for performance evaluation rather than as a definitive measure of GPU capabilities.