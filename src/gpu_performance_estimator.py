import argparse
import time
import torch
import subprocess
import json
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress PyTorch warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# ARCHITECTURE SPECIFICATIONS
# ============================================================================

# CUDA Cores per Streaming Multiprocessor (SM) by compute capability
CORES_PER_SM_DICT = {
    # Blackwell (10.0)
    (10, 0): 128,
    # Hopper (9.0)
    (9, 0): 128,
    # Ampere (8.0, 8.6, 8.9)
    (8, 0): 64, (8, 6): 128, (8, 9): 128,
    # Turing (7.5)
    (7, 5): 64,
    # Volta (7.0)
    (7, 0): 64,
    # Pascal (6.0, 6.1, 6.2)
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    # Maxwell (5.0, 5.2, 5.3)
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    # Kepler (3.0, 3.2, 3.5, 3.7)
    (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
}

# Tensor Core throughput: operations per SM per cycle for each precision
# Format: (major, minor): {precision: ops_per_cycle}
TENSOR_CORE_THROUGHPUT = {
    # Blackwell (10.0) - 5th gen Tensor Cores
    (10, 0): {'fp8': 1024, 'fp64': 256, 'fp32': 256, 'fp16': 512, 'bf16': 512, 'int8': 1024, 'int4': 2048},
    # Hopper (9.0) - 4th gen Tensor Cores  
    (9, 0): {'fp8': 1024, 'fp64': 256, 'fp32': 256, 'fp16': 512, 'bf16': 512, 'int8': 1024, 'int4': 2048},
    # Ampere (8.0, 8.6) - 3rd gen Tensor Cores
    (8, 0): {'fp32': 128, 'fp16': 256, 'bf16': 256, 'int8': 512},
    (8, 6): {'fp32': 128, 'fp16': 256, 'bf16': 256, 'int8': 512},
    # Ampere Ada (8.9) - Enhanced 3rd gen
    (8, 9): {'fp32': 256, 'fp16': 512, 'bf16': 512, 'int8': 1024},
    # Turing (7.5) - 1st gen Tensor Cores
    (7, 5): {'fp32': 64, 'fp16': 128, 'bf16': 128},
    # Volta (7.0) - 1st gen Tensor Cores
    (7, 0): {'fp32': 64, 'fp16': 128},
}

# Precision multiplier for CUDA core operations (relative to fp32)
PRECISION_MULTIPLIER = {
    'fp64': 1,
    'fp32': 1,
    'fp16': 2,  # Two 16-bit ops per 32-bit cycle
    'bf16': 2,
    'int8': 4,  # Four 8-bit ops per 32-bit cycle
    'int4': 8,  # Eight 4-bit ops per 32-bit cycle
}


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    return torch.cuda.device_count()


def get_gpu_clock_rate(gpu_id: int) -> Optional[float]:
    """Get GPU clock rate in GHz using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.max.mem,clocks.max.sm', 
             '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                sm_clock_mhz = float(parts[1].strip())
                return sm_clock_mhz / 1000.0  # Convert to GHz
    except Exception:
        pass
    
    # Fallback to PyTorch properties
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        for attr in ['max_clock_rate', 'clock_rate']:
            if hasattr(props, attr):
                return getattr(props, attr) / 1000.0
    except Exception:
        pass
    
    return None  # Will use estimated value


def get_gpu_hardware_info(gpu_id: int) -> Dict[str, any]:
    """Retrieve GPU hardware info including estimated core counts and tensor core counts."""
    props = torch.cuda.get_device_properties(gpu_id)
    cc = (props.major, props.minor)
    sm_count = props.multi_processor_count
    
    # Get CUDA Cores
    cores_per_sm = CORES_PER_SM_DICT.get(cc, 128)
    total_cuda_cores = sm_count * cores_per_sm
    
    # Check if tensor cores are supported
    tensor_core_info = TENSOR_CORE_THROUGHPUT.get(cc, {})
    tensor_cores_supported = len(tensor_core_info) > 0
    
    # Get clock rate
    clock_rate_ghz = get_gpu_clock_rate(gpu_id)
    if clock_rate_ghz is None:
        # Estimate based on GPU name
        gpu_name = props.name.lower()
        if 'h100' in gpu_name:
            clock_rate_ghz = 1.65
        elif 'a100' in gpu_name:
            clock_rate_ghz = 1.41
        elif 'rtx 3090' in gpu_name:
            clock_rate_ghz = 1.53
        elif 'rtx 4090' in gpu_name:
            clock_rate_ghz = 2.52
        else:
            clock_rate_ghz = 1.5  # Default estimate
    
    return {
        "name": props.name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "sm_count": sm_count,
        "cuda_cores": total_cuda_cores,
        "tensor_cores_supported": tensor_cores_supported,
        "tensor_core_throughput": tensor_core_info,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "clock_rate_ghz": round(clock_rate_ghz, 3),
    }


def validate_precision(precision: str) -> bool:
    """Validate if the precision type is supported."""
    supported_precisions = ['fp64', 'fp32', 'fp16', 'bf16', 'int8', 'int4']
    return precision in supported_precisions


def get_precision_dtype(precision: str) -> torch.dtype:
    """Get the torch dtype for the given precision."""
    dtype_map = {
        'fp64': torch.float64,
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'int8': torch.int8,
        'int4': torch.int8,  # int4 not natively supported, will use packed int8 simulation
    }
    if precision not in dtype_map:
        raise ValueError(f"Precision {precision} mapping not found.")
    return dtype_map[precision]


def test_precision_performance(gpu_id: int, precision: str, tensor_size: Tuple[int, int, int],
                               min_test_time: float = 2.0, use_tensor_cores: bool = False) -> float:
    """
    Test the performance of a GPU in a specific precision.
    """
    device = torch.device(f'cuda:{gpu_id}')
    dtype = get_precision_dtype(precision)
    M, K, N = tensor_size
    
    # Align dimensions for tensor cores (multiples of 8 or 16)
    if use_tensor_cores:
        alignment = 16 if precision in ['fp16', 'bf16', 'int8'] else 8
        M = ((M + alignment - 1) // alignment) * alignment
        K = ((K + alignment - 1) // alignment) * alignment
        N = ((N + alignment - 1) // alignment) * alignment
    
    # Store original settings
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.enabled
    
    try:
        # Configure for tensor cores or raw compute
        if use_tensor_cores:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.enabled = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.enabled = False
        
        # Create tensors
        if precision in ['int8', 'int4']:
            # Use int8 tensors for integer operations
            a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
            b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        else:
            a = torch.randn((M, K), dtype=dtype, device=device)
            b = torch.randn((K, N), dtype=dtype, device=device)
        
        # Warm up
        for _ in range(50):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Find appropriate iteration count
        iters = 100
        while True:
            start = time.perf_counter()
            for _ in range(iters):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if elapsed > 0.1:
                break
            iters *= 2
        
        target_iters = max(1, int(iters * (min_test_time / elapsed)))
        
        # Actual benchmark
        start_time = time.perf_counter()
        for _ in range(target_iters):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Calculate FLOPs
        # Matrix multiplication: 2 * M * K * N operations per matmul
        total_ops = 2 * M * K * N * target_iters
        
        # Adjust for int4 (2x density)
        if precision == 'int4':
            total_ops *= 2
        
        gflops = total_ops / (total_time * 1e9)
        return gflops
        
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.enabled = original_cudnn


def calculate_theoretical_peak(gpu_id: int, precision: str, use_tensor_cores: bool = False) -> float:
    """
    Calculate theoretical peak performance for a GPU at a given precision.
    
    Returns theoretical peak in GFLOPS.
    """
    info = get_gpu_hardware_info(gpu_id)
    sm_count = info['sm_count']
    clock_rate_ghz = info['clock_rate_ghz']
    cc = (int(info['compute_capability'].split('.')[0]), 
          int(info['compute_capability'].split('.')[1]))
    
    if use_tensor_cores and info['tensor_cores_supported']:
        # Tensor core peak
        throughput = info['tensor_core_throughput'].get(precision, 0)
        if throughput > 0:
            # Peak = SMs * ops_per_cycle * clock_rate (GHz) = GFLOPS
            peak_gflops = sm_count * throughput * clock_rate_ghz
            return peak_gflops
        return 0.0
    else:
        # CUDA core peak
        cores_per_sm = CORES_PER_SM_DICT.get(cc, 128)
        total_cores = sm_count * cores_per_sm
        precision_mult = PRECISION_MULTIPLIER.get(precision, 1)
        
        # Peak = cores * 2 (MAD) * precision_mult * clock_rate (GHz) = GFLOPS
        peak_gflops = total_cores * 2 * precision_mult * clock_rate_ghz
        return peak_gflops


def run_performance_test(gpu_ids: List[int], precisions: List[str], 
                         tensor_size: Tuple[int, int, int],
                         min_test_time: float = 2.0) -> Dict:
    """Run performance tests on specified GPUs and precisions."""
    results = {}
    
    for gpu_id in gpu_ids:
        if gpu_id >= get_gpu_count():
            print(f"GPU {gpu_id} is not available. Skipping.")
            continue
            
        print(f"\n=== Testing GPU {gpu_id} ===")
        
        try:
            info = get_gpu_hardware_info(gpu_id)
            print(f"  Name: {info['name']}")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  SM Count: {info['sm_count']}")
            print(f"  CUDA Cores: {info['cuda_cores']}")
            print(f"  Tensor Cores: {'Supported' if info['tensor_cores_supported'] else 'Not supported'}")
            print(f"  Clock Rate: {info['clock_rate_ghz']} GHz")
            print(f"  Total Memory: {info['total_memory_gb']} GB")
        except Exception as e:
            print(f"  Could not retrieve hardware info: {e}")
            continue

        results[f'GPU_{gpu_id}'] = {'info': info, 'measurements': {}}
        
        for precision in precisions:
            if not validate_precision(precision):
                print(f"  Precision {precision} is not supported. Skipping.")
                continue
            
            # For floating point types, test both raw and tensor core performance
            if precision in ['fp64', 'fp32', 'fp16', 'bf16']:
                # Raw CUDA core performance
                try:
                    print(f"  Testing {precision} (raw CUDA cores)...", end=' ', flush=True)
                    gflops_raw = test_precision_performance(gpu_id, precision, tensor_size, 
                                                            min_test_time, use_tensor_cores=False)
                    theoretical_raw = calculate_theoretical_peak(gpu_id, precision, use_tensor_cores=False)
                    efficiency_raw = (gflops_raw / theoretical_raw * 100) if theoretical_raw > 0 else 0
                    results[f'GPU_{gpu_id}']['measurements'][f'{precision}_raw'] = {
                        'gflops': gflops_raw,
                        'theoretical': theoretical_raw,
                        'efficiency': efficiency_raw
                    }
                    print(f"{gflops_raw:.1f} GFLOPS ({efficiency_raw:.1f}% of {theoretical_raw:.0f}G theoretical)")
                except Exception as e:
                    print(f"Error: {e}")
                
                # Tensor core performance (if supported)
                if info['tensor_cores_supported']:
                    try:
                        print(f"  Testing {precision} (tensor cores)...", end=' ', flush=True)
                        gflops_tensor = test_precision_performance(gpu_id, precision, tensor_size,
                                                                  min_test_time, use_tensor_cores=True)
                        theoretical_tensor = calculate_theoretical_peak(gpu_id, precision, use_tensor_cores=True)
                        efficiency_tensor = (gflops_tensor / theoretical_tensor * 100) if theoretical_tensor > 0 else 0
                        results[f'GPU_{gpu_id}']['measurements'][f'{precision}_tensor'] = {
                            'gflops': gflops_tensor,
                            'theoretical': theoretical_tensor,
                            'efficiency': efficiency_tensor
                        }
                        print(f"{gflops_tensor:.1f} GFLOPS ({efficiency_tensor:.1f}% of {theoretical_tensor:.0f}G theoretical)")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # For quantized types
            elif precision in ['int8', 'int4']:
                try:
                    print(f"  Testing {precision}...", end=' ', flush=True)
                    gflops = test_precision_performance(gpu_id, precision, tensor_size,
                                                        min_test_time, use_tensor_cores=False)
                    theoretical = calculate_theoretical_peak(gpu_id, precision, use_tensor_cores=False)
                    efficiency = (gflops / theoretical * 100) if theoretical > 0 else 0
                    results[f'GPU_{gpu_id}']['measurements'][precision] = {
                        'gflops': gflops,
                        'theoretical': theoretical,
                        'efficiency': efficiency
                    }
                    print(f"{gflops:.1f} GFLOPS ({efficiency:.1f}% of {theoretical:.0f}G theoretical)")
                except Exception as e:
                    print(f"Error: {e}")
      
    return results


def print_performance_summary(results: Dict):
    """Print a comprehensive performance summary."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    for gpu_key, gpu_data in results.items():
        info = gpu_data.get('info', {})
        measurements = gpu_data.get('measurements', {})
        
        print(f"\n{gpu_key}: {info.get('name', 'Unknown')}")
        print("-" * 50)
        
        # Group by precision
        precision_groups = {}
        for key, data in measurements.items():
            if '_' in key:
                base_precision, mode = key.rsplit('_', 1)
            else:
                base_precision = key
                mode = 'raw'
            
            if base_precision not in precision_groups:
                precision_groups[base_precision] = {}
            precision_groups[base_precision][mode] = data
        
        for precision, modes in sorted(precision_groups.items()):
            print(f"  {precision.upper()}:")
            if 'raw' in modes:
                m = modes['raw']
                print(f"    CUDA cores:   {m['gflops']:8.1f} GFLOPS ({m['efficiency']:5.1f}% of {m['theoretical']:8.0f}G peak)")
            if 'tensor' in modes:
                m = modes['tensor']
                print(f"    Tensor cores: {m['gflops']:8.1f} GFLOPS ({m['efficiency']:5.1f}% of {m['theoretical']:8.0f}G peak)")
            if precision in modes and 'raw' not in modes:
                m = modes[precision]
                print(f"    {m['gflops']:8.1f} GFLOPS ({m['efficiency']:5.1f}% of {m['theoretical']:8.0f}G peak)")


def main():
    parser = argparse.ArgumentParser(description="GPU Performance Estimator")
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU IDs to test')
    parser.add_argument('--precisions', nargs='+', type=str, default=['fp32'], 
                        help='Precision types (fp64, fp32, fp16, bf16, int8, int4)')
    parser.add_argument('--tensor-size', nargs=3, type=int, default=[1024, 1024, 1024], 
                        help='Tensor dimensions (M K N)')
    parser.add_argument('--min-time', type=float, default=2.0, 
                        help='Minimum test duration per precision (seconds)')
    
    args = parser.parse_args()
    
    if any(size <= 0 for size in args.tensor_size):
        raise ValueError("Tensor sizes must be positive integers.")
    
    available_gpus = get_gpu_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    
    if not args.gpus or any(gpu >= available_gpus for gpu in args.gpus):
        print(f"Available GPUs: {available_gpus}")
        raise ValueError("Invalid GPU ID specified.")
    
    print(f"Running GPU Performance Estimator")
    print(f"GPUs: {args.gpus}")
    print(f"Precisions: {args.precisions}")
    print(f"Tensor size: {args.tensor_size}")
    print(f"Min test time: {args.min_time}s")
    
    results = run_performance_test(args.gpus, args.precisions, 
                                   tuple(args.tensor_size), args.min_time)
    
    print_performance_summary(results)


if __name__ == "__main__":
    main()
