"""
Basic unit tests for GPU Performance Estimator
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_performance_estimator import (
    validate_precision,
    get_precision_dtype,
    PRECISION_MULTIPLIER,
    CORES_PER_SM_DICT,
    TENSOR_CORE_THROUGHPUT,
)


class TestPrecisionValidation:
    """Test precision validation functions"""
    
    def test_valid_precisions(self):
        """Test that all supported precisions are valid"""
        valid_precisions = ['fp64', 'fp32', 'fp16', 'bf16', 'int8', 'int4']
        for precision in valid_precisions:
            assert validate_precision(precision), f"{precision} should be valid"
    
    def test_invalid_precisions(self):
        """Test that invalid precisions are rejected"""
        invalid_precisions = ['fp128', 'float32', 'int16', '']
        for precision in invalid_precisions:
            assert not validate_precision(precision), f"{precision} should be invalid"


class TestPrecisionDtype:
    """Test precision to dtype mapping"""
    
    def test_dtype_mapping(self):
        """Test that precisions map to correct dtypes"""
        import torch
        
        mappings = {
            'fp64': torch.float64,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'int8': torch.int8,
            'int4': torch.int8,  # Uses int8 as fallback
        }
        
        for precision, expected_dtype in mappings.items():
            dtype = get_precision_dtype(precision)
            assert dtype == expected_dtype, f"{precision} should map to {expected_dtype}"


class TestArchitectureData:
    """Test architecture specification data"""
    
    def test_cores_per_sm_dict(self):
        """Test that CORES_PER_SM_DICT has expected entries"""
        # Check some known architectures
        assert (8, 6) in CORES_PER_SM_DICT, "RTX 3090 (8.6) should be in dict"
        assert (7, 5) in CORES_PER_SM_DICT, "RTX 2080 (7.5) should be in dict"
        assert CORES_PER_SM_DICT[(8, 6)] == 128, "Ampere 8.6 should have 128 cores/SM"
    
    def test_tensor_core_throughput(self):
        """Test that TENSOR_CORE_THROUGHPUT has expected entries"""
        # Check that newer architectures are included
        assert (9, 0) in TENSOR_CORE_THROUGHPUT, "Hopper (9.0) should be in dict"
        assert (8, 6) in TENSOR_CORE_THROUGHPUT, "Ampere (8.6) should be in dict"
        
        # Check that fp16 has higher throughput than fp32
        ampere_throughput = TENSOR_CORE_THROUGHPUT[(8, 6)]
        assert ampere_throughput.get('fp16', 0) > ampere_throughput.get('fp32', 0), \
            "fp16 should have higher throughput than fp32"


class TestPrecisionMultiplier:
    """Test precision multiplier values"""
    
    def test_multiplier_values(self):
        """Test that precision multipliers are correct"""
        # Lower precision = higher multiplier (more ops per cycle)
        assert PRECISION_MULTIPLIER['fp64'] == 1
        assert PRECISION_MULTIPLIER['fp32'] == 1
        assert PRECISION_MULTIPLIER['fp16'] == 2
        assert PRECISION_MULTIPLIER['bf16'] == 2
        assert PRECISION_MULTIPLIER['int8'] == 4
        assert PRECISION_MULTIPLIER['int4'] == 8
        
        # Verify ordering
        assert PRECISION_MULTIPLIER['int4'] > PRECISION_MULTIPLIER['int8'] > \
               PRECISION_MULTIPLIER['fp16'] > PRECISION_MULTIPLIER['fp32']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
