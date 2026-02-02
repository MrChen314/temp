"""
Setup script for test_2sm_mma_cuda extension.
Compiles the CUDA kernel for testing utcmma_ss on SM100 (Blackwell).
"""

import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前目录和 FlashMLA 目录
this_dir = Path(__file__).parent.absolute()
flash_mla_dir = this_dir.parent.parent / "FlashMLA"

# 检查 FlashMLA 目录存在
assert flash_mla_dir.exists(), f"FlashMLA directory not found: {flash_mla_dir}"

# NVCC 编译参数
nvcc_flags = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-D_USE_MATH_DEFINES",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo",
    # SM100 架构
    "-gencode=arch=compute_100a,code=sm_100a",
    # 并行编译线程数
    "--threads", os.getenv("NVCC_THREADS", "32"),
]

# C++ 编译参数
cxx_flags = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-Wno-deprecated-declarations",
]

# Include 目录
include_dirs = [
    str(flash_mla_dir / "csrc"),
    str(flash_mla_dir / "csrc" / "kerutils" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "include"),
    str(flash_mla_dir / "csrc" / "cutlass" / "tools" / "util" / "include"),
]

# 验证 include 目录存在
for inc_dir in include_dirs:
    if not Path(inc_dir).exists():
        print(f"Warning: include directory not found: {inc_dir}")

ext_modules = [
    CUDAExtension(
        name="test_2sm_mma_cuda",
        sources=["test_2sm_mma.cu"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=include_dirs,
    )
]

setup(
    name="test_2sm_mma_cuda",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
