/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/device/common.cu — device kernel 公共定义与入口
 * ----------------------------------------------------------------------------
 * 定义各集合 kernel 共享的全局资源：ncclShmem（共享内存数据块）、ncclShmemPerWarp
 * 等，以及 RunWork* 系列入口函数（RunWorkSend/Recv/Reduce 等），是 kernel 启动的
 * 统一落脚点。被 all_reduce.h 等算法 kernel 包含。
 */

#include "device.h"
#include "collectives.h"
#include "common.h"
#include "nccl_device.h"
#include "comm.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
__shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS / WARP_SIZE) / sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}

__device__ void ncclDevFunc_Nop() {}
