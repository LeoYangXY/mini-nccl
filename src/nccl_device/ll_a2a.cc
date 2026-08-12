/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/nccl_device/ll_a2a.cc — 设备端 LL all-to-all 辅助实现
 * ----------------------------------------------------------------------------
 * 实现 Low-Latency(LL) 协议的 all-to-all 辅助计算：ncclLLA2ACalcSlots 等计算 LL
 * 协议所需的 slot 数量，供 LL/LL128 协议的 device kernel 使用。
 */

#include "core.h"
#include "nccl_device/impl/ll_a2a__funcs.h"

NCCL_API(int, ncclLLA2ACalcSlots, int maxElts, int maxEltSize);
int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) {
  return maxElts * divUp(maxEltSize, 8);
}

NCCL_API(ncclResult_t, ncclLLA2ACreateRequirement, int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle,
         ncclDevResourceRequirements_t* outReq);
ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle,
                                        ncclDevResourceRequirements_t* outReq) {
  outHandle->nSlots = nSlots;
  memset(outReq, 0, sizeof(*outReq));
  outReq->bufferSize = nBlocks * (1 + 2 * nSlots) * 16;
  outReq->bufferAlign = 16;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
