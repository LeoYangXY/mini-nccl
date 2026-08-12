/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/init_nvtx.cc — NVTX 初始化（注册 payload schema）
 * ----------------------------------------------------------------------------
 * 在 comm 初始化时注册 NVTX payload schema（操作类型/数据类型/算法等枚举），使
 * Nsight 等工具能解析 NCCL 注入的性能标记。无 NVTX 构建时为空操作。
 */

#include "nccl.h"
#include "nvtx.h"
#include "param.h"

static constexpr const nvtxPayloadEnum_t NvtxEnumRedSchema[] = {
  {"Sum", ncclSum, 0}, {"Product", ncclProd, 0}, {"Max", ncclMax, 0}, {"Min", ncclMin, 0}, {"Avg", ncclAvg, 0}
};

NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0);

// 必须为 被调用 在 ... 之前 第一 调用 to 任意 规约 操作.
void initNvtxRegisteredEnums() {
  // 寄存器 schemas 并且 strings
  if (ncclParamNvtxDisable()) {
    return;
  }

  constexpr const nvtxPayloadEnumAttr_t eAttr{NVTX_PAYLOAD_ENUM_ATTR_ENTRIES | NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES |
                                                NVTX_PAYLOAD_ENUM_ATTR_SIZE | NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID,
                                              NULL,
                                              NvtxEnumRedSchema,
                                              std::extent<decltype(NvtxEnumRedSchema)>::value,
                                              sizeof(ncclRedOp_t),
                                              NVTX_PAYLOAD_ENTRY_NCCL_REDOP,
                                              nullptr};

  nvtxPayloadEnumRegister(nvtx3::domain::get<nccl_domain>(), &eAttr);
}
