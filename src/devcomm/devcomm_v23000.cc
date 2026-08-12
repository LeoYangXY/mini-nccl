/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/devcomm/devcomm_v23000.cc — 设备通信兼容层(CUDA v2.30.0)
 * ----------------------------------------------------------------------------
 * 为 CUDA v2.30.0 提供对应的 ncclDevCommCompat 兼容表：声明该版本的设备结构布局、
 * 版本范围与各类过滤器/拷贝函数（多数为 nullptr，因精简版未启用复杂特性）。
 */

#include "dev_runtime.h"

struct ncclDevCommCompat ncclDevCommCompat_v23000 = {
  NCCL_VERSION(2, 30, 0),
  NCCL_VERSION_CODE, // minVersion, maxVersion
  nullptr,                                   // commPropertiesFilter
  nullptr,                                   // devCommRequirementsFilter
  nullptr,                                   // devCommCopyNewToOld
  nullptr,                                   // devCommCopyOldToNew
};
