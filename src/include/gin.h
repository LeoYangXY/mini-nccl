/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/gin.h — [第三方] GIN(GPU 内部接口库)声明
 * ----------------------------------------------------------------------------
 * GIN 是源自 Meta 的 GPU 内部接口库（BSD-3），用于探测 GPU 内部拓扑/属性。NCCL
 * 通过它获取底层硬件信息。第三方代码，非 mini-nccl 自有逻辑，不建议改动。
 */

#ifndef NCCL_INT_GIN_H_
#define NCCL_INT_GIN_H_

#include "nccl_gin.h"

ncclResult_t ncclGinInit(struct ncclComm* comm);
ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclGinFinalize(struct ncclComm* comm);

extern ncclGin_t ncclGinIbGdaki;

#endif
