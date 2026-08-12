/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/mnnvl.h — 多节点 NVLink(MNNVL)声明
 * ----------------------------------------------------------------------------
 * 声明跨节点 NVLink（Multi-Node NVLink）相关的检测与结构体：当多个节点通过 NVLink
 * 桥接互联时，NCCL 可将其视为“超节点”统一调度。mini-nccl 多为单节点，接口保留。
 */

#ifndef NCCL_MNNVL_H_
#define NCCL_MNNVL_H_

#include "nccl.h"
#include "comm.h"

ncclResult_t ncclMnnvlCheck(struct ncclComm* comm);

#endif
