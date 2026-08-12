/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/tuner.h — 调优器(tuner)接口声明
 * ----------------------------------------------------------------------------
 * 声明 NCCL tuner 插件接口：按通信大小/拓扑动态选择最优算法与协议，替代静态
 * 调优表(ncclTopoTuneModel)。实现以插件形式注册，运行时提供建议。
 */

#ifndef NCCL_INT_TUNER_H_
#define NCCL_INT_TUNER_H_

#include "nccl_tuner.h"
#include "comm.h"

// Tuning 插件 to override NCCL's 默认 算法/protocol tuning.

// Attempts to 加载 NCCL tuner from environmental 变量.
// 返回 ncclSuccess 若 正确 tuner symbol 已经 已找到 并且
// successully loaded.  否则 返回 an 错误 并且 也 日志 the 错误.
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm);

// Cleans up NCCL tuner 插件.
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm);
#endif
