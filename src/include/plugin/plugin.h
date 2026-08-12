/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/plugin.h — 插件系统总入口头 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义插件类型枚举 ncclPluginType（Net/Gin/Rma/Tuner/Profiler/...）与
 * 插件加载入口，是各类插件聚合头的统一入口。mini-nccl 中多为 stub。
 */

#ifndef NCCL_PLUGIN_H_
#define NCCL_PLUGIN_H_

#include "nccl.h"

enum ncclPluginType {
  ncclPluginTypeNet,
  ncclPluginTypeGin,
  ncclPluginTypeRma,
  ncclPluginTypeTuner,
  ncclPluginTypeProfiler,
  ncclPluginTypeEnv,
};

void* ncclOpenNetPluginLib(const char* name);
void* ncclOpenGinPluginLib(const char* name);
void* ncclOpenRmaPluginLib(const char* name);
void* ncclOpenTunerPluginLib(const char* name);
void* ncclOpenProfilerPluginLib(const char* name);
void* ncclOpenEnvPluginLib(const char* name);
void* ncclGetNetPluginLib(enum ncclPluginType type);
void* ncclGetGinPluginLib(enum ncclPluginType type);
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type);

extern char* ncclPluginLibPaths[];
const char* ncclGetPluginLibName(enum ncclPluginType type);

#endif
