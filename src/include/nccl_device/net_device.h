/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/net_device.h — 网络设备(device)接口
 * ----------------------------------------------------------------------------
 * 定义 NCCL 网络设备(device 侧)的抽象：ncclNetDevice*[ 等结构与版本/MTU 等常量，
 * 供网络传输在 device kernel 内直接收发数据。属 NVIDIA 官方设备 API 头。
 */

#ifndef NCCL_NET_DEVICE_H_
#define NCCL_NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION 0x0
#define NCCL_NET_MTU_SIZE 4096

// Arbitrary 版本 number - A 给定的 NCCL 构建 will 仅 be compatible with a 单个 设备 networking 插件
// 版本. NCCL will 检查 supplied 版本 number from 网络->getProperties() 并且 compare to its 内部 版本.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  NCCL_NET_DEVICE_HOST = 0,
  NCCL_NET_DEVICE_UNPACK = 1,
  NCCL_NET_DEVICE_GIN_PROXY = 2,
  NCCL_NET_DEVICE_GIN_GDAKI = 3,
  NCCL_NET_DEVICE_GIN_GPI = 4,
} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_v10_t;
typedef ncclNetDeviceHandle_v10_t ncclNetDeviceHandle_v11_t;
typedef ncclNetDeviceHandle_v11_t ncclNetDeviceHandle_v12_t;
typedef ncclNetDeviceHandle_v12_t ncclNetDeviceHandle_t;

#endif
