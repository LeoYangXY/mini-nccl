/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/dev_runtime_internal.h — 设备运行时(device runtime)内部私有定义
 * ----------------------------------------------------------------------------
 * 本头文件是 dev_runtime.cc / dev_runtime_segments.cc 的“内部实现头”，不对外暴露。
 * 主要内容：
 *   1. ncclDevrMemory —— 描述一块“已向设备运行时注册的内存”的完整类型。
 *      在公共头 src/include/dev_runtime.h 中它只是一个前向声明(不完全类型)，
 *      真正的字段布局放在这里，从而把实现细节对上层调用者隐藏起来。
 *   2. ncclDevrGinSegmentInfo —— GIN(GPU Initiated Networking)按段(segment)注册
 *      所需要的窗口(window)句柄集合。
 *   3. 一组内部辅助函数声明：物理段大小探测、注册能力检查、段布局校验、
 *      GIN 段信息构建、段窗口(segment window)分配与更新等。
 *
 * 背景概念：
 *   - 一块用户 buffer 在底层可能由多个“物理段(segment)”拼接而成(尤其是 VMM 分配)，
 *     每个段可能位于显存(DEVICE)或主机内存(HOST_NUMA)，注册时需要逐段处理。
 *   - LSA(Local Scope Alias) 团队指的是同节点内可直接互访显存的一组 rank。
 */

#ifndef NCCL_DEVICE_RUNTIME_INTERNAL_H_
#define NCCL_DEVICE_RUNTIME_INTERNAL_H_

#include "dev_runtime.h"
#include "nccl_device/core.h"
#include <cuda.h>
#include <cuda_runtime.h>

struct ncclComm;
struct ncclSegmentWindow;
struct ncclWindow_vidmem;

struct ncclDevrGinSegmentInfo {
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS];      // 主机侧 GIN 窗口对象指针，每条 GIN 连接一个
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS]; // 设备侧 GIN 窗口句柄，供 kernel 直接使用
  CUmemLocationType memType;                        // 该段所在的内存位置类型(显存/主机内存)
  size_t segmentSize;                               // 该段的字节大小
};

// 与 src/include/dev_runtime.h 中前向声明相对应的完整类型定义。
struct ncclDevrMemory {
  int refCount;                                // 引用计数：同一块内存可被多次注册/共享
  struct ncclDevrMemory* next;                 // 链表指针：设备运行时用链表串起所有已注册内存
  CUmemGenericAllocationHandle* memHandles;    // 每个物理段对应的 CUDA VMM 分配句柄数组
  void* primaryAddr;                           // 主地址：本块内存第一个映射的虚拟地址(VA)
  size_t size;                                 // 本 rank 注册的总字节数
  size_t bigOffset;                            // 在“大 VA 空间”中的偏移量(统一虚拟地址布局用)
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]; // 整块内存级别的主机侧 GIN 窗口
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS]; // 整块内存级别的设备侧 GIN 窗口
  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]; // RMA(远程内存访问)主机侧窗口
  int winFlags;                                // 窗口标志位：记录该内存注册时启用了哪些能力
  // 以下字段由“本 rank 自己的分配”推导得到。
  int numSegments;         // 支撑本 rank buffer 的物理段数量
  bool hasSysmemSegment;   // 是否存在 CPU 端内存(HOST_NUMA)构成的段
  size_t* segmentSizes;    // 每个段的大小，数组长度为 numSegments
  // 以下字段是对全部 nRanks 做 bootstrapAllGather 之后得到的“全局聚合值”。
  int maxGlobalNumSegments;    // 通信域内所有 rank 的 numSegments 最大值
  bool globalHasSysmemSegment; // 通信域内只要有任一 rank 含主机内存段即为 true
  // 以下字段是 LSA 团队(同节点可直连的 rank 集合)范围内的聚合值，同样来自一次全局 allgather。
  int* lsaNumSegments;   // LSA 团队中每个 rank 的段数量，数组长度为 lsaSize
  size_t lsaMinSize; // LSA 团队内各 rank 注册大小的最小值
  size_t lsaMaxSize; // LSA 团队内各 rank 注册大小的最大值，供 ncclSpaceAlloc/Free 使用
  // GIN 注册状态。
  int numGinSegments;                              // GIN 窗口数量：除非 GIN 需要按段建多个窗口，否则为 1
  struct ncclDevrGinSegmentInfo* ginSegmentInfos;  // 每个 GIN 段的详细信息数组
};

// 探测并填充 mem->segmentSizes / numSegments：按 numSegments 个物理段逐段查询其大小。
ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory* mem, int numSegments);

// 检查给定用户指针与长度是否满足“可注册”的前提条件(例如是否含主机内存段、平台是否支持)。
ncclResult_t ncclDevrCheckRegistrationSupport(void* userPtr, size_t userSize, struct ncclComm* comm,
                                              bool hasSysmemSegment);

// 校验某个 VMM 分配句柄的内存位置类型是否合法(第 segment 个段)。
ncclResult_t ncclDevrValidateHandleLocationType(CUmemGenericAllocationHandle memHandle, int segment);

// 校验通信域内各 rank 的段布局是否一致/兼容，不一致时无法做对称内存访问。
ncclResult_t ncclDevrVerifySegmentLayouts(struct ncclDevrMemory* mem, struct ncclComm* comm);

// 根据物理段信息构建 GIN 段信息数组(ginSegmentInfos)。
ncclResult_t ncclDevrBuildGinSegmentInfos(struct ncclDevrMemory* mem);

// 在设备端分配并填充“段窗口(segment window)”数组，kernel 通过它把偏移换算到具体段。
ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                    cudaStream_t stream,
                                                    struct ncclSegmentWindow** outSegmentWindowsDev);

// 当段布局发生变化时，替换 window 对象中缓存的段窗口数组(必要时才做，避免无谓拷贝)。
ncclResult_t ncclDevrReplaceSegmentWindowsIfNeeded(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                   struct ncclWindow_vidmem* winHost, cudaStream_t stream);

#endif
