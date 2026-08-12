/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/dev_runtime.h — device runtime(kernel 运行时/版本兼容)接口
 * ----------------------------------------------------------------------------
 * 定义 device 端 runtime 抽象：处理不同 CUDA 版本下 kernel 启动与设备函数的兼容，
 * 使 NCCL 能在多种 CUDA toolkit 上编译运行（devcomm/ 提供各版本实现）。
 */

#ifndef NCCL_DEVICE_RUNTIME_H_
#define NCCL_DEVICE_RUNTIME_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// ncclDevr[_]：对称 API 的运行时实现。

struct ncclDevrMemory;
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset; // Offset in big VA space.
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem; // key for intrusive map
  struct ncclDevrWindow* next; // next for intrusive map
  struct ncclComm* comm; // comm for intrusive map window <> comm look up
};
struct ncclDevrWindowSorted;
struct ncclDevrTeam;

struct ncclDevrRegTask {
  struct ncclDevrRegTask* next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;
};

struct ncclDevrCommCreateTask {
  struct ncclDevrCommCreateTask* next;
  struct ncclDevCommRequirements* reqs;
  struct ncclDevComm* outDevComm;
  struct ncclDevCommCompat* devCompat;
};

struct ncclDevrState {
  // Like localRank/localRanks except "lsa" ranks 必须为 consecutive 在 ... 中 world
  // 并且 所有 lsa subsets have 相同 数量： ranks. 如果有的话 condition is
  // 假 那么 lsa team is 仅 the singleton of 自身.
  int lsaSelf;
  int lsaSize;
  int* lsaRankList;
  int nLsaTeams;

  size_t granularity; // cuMemGetAllocationGranularity
  bool ginEnabled;
  bool rmaProxyEnabled;
  struct ncclDevrMemory* memHead;
  struct ncclDevrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  struct ncclDevrTeam* teamHead;
  size_t bigSize; // size of our big logical space (128GB?)
  struct ncclSpace bigSpace; // allocates our big VA space.
  void* lsaFlatBase; // base ptr for all lsa ranks big VA's concatenated together: size = lsaRanks*bigSize
  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
  struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};

struct ncclDevCommCompat {
  int minVersion, maxVersion;
  ncclResult_t (*commPropertiesFilter)(ncclComm_t comm, struct ncclCommProperties* props);
  ncclResult_t (*devCommRequirementsFilter)(ncclComm_t comm, ncclDevCommRequirements_t* reqs);
  ncclResult_t (*devCommCopyNewToOld)(ncclComm_t comm, void* oldDevComm, struct ncclDevComm const* newDevComm);
  ncclResult_t (*devCommCopyOldToNew)(ncclComm_t comm, struct ncclDevComm* newDevComm, void const* oldDevComm);
};

// 检查 若 GIN resources 已经 requested as part of `reqs`.
bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs);

// 检查 若re is 仅 one LSA team. 该函数 使用 the cached 值 of 通信域 或者 computes the
// 值 从 通信域 拓扑.
bool ncclDevrIsOneLsaTeam(struct ncclComm* comm);

// We 假设 ncclComm has a `ncclDevrState symState` 成员.
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm);
ncclResult_t ncclDevrFinalize(struct ncclComm* comm);

// 若 已找到 *outWinHost 将会 populated 并且 *outWinId >= 0, 否则 *outWinId == -1
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* ptr, size_t size, int winFlags,
                                           ncclWindow_t* outWinDev);

ncclResult_t ncclDevrCommCreateInternal(struct ncclComm* comm, struct ncclDevCommRequirements* reqs,
                                        struct ncclDevComm* outDevComm, bool isInternal = false,
                                        struct ncclDevCommCompat* devCompat = nullptr);
void freeDevCommRequirements(struct ncclDevCommRequirements* reqs);

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win);
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win);

// 获取 corresponding 指针 入 另一个 lsa rank's symmetric 内存 window
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank,
                                   void** outPtr);

// 将 ... 转换 world rank to an LSA rank.
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm* comm, int peerWorldRank, int* peerLsaRank);

// 获取 RMA window 句柄 for a 特定的 上下文
void* ncclDevrGetRmaWin(struct ncclDevrWindow* winHost, int ctx);

// 获取 multicast 地址 for a 给定的 team
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset,
                                     struct ncclTeam lsaTeam, void** outPtr);

// 拷贝 the devComm 数据 from "rank" to "lsaBarrier".  Assumes 相同 内存 布局 at 源文件 并且 目标.
void ncclDevCommCopyLsaData(void* dstRankPtr, void const* srcRankPtr);
#endif
