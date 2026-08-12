/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/mem_manager.h — 内存管理器(mem manager)结构定义
 * ----------------------------------------------------------------------------
 * 定义 NCCL 的“用户注册内存管理”结构：把用户传入的 host/device buffer 登记为可
 * 被 transport 直接访问的段，管理其 IPC 句柄、引用计数与生命周期。
 */

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

#if CUDART_VERSION < 12030
// MNNVL: FABRIC 句柄 支持 lifted from CUDA 12.3
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE 64
#endif
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

struct ncclComm;

// 初始的 capacity for exported 对等端 数组
#define NCCL_MEM_EXPORT_PEERS_INIT 8

// 内存 类型 for NCCL 分配
typedef enum {
  ncclMemPersist = 0,  // Persistent memory - track stats only, never release/offload
  ncclMemScratch = 1,  // Free without saving
  ncclMemOffload = 2   // Copy to CPU before free, restore on resume
} ncclMemType_t;

// 内存 entry 状态
typedef enum {
  ncclDynMemStateActive = 0,  // Memory is allocated and usable
  ncclDynMemStateReleased = 1   // Memory has been released
} ncclDynMemState_t;

// 本地 owned 内存 descriptor
typedef struct ncclDynMemLocalDesc {
  // Shareable 句柄 for P2P exports
  // 待办: Remove the 'fd' 字段 - POSIX FD 句柄 are converted on-demand via 代理
  // (ncclProxyClientGetFdBlocking), 所以 we 不再 export them upfront. 仅 FABRIC
  // 句柄 需要 upfront export 自 they 可以 shared directly via messaging.
  union {
    int fd;            // For POSIX_FILE_DESCRIPTOR (unused)
    CUmemFabricHandle fabricHandle;  // For FABRIC
  } shareableHandle;
  bool shareableHandleValid;
  // 对等端 tracking for P2P exports
  int numExportedPeers;
  int exportedPeersCapacity;
  int* exportedPeerRanks;
} ncclDynMemLocalDesc;

// Imported from 对等端 内存 descriptor
typedef struct ncclDynMemImportDesc {
  int ownerRank;     // Rank that owns the original buffer
  int ownerDev;      // CUDA device of the owner
  void* ownerPtr;      // Owner's virtual address
} ncclDynMemImportDesc;

// Individual tracked 内存 entry (仅 track scratch 并且 offload 分配)
typedef struct ncclDynMemEntry {
  void* ptr;           // GPU virtual address
  size_t size;          // Allocation size
  CUmemGenericAllocationHandle handle;        // Physical memory handle
  CUmemAllocationHandleType handleType;
  ncclMemType_t memType;
  ncclDynMemState_t state;
  int cudaDev;

  // CPU backup for OFFLOAD 类型 内存
  void* cpuBackup;     // Host memory for offloaded data

  // Ownership 类型 并且 类型-特定的 数据
  bool isImportedFromPeer;  // true if this is a peer-imported buffer
  union {
    ncclDynMemLocalDesc local;
    ncclDynMemImportDesc imported;
  } desc;

  // 已链接 列表 指针
  struct ncclDynMemEntry* next;
} ncclDynMemEntry;

// P2P 句柄 Exchange 结构
typedef struct ncclDynMemP2pHandleInfo {
  void* ptr;
  int ownerRank;
  int ownerDev;
  size_t size;
  int handleType;
  union {
    uint64_t handleData;
    CUmemFabricHandle fabricHandle;
  };
} ncclDynMemP2pHandleInfo;

// 内存 管理器 attached to ncclComm
typedef struct ncclMemManager {
  ncclDynMemEntry* entries;  // Linked list of tracked allocations, only track scratch and offload allocations
  int numEntries;
  std::mutex lock;
  int released;
  int initialized;
  int refCount;

  size_t totalPersist;
  size_t totalPersistImported;
  size_t totalScratch;
  size_t totalScratchImported;
  size_t totalOffload;
  size_t totalOffloadImported;
  size_t cpuBackupUsage;

  int commCudaDev;
} ncclMemManager;

struct ncclMemManagerTask {
  struct ncclMemManagerTask* next;
  struct ncclComm* comm;
};

// 初始化 内存 管理器
ncclResult_t ncclMemManagerInit(struct ncclComm* comm);

// 销毁 内存 管理器 并且 释放 所有 resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm);

// Track a new 分配
ncclResult_t ncclMemTrack(struct ncclMemManager* manager, void* ptr, size_t size, CUmemGenericAllocationHandle handle,
                          CUmemAllocationHandleType handleType, ncclMemType_t memType);

// Track imported 分配 from 对等端
ncclResult_t ncclMemTrackImportFromPeer(struct ncclMemManager* manager, void* ptr, size_t size,
                                        CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType,
                                        ncclMemType_t memType, int ownerRank, int ownerDev, void* ownerPtr);

// Untrack 分配
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size);

// Add 对等端 信息 for 缓冲区 在 ... 中 已链接 列表 entries (仅 for dynamic 内存: scratch/offload)
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm);
ncclResult_t ncclCommMemResume(struct ncclComm* comm);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_MEM_MANAGER_H_ */
