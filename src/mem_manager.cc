/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/mem_manager.cc — 用户注册内存管理器实现
 * ----------------------------------------------------------------------------
 * 实现 ncclMemoryManager：把用户传入的 host/device buffer 登记为可被 transport 直接
 * 访问的“注册段”，管理其 IPC 句柄、引用计数、地址映射与生命周期。被 register.cc
 * 与 transport 复用。
 */

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "argcheck.h"
#include "cudawrap.h"
#include "debug.h"
#include "bootstrap.h"
#include "proxy.h"
#include "transport.h"
#include "nvtx.h"
#include "param.h"
#include "group.h"
#include "compiler.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <mutex>

// 内部参数：用于测试时禁用内存管理器
NCCL_PARAM(MemManagerDisable, "DISABLE_MEM_MANAGER", 0);

// 初始化内存管理器
ncclResult_t ncclMemManagerInit(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclMemManager* mgr;
  NCCLCHECK(ncclCalloc(&mgr, 1));
  // 用 placement new 显式构造 std::mutex
  new (&mgr->lock) std::mutex();

  mgr->entries = nullptr;
  mgr->numEntries = 0;
  mgr->released = 0;
  mgr->refCount = 1;
  mgr->totalPersist = 0;
  mgr->totalPersistImported = 0;
  mgr->totalScratch = 0;
  mgr->totalScratchImported = 0;
  mgr->totalOffload = 0;
  mgr->totalOffloadImported = 0;
  mgr->cpuBackupUsage = 0;
  mgr->commCudaDev = comm->cudaDev;

  COMPILER_ATOMIC_STORE(&mgr->initialized, 1, std::memory_order_release);

  comm->memManager = mgr;

  INFO(NCCL_ALLOC, "MemManager: Initialized for device %d", comm->cudaDev);
  return ncclSuccess;
}

// 销毁内存管理器并释放全部资源
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclSuccess;

  ncclMemManager* mgr = comm->memManager;

  if (!COMPILER_ATOMIC_LOAD(&mgr->initialized, std::memory_order_acquire)) {
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  // 递减引用计数
  int refCount = ncclAtomicRefCountDecrement(&mgr->refCount);

  if (refCount > 0) {
    // 还有其它 comm 正在使用本管理器
    INFO(NCCL_ALLOC, "MemManager: Decremented refCount to %d", refCount);
    comm->memManager = nullptr;  // Clear this comm's pointer
    return ncclSuccess;
  }

  // 引用计数为 0，此时 proxy 线程应当已经 join 完毕
  INFO(NCCL_ALLOC, "MemManager: Destroying (refCount=0)");
  COMPILER_ATOMIC_STORE(&mgr->initialized, 0, std::memory_order_release);

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    ncclDynMemEntry* next = entry->next;

    // 若存在则释放 CPU 备份内存
    if (entry->cpuBackup != nullptr) {
      ncclCudaHostFree(entry->cpuBackup);
    }

    // 若共享 FD 有效则关闭(对 POSIX FD 句柄类型的防御性清理)
    if (!entry->isImportedFromPeer && entry->desc.local.shareableHandleValid &&
        entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // 只有本地条目才有 exportedPeerRanks(导入条目使用 desc.imported 联合体成员)
    if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
      free(entry->desc.local.exportedPeerRanks);
    }

    // 释放条目自身
    free(entry);
    entry = next;
  }

  mgr->entries = nullptr;
  mgr->numEntries = 0;

  // 显式调用 std::mutex 的析构函数
  mgr->lock.~mutex();
  // 释放管理器结构体
  free(mgr);
  comm->memManager = nullptr;

  INFO(NCCL_ALLOC, "MemManager: Destroyed");
  return ncclSuccess;
}

// 内部辅助函数：创建并跟踪一个内存条目
static ncclResult_t ncclMemTrackInternal(struct ncclMemManager* manager, void* ptr, size_t size,
                                         CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType,
                                         ncclMemType_t memType, bool isImportedFromPeer, int ownerRank, int ownerDev,
                                         void* ownerPtr) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!COMPILER_ATOMIC_LOAD(&manager->initialized, std::memory_order_acquire)) {
    WARN("MemManager: Cannot track allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // 持久内存：仅做原子更新
  if (memType == ncclMemPersist) {
    if (isImportedFromPeer) {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalPersistImported, size, std::memory_order_relaxed);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent Import ptr=%p size=%zu from rank=%d", ptr, size, ownerRank);
    } else {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalPersist, size, std::memory_order_relaxed);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent ptr=%p size=%zu dev=%d", ptr, size, manager->commCudaDev);
    }
    return ncclSuccess;
  }

  // Scratch/Offload：创建链表条目
  ncclDynMemEntry* entry = (ncclDynMemEntry*)malloc(sizeof(ncclDynMemEntry));
  if (entry == nullptr) {
    WARN("MemManager: Failed to allocate memory entry");
    return ncclSystemError;
  }

  // 初始化公共字段
  memset(entry, 0, sizeof(ncclDynMemEntry));
  entry->ptr = ptr;
  entry->size = size;
  entry->handle = handle;
  entry->handleType = handleType;
  entry->memType = memType;
  entry->state = ncclDynMemStateActive;
  entry->cudaDev = manager->commCudaDev;
  entry->cpuBackup = nullptr;
  entry->isImportedFromPeer = isImportedFromPeer;

  // 初始化与所有权相关的字段
  if (isImportedFromPeer) {
    entry->desc.imported.ownerRank = ownerRank;
    entry->desc.imported.ownerDev = ownerDev;
    entry->desc.imported.ownerPtr = ownerPtr;
  } else {
    if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      entry->desc.local.shareableHandle.fd = -1;  // avoid using 0 which is stdin
    }
    entry->desc.local.shareableHandleValid = false;
    entry->desc.local.numExportedPeers = 0;
    entry->desc.local.exportedPeersCapacity = 0;
    entry->desc.local.exportedPeerRanks = nullptr;
  }

  { // lock the mutex to add the entry to the linked list
    std::lock_guard<std::mutex> lock(manager->lock);
    // 加入链表(头插法)
    entry->next = manager->entries;
    manager->entries = entry;
    manager->numEntries++;
  } // lock_guard automatically releases mutex

  // 更新统计信息
  if (isImportedFromPeer) {
    if (memType == ncclMemScratch) {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalScratchImported, size, std::memory_order_relaxed);
    } else if (memType == ncclMemOffload) {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalOffloadImported, size, std::memory_order_relaxed);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track imported ptr=%p size=%zu type=%d from rank=%d entries=%d", ptr, size, memType,
          ownerRank, manager->numEntries);
  } else {
    if (memType == ncclMemScratch) {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalScratch, size, std::memory_order_relaxed);
    } else if (memType == ncclMemOffload) {
      (void)COMPILER_ATOMIC_ADD_FETCH(&manager->totalOffload, size, std::memory_order_relaxed);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track ptr=%p size=%zu type=%d dev=%d entries=%d", ptr, size, memType,
          manager->commCudaDev, manager->numEntries);
  }

  return ncclSuccess;
}

// 跟踪一次新的分配
ncclResult_t ncclMemTrack(struct ncclMemManager* manager, void* ptr, size_t size, CUmemGenericAllocationHandle handle,
                          CUmemAllocationHandleType handleType, ncclMemType_t memType) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType, false, -1, -1, nullptr);
}

// 跟踪从对端导入的分配
ncclResult_t ncclMemTrackImportFromPeer(struct ncclMemManager* manager, void* ptr, size_t size,
                                        CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType,
                                        ncclMemType_t memType, int ownerRank, int ownerDev, void* ownerPtr) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType, true, ownerRank, ownerDev, ownerPtr);
}

// 取消对分配的跟踪
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;

  // 用原子检查避免对已被销毁的互斥量加锁
  if (!COMPILER_ATOMIC_LOAD(&manager->initialized, std::memory_order_acquire)) {
    WARN("MemManager: Cannot untrack allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // 在释放锁之前保存值的临时变量
  size_t entrySize = 0;
  int numEntries COMPILER_ATTRIBUTE_UNUSED = 0;  // May be unused if TRACE compiled out
  bool isImportedFromPeer = false;
  ncclMemType_t memType = ncclMemScratch;

  {
    std::lock_guard<std::mutex> lock(manager->lock);

    ncclDynMemEntry* prev = nullptr;
    ncclDynMemEntry* entry = manager->entries;

    while (entry != nullptr) {
      if (entry->ptr == ptr) {
        // 从链表中移除
        if (prev == nullptr) {
          manager->entries = entry->next;
        } else {
          prev->next = entry->next;
        }
        manager->numEntries--;

        // 若存在则释放 CPU 备份内存
        if (entry->cpuBackup != nullptr) {
          manager->cpuBackupUsage -= entry->size;
          ncclCudaHostFree(entry->cpuBackup);
        }

        // 若共享 FD 有效则关闭(对 POSIX FD 句柄类型的防御性清理)
        if (!entry->isImportedFromPeer && entry->desc.local.shareableHandleValid &&
            entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
            entry->desc.local.shareableHandle.fd >= 0) {
          close(entry->desc.local.shareableHandle.fd);
          entry->desc.local.shareableHandle.fd = -1;
          entry->desc.local.shareableHandleValid = false;
        }

        // 只有本地条目才有 exportedPeerRanks(导入条目使用 desc.imported 联合体成员)
        if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
          free(entry->desc.local.exportedPeerRanks);
        }

        // 在解锁前保存值以供日志使用(若 TRACE 被编译掉则可能未使用)
        entrySize = entry->size;
        numEntries = manager->numEntries;
        isImportedFromPeer = entry->isImportedFromPeer;
        memType = entry->memType;

        // 安全检查：若跟踪到的尺寸与传入尺寸不符则记录日志
        if (entrySize != size) {
          INFO(NCCL_ALLOC, "MemManager: Untrack size mismatch ptr=%p tracked=%zu passed=%zu", ptr, entrySize, size);
        }

        free(entry);
        break;
      }
      prev = entry;
      entry = entry->next;
    }
  } // lock_guard automatically releases mutex

  // 更新统计信息
  if (entrySize > 0) {
    // 在链表中找到了该条目
    if (isImportedFromPeer) {
      if (memType == ncclMemScratch) {
        (void)COMPILER_ATOMIC_SUB_FETCH(&manager->totalScratchImported, entrySize, std::memory_order_relaxed);
      } else if (memType == ncclMemOffload) {
        (void)COMPILER_ATOMIC_SUB_FETCH(&manager->totalOffloadImported, entrySize, std::memory_order_relaxed);
      }
    } else {
      if (memType == ncclMemScratch) {
        (void)COMPILER_ATOMIC_SUB_FETCH(&manager->totalScratch, entrySize, std::memory_order_relaxed);
      } else if (memType == ncclMemOffload) {
        (void)COMPILER_ATOMIC_SUB_FETCH(&manager->totalOffload, entrySize, std::memory_order_relaxed);
      }
    }

    TRACE(NCCL_ALLOC, "MemManager: Untrack ptr=%p size=%zu entries=%d", ptr, entrySize, numEntries);
  } else {
    // 链表中未找到该条目——必为持久内存
    (void)COMPILER_ATOMIC_SUB_FETCH(&manager->totalPersist, size, std::memory_order_relaxed);
    TRACE(NCCL_ALLOC, "MemManager: Untrack Persistent ptr=%p size=%zu", ptr, size);
  }

  return ncclSuccess;
}

// 标记某缓冲区正与对端共享(用于挂起/恢复时的协调)
// 注意：仅对链表中的动态内存(scratch/offload)有效。
// 持久内存无需导出跟踪，因为它永远不会被挂起。
// 在分配动态内存、且对端导入它之后调用本函数。
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!COMPILER_ATOMIC_LOAD(&manager->initialized, std::memory_order_acquire)) {
    WARN("MemManager: Cannot mark export for ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }
  std::lock_guard<std::mutex> lock(manager->lock);

  // 在链表中查找条目(链表只含 scratch/offload，不含持久内存)
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr && entry->ptr != ptr) {
    entry = entry->next;
  }

  if (entry == nullptr) {
    WARN("MemManager: Cannot mark export for ptr=%p - not found in tracked entries. "
         "Only dynamic memory (scratch/offload) needs export tracking for suspend/resume.",
         ptr);
    return ncclInternalError;
  }

  // 确认这是一个本地条目，而非导入条目
  if (entry->isImportedFromPeer) {
    WARN("MemManager: Cannot mark export for ptr=%p - this is an imported buffer, not a local one", ptr);
    return ncclInternalError;
  }

  // 检查该对端是否已存在
  for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
    if (entry->desc.local.exportedPeerRanks[i] == peerRank) {
      WARN("MemManager: Buffer ptr=%p already exported to peer rank %d", ptr, peerRank);
      return ncclInternalError;
    }
  }

  if (entry->desc.local.numExportedPeers >= entry->desc.local.exportedPeersCapacity) {
    int newCapacity = entry->desc.local.exportedPeersCapacity == 0 ? NCCL_MEM_EXPORT_PEERS_INIT :
                                                                     entry->desc.local.exportedPeersCapacity * 2;
    ncclResult_t ret =
      ncclRealloc(&entry->desc.local.exportedPeerRanks, entry->desc.local.exportedPeersCapacity, newCapacity);
    if (ret != ncclSuccess) {
      WARN("MemManager: Failed to grow exportedPeerRanks array for ptr=%p", ptr);
      return ret;
    }
    entry->desc.local.exportedPeersCapacity = newCapacity;
  }

  // 把该对端加入导出列表
  entry->desc.local.exportedPeerRanks[entry->desc.local.numExportedPeers++] = peerRank;

  TRACE(NCCL_ALLOC, "MemManager: ExportToPeer ptr=%p peerRank=%d numExportedPeers=%d", ptr, peerRank,
        entry->desc.local.numExportedPeers);
  return ncclSuccess;
}

/*
 * Internal: Suspend all dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Unmap all peer-imported buffers
 * 2. Second pass: Offload local buffers to CPU and suspend physical memory
 */
ncclResult_t ncclCommMemSuspend(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: Suspend failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;

  if (manager->released) {
    WARN("MemManager: Already suspended");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  size_t releasedScratch = 0;
  size_t releasedOffload = 0;
  size_t releasedPeerImport = 0;
  int releasedCount = 0;
  int peerImportCount = 0;
  ncclDynMemEntry* entry = nullptr;

  CUDACHECK(cudaDeviceSynchronize());
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF), ret, fail);

  // 第一步：先解除所有对端导入缓冲区的映射
  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer && entry->state == ncclDynMemStateActive) {
      TRACE(NCCL_ALLOC, "MemManager: Unmapping peer-imported buffer ptr=%p from rank %d", entry->ptr,
            entry->desc.imported.ownerRank);

      // 解除我们对端内存的本地映射
      CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));

      // 若持有对端句柄的引用则释放它
      // 对同进程导入，若映射后引用已被释放，句柄可能为 0
      if (entry->handle != 0) {
        CUCHECKIGNORE(cuMemRelease(entry->handle));
        entry->handle = 0;  // Clear invalid handle
      }

      entry->state = ncclDynMemStateReleased;
      releasedPeerImport += entry->size;
      peerImportCount++;
    }
    entry = entry->next;
  }

  // 第二步：卸载(offload)并释放本地内存
  entry = manager->entries;
  while (entry != nullptr) {
    // 跳过从对端导入的缓冲区
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }

    // 跳过已释放的缓冲区
    if (entry->state == ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    // 对 OFFLOAD 类型：先拷到 CPU 备份
    if (entry->memType == ncclMemOffload) {
      NCCLCHECKGOTO(ncclCudaHostCalloc((char**)&entry->cpuBackup, entry->size), ret, fail);
      if (entry->cpuBackup == nullptr) {
        WARN("MemManager: Failed to allocate CPU backup for offload");
        ret = ncclSystemError;
        goto fail;
      }

      // 把 GPU 数据拷到 CPU
      cudaError_t err = cudaMemcpy(entry->cpuBackup, entry->ptr, entry->size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        ncclCudaHostFree(entry->cpuBackup);
        entry->cpuBackup = nullptr;
        WARN("MemManager: Failed to copy to CPU backup: %s", cudaGetErrorString(err));
        ret = ncclUnhandledCudaError;
        goto fail;
      }

      manager->cpuBackupUsage += entry->size;
      releasedOffload += entry->size;
    } else {
      releasedScratch += entry->size;
    }

    // 若共享 FD 有效则关闭(针对 POSIX 句柄)
    if (entry->desc.local.shareableHandleValid && entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // 解除物理内存映射，但保留虚拟地址的预留
    CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));

    // 释放物理内存句柄
    CUCHECKIGNORE(cuMemRelease(entry->handle));
    entry->handle = 0;  // Clear invalid handle

    entry->state = ncclDynMemStateReleased;
    releasedCount++;

    entry = entry->next;
  }

  manager->released = 1;

  INFO(NCCL_ALLOC,
       "MemManager: rank %d suspended %d local + %d peer entries (scratch=%zu, offload=%zu, peerImport=%zu, "
       "cpuBackup=%zu)",
       comm->rank, releasedCount, peerImportCount, releasedScratch, releasedOffload, releasedPeerImport,
       manager->cpuBackupUsage);

  return ncclSuccess;

fail:
  return ret;
}

/*
 * Internal: Resume previously suspended dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Resume local memory (re-allocate, re-map, restore offloaded data)
 * 2. Exchange: AllGather new handle info for P2P buffers
 * 3. Second pass: Re-import peer buffers using new handles
 */
ncclResult_t ncclCommMemResume(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: Resume failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;

  if (!manager->released) {
    WARN("MemManager: Not in suspended state");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  int restoredLocalCount = 0;
  int restoredPeerCount = 0;
  size_t restoredLocalBytes = 0;
  size_t restoredPeerBytes = 0;

  int localBroadcastCount = 0;
  int* allCounts = nullptr;
  int totalInfoCount = 0;
  ncclDynMemP2pHandleInfo* localInfos = nullptr;
  ncclDynMemP2pHandleInfo* allInfos = nullptr;

  // 第一步：恢复所有本地内存
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr) {
    // 跳过从对端导入的条目
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }

    // 跳过未被释放的条目
    if (entry->state != ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    // 重新创建物理分配
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = entry->cudaDev;
    prop.requestedHandleTypes = entry->handleType;

    CUmemGenericAllocationHandle newHandle;
    CUCHECKGOTO(cuMemCreate(&newHandle, entry->size, &prop, 0), ret, fail);

    // 重新映射到同一虚拟地址，并为本地设备设置访问权限
    ret = ncclCuMemMapAndSetAccess(entry->ptr, entry->size, newHandle, entry->cudaDev);
    if (ret != ncclSuccess) {
      CUCHECKIGNORE(cuMemRelease(newHandle));
      entry->handle = 0;  // Clear to avoid dangling handle
      goto fail;
    }

    // 为所有此前已导出的对端恢复访问权限
    for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
      int peerRank = entry->desc.local.exportedPeerRanks[i];
      if (peerRank >= 0 && peerRank < comm->nRanks) {
        // 仅为同一节点、同一进程内的对端设置访问权限
        if (comm->peerInfo[peerRank].pidHash == comm->peerInfo[comm->rank].pidHash &&
            comm->peerInfo[peerRank].hostHash == comm->peerInfo[comm->rank].hostHash) {
          int peerDev = comm->peerInfo[peerRank].cudaDev;
          CUmemAccessDesc peerAccessDesc = {};
          peerAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          peerAccessDesc.location.id = peerDev;
          peerAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          CUCHECKIGNORE(cuMemSetAccess((CUdeviceptr)entry->ptr, entry->size, &peerAccessDesc, 1));
          TRACE(NCCL_ALLOC, "MemManager: Restored peer access for ptr=%p to rank %d dev %d", entry->ptr, peerRank,
                peerDev);
        }
      }
    }

    // 更新句柄
    entry->handle = newHandle;

    // 对 OFFLOAD 类型：从 CPU 备份恢复数据
    if (entry->memType == ncclMemOffload && entry->cpuBackup != NULL) {
      cudaError_t err = cudaMemcpy(entry->ptr, entry->cpuBackup, entry->size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        WARN("MemManager: Failed to restore from CPU backup: %s (backup preserved)", cudaGetErrorString(err));
        ret = ncclUnhandledCudaError;
        goto fail;
      }
      // 恢复成功后释放 CPU 备份
      ncclCudaHostFree(entry->cpuBackup);
      entry->cpuBackup = nullptr;
      manager->cpuBackupUsage -= entry->size;
    }

    entry->desc.local.shareableHandleValid = false;
    if (entry->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUresult exportRet = CUPFN(cuMemExportToShareableHandle(&entry->desc.local.shareableHandle.fabricHandle,
                                                              newHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
      if (exportRet != CUDA_SUCCESS) {
        WARN("MemManager: cuMemExportToShareableHandle (FABRIC) failed for ptr=%p", entry->ptr);
        CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));
        CUCHECKIGNORE(cuMemRelease(newHandle));
        entry->handle = 0;
        ret = ncclUnhandledCudaError;
        goto fail;
      }
      entry->desc.local.shareableHandleValid = true;
    }

    entry->state = ncclDynMemStateActive;
    restoredLocalCount++;
    restoredLocalBytes += entry->size;

    TRACE(NCCL_ALLOC, "MemManager: Resumed local buffer ptr=%p size=%zu numExportedPeers=%d", entry->ptr, entry->size,
          entry->desc.local.numExportedPeers);

    entry = entry->next;
  }

  // 第二步：用屏障确保所有 rank 都已恢复各自的本地内存
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local entries, waiting at barrier", comm->rank,
         restoredLocalCount);
    ret = bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF);
    if (ret != ncclSuccess) {
      WARN("MemManager: Barrier failed during resume");
      return ret;
    }
  }

  /*
   * Step 3: Exchange new handle info for P2P coordination
   *
   * Each rank broadcasts info about its local buffers that have peers.
   * We use a simple approach: each rank sends to all peers that imported its buffers.
   */

  // 统计拥有对端的本地缓冲区(需要广播新的句柄信息)
  localBroadcastCount = 0;
  entry = manager->entries;
  while (entry != nullptr) {
    if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 && entry->state == ncclDynMemStateActive) {
      localBroadcastCount++;
    }
    entry = entry->next;
  }

  // 用 AllGather 从所有 rank 收集计数
  if (comm->bootstrap != nullptr && comm->nRanks > 1) {
    // 为所有计数分配缓冲区
    allCounts = (int*)malloc(comm->nRanks * sizeof(int));
    if (allCounts == nullptr) {
      WARN("MemManager: Failed to allocate allCounts");
      return ncclSystemError;
    }
    memset(allCounts, 0, comm->nRanks * sizeof(int));
    allCounts[comm->rank] = localBroadcastCount;

    // AllGather 计数：每个 rank 在自身位置贡献 sizeof(int) 个字节
    ret = bootstrapAllGather(comm->bootstrap, allCounts, sizeof(int));
    if (ret != ncclSuccess) {
      free(allCounts);
      WARN("MemManager: AllGather counts failed");
      return ret;
    }

    // 计算总量与各 rank 的偏移
    int* offsets = (int*)malloc(comm->nRanks * sizeof(int));
    if (offsets == nullptr) {
      free(allCounts);
      return ncclSystemError;
    }

    totalInfoCount = 0;
    for (int r = 0; r < comm->nRanks; r++) {
      offsets[r] = totalInfoCount;
      totalInfoCount += allCounts[r];
    }

    if (totalInfoCount > 0) {
      // 准备要发送的本地信息
      int localAllocCount = localBroadcastCount > 0 ? localBroadcastCount : 1;
      localInfos = (ncclDynMemP2pHandleInfo*)malloc(localAllocCount * sizeof(ncclDynMemP2pHandleInfo));
      if (localInfos == nullptr) {
        free(allCounts);
        free(offsets);
        return ncclSystemError;
      }

      int idx = 0;
      entry = manager->entries;
      while (entry != nullptr && idx < localBroadcastCount) {
        if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 &&
            entry->state == ncclDynMemStateActive) {
          localInfos[idx].ptr = entry->ptr;
          localInfos[idx].ownerRank = comm->rank;
          localInfos[idx].ownerDev = entry->cudaDev;
          localInfos[idx].size = entry->size;
          localInfos[idx].handleType = entry->handleType;

          if (entry->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
            // 对 FABRIC：拷贝导出的 fabric 句柄(可直接共享)
            if (entry->desc.local.shareableHandleValid) {
              memcpy(&localInfos[idx].fabricHandle, &entry->desc.local.shareableHandle.fabricHandle,
                     sizeof(CUmemFabricHandle));
            } else {
              WARN("MemManager: FABRIC handle not valid for entry ptr=%p", entry->ptr);
            }
          } else {
            // 对 POSIX FD：保存 cuMem 句柄(供经由 proxy 转换为 FD)
            memcpy(&localInfos[idx].handleData, &entry->handle, sizeof(CUmemGenericAllocationHandle));
          }

          idx++;
        }
        entry = entry->next;
      }

      // 为所有信息分配缓冲区
      allInfos = (ncclDynMemP2pHandleInfo*)malloc(totalInfoCount * sizeof(ncclDynMemP2pHandleInfo));
      if (allInfos == nullptr) {
        free(allCounts);
        free(offsets);
        free(localInfos);
        return ncclSystemError;
      }

      // 把本地数据拷到正确位置
      if (localBroadcastCount > 0) {
        memcpy(allInfos + offsets[comm->rank], localInfos, localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo));
      }

      // 用 Send/Recv 交换(先发后收，避免死锁)
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && localBroadcastCount > 0) {
          ret = bootstrapSend(comm->bootstrap, r, 0xFEED, localInfos,
                              localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo));
          if (ret != ncclSuccess) {
            WARN("MemManager: Send to rank %d failed - handle exchange incomplete", r);
            free(offsets);
            free(allCounts);
            free(localInfos);
            free(allInfos);
            return ret;
          }
        }
      }

      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && allCounts[r] > 0) {
          ret = bootstrapRecv(comm->bootstrap, r, 0xFEED, allInfos + offsets[r],
                              allCounts[r] * sizeof(ncclDynMemP2pHandleInfo));
          if (ret != ncclSuccess) {
            WARN("MemManager: Recv from rank %d failed - handle exchange incomplete", r);
            free(offsets);
            free(allCounts);
            free(localInfos);
            free(allInfos);
            return ret;
          }
        }
      }
    }

    free(offsets);
  }

  /*
   * Step 4: Re-import peer buffers using exchanged handle info
   */

  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer && entry->state == ncclDynMemStateReleased) {
      // 从 AllGather 结果中找到匹配的信息
      ncclDynMemP2pHandleInfo* matchedInfo = nullptr;

      if (allInfos != nullptr) {
        for (int i = 0; i < totalInfoCount; i++) {
          if (allInfos[i].ownerRank == entry->desc.imported.ownerRank &&
              allInfos[i].ptr == entry->desc.imported.ownerPtr && allInfos[i].size == entry->size) {
            matchedInfo = &allInfos[i];
            break;
          }
        }
      }

      if (matchedInfo == nullptr) {
        WARN("MemManager: Could not find matching handle info for ptr=%p from rank %d", entry->ptr,
             entry->desc.imported.ownerRank);
        entry = entry->next;
        continue;
      }

      TRACE(NCCL_ALLOC, "MemManager: Re-importing peer buffer ptr=%p from rank %d (owner ptr=%p)", entry->ptr,
            entry->desc.imported.ownerRank, matchedInfo->ptr);

      CUmemGenericAllocationHandle newHandle;
      CUresult curet;

      if (matchedInfo->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        // POSIX FD 句柄只在同一节点内有效——检查 hostHash
        if (comm->peerInfo && comm->peerInfoValid &&
            comm->peerInfo[entry->desc.imported.ownerRank].hostHash != comm->peerInfo[comm->rank].hostHash) {
          WARN("MemManager: Cannot re-import peer buffer from rank %d (different node) using POSIX FD - skipping",
               entry->desc.imported.ownerRank);
          entry = entry->next;
          continue;
        }

        // 对 POSIX FD：需要从拥有者处获取 FD
        // 用 proxy 把 cuMem 句柄转换为 FD
        int fd = -1;

        // handleData 中含有 cuMem 句柄——请求进行 FD 转换
        ret = ncclProxyClientGetFdBlocking(comm, entry->desc.imported.ownerRank, &matchedInfo->handleData, &fd);
        if (ret != ncclSuccess || fd < 0) {
          WARN("MemManager: Failed to get FD from rank %d for ptr=%p", entry->desc.imported.ownerRank, entry->ptr);
          entry = entry->next;
          continue;
        }

        curet = CUPFN(cuMemImportFromShareableHandle(&newHandle, (void*)(uintptr_t)fd,
                                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(fd);
      } else if (matchedInfo->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
        // 对 FABRIC：用 fabric 句柄直接导入
        curet =
          CUPFN(cuMemImportFromShareableHandle(&newHandle, &matchedInfo->fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC));
      } else {
        WARN("MemManager: Unknown handle type %d for peer import", matchedInfo->handleType);
        entry = entry->next;
        continue;
      }

      if (curet != CUDA_SUCCESS) {
        WARN("MemManager: cuMemImportFromShareableHandle failed for ptr=%p (curet=%d)", entry->ptr, curet);
        entry = entry->next;
        continue;
      }

      // 重新映射到同一虚拟地址并设置访问权限
      ncclResult_t mapResult = ncclCuMemMapAndSetAccess(entry->ptr, entry->size, newHandle, comm->cudaDev);
      if (mapResult != ncclSuccess) {
        CUCHECKIGNORE(cuMemRelease(newHandle));
        entry->handle = 0;
        WARN("MemManager: ncclCuMemMapAndSetAccess failed for re-imported ptr=%p", entry->ptr);
        entry = entry->next;
        continue;
      }

      entry->handle = newHandle;
      entry->state = ncclDynMemStateActive;
      restoredPeerCount++;
      restoredPeerBytes += entry->size;

      TRACE(NCCL_ALLOC, "MemManager: Successfully re-imported peer buffer ptr=%p from rank %d", entry->ptr,
            entry->desc.imported.ownerRank);
    }
    entry = entry->next;
  }

  manager->released = 0;

  // 最后用屏障确保所有 rank 都已完成对端导入的设置
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local + %d peer entries (%zu + %zu bytes)", comm->rank,
         restoredLocalCount, restoredPeerCount, restoredLocalBytes, restoredPeerBytes);
    ret = bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xCAFE);
    if (ret != ncclSuccess) {
      // 清理
      if (allCounts) free(allCounts);
      if (localInfos) free(localInfos);
      if (allInfos) free(allInfos);
      WARN("MemManager: Final barrier failed during resume");
      return ret;
    }
  }

  // 清理
  if (allCounts) free(allCounts);
  if (localInfos) free(localInfos);
  if (allInfos) free(allInfos);

  return ncclSuccess;

fail:
  if (allCounts) free(allCounts);
  if (localInfos) free(localInfos);
  if (allInfos) free(allInfos);
  return ret;
}

/*
 * Public Communicator Suspend/Resume APIs
 */

NCCL_API(ncclResult_t, ncclCommSuspend, ncclComm_t comm, int flags);
ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommSuspend", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  if (flags & NCCL_SUSPEND_MEM) {
    if (ncclParamMemManagerDisable()) {
      WARN("MemManager: Suspend not supported, memory manager is disabled");
      ret = ncclInvalidUsage;
      goto fail;
    }
    // 检查管理器是否被共享
    if (comm->memManager && comm->memManager->refCount > 1) {
      WARN("Memory suspend not supported with split_share communicators (refCount=%d)", comm->memManager->refCount);
      ret = ncclInvalidUsage;
      goto fail;
    }
    INFO(NCCL_INIT, "ncclCommSuspend: rank %d suspending memory", comm->rank);
    struct ncclMemManagerTask* task;
    NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
    task->comm = comm;
    ncclIntruQueueEnqueue(&comm->suspendTaskQueue, task);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister); // Reuse to avoid creating a new task type
  }

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->config.blocking) NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
  CUDACHECK(cudaSetDevice(saveDev));
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommResume, ncclComm_t comm);
ncclResult_t ncclCommResume(ncclComm_t comm) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommResume", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: Resume not supported, memory manager is disabled");
    ret = ncclInvalidUsage;
    goto fail;
  }
  // 检查管理器是否被共享
  if (comm->memManager && comm->memManager->refCount > 1) {
    WARN("Memory resume not supported with split_share communicators (refCount=%d)", comm->memManager->refCount);
    ret = ncclInvalidUsage;
    goto fail;
  }
  INFO(NCCL_INIT, "ncclCommResume: rank %d resuming all resources", comm->rank);
  struct ncclMemManagerTask* task;
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->comm = comm;
  ncclIntruQueueEnqueue(&comm->resumeTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister); // Reuse to avoid creating a new task type

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->config.blocking) NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
  CUDACHECK(cudaSetDevice(saveDev));
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommMemStats, ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value);
ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommMemStats", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  if (value == nullptr) return ncclInvalidArgument;

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: MemStats not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }

  if (comm->memManager == nullptr) {
    *value = 0;
    return ncclSuccess;
  }

  ncclMemManager* manager = comm->memManager;
  switch (stat) {
  case ncclStatGpuMemTotal:
    *value = COMPILER_ATOMIC_LOAD(&manager->totalPersist, std::memory_order_relaxed) +
             COMPILER_ATOMIC_LOAD(&manager->totalScratch, std::memory_order_relaxed) +
             COMPILER_ATOMIC_LOAD(&manager->totalOffload, std::memory_order_relaxed);
    return ncclSuccess;
  case ncclStatGpuMemPersist:
    *value = COMPILER_ATOMIC_LOAD(&manager->totalPersist, std::memory_order_relaxed);
    return ncclSuccess;
  case ncclStatGpuMemSuspend:
    *value = COMPILER_ATOMIC_LOAD(&manager->totalScratch, std::memory_order_relaxed) +
             COMPILER_ATOMIC_LOAD(&manager->totalOffload, std::memory_order_relaxed);
    return ncclSuccess;
  case ncclStatGpuMemSuspended:
    // 布尔量：0=活跃，1=已挂起
    *value = manager->released ? 1 : 0;
    return ncclSuccess;
  default:
    return ncclInvalidArgument;
  }
}
