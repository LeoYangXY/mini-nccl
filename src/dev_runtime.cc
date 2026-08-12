/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/dev_runtime.cc — device runtime：跨 CUDA 版本的 kernel 启动兼容
 * ----------------------------------------------------------------------------
 * 封装不同 CUDA toolkit 下的 kernel 启动与设备函数查询，使 NCCL 二进制能在多种
 * CUDA 版本上运行。devcomm/ 提供各 CUDA 版本的具体实现，本文件做统一调度。
 */

#include "dev_runtime_internal.h"
#include "comm.h"
#include "nccl_device/core.h"
#include "nccl_device/gin_barrier.h"
#include "nccl_device/lsa_barrier.h"
#include "rma/rma.h"
#include "device.h"
#include "sym_kernels.h"
#include "transport.h"
#include "group.h"
#include "nccl_device.h"
#include "utils.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif
#include "argcheck.h"
#include <mutex>

NCCL_PARAM(WinStride, "WIN_STRIDE", -1);
NCCL_PARAM(EnableVersionCheck, "ENABLE_VERSION_CHECK", 1);
NCCL_PARAM(ElasticBufferRegister, "ELASTIC_BUFFER_REGISTER", 1);
NCCL_PARAM(SymReuseSysmemHandles, "SYM_REUSE_SYSMEM_HANDLES", 0);
NCCL_PARAM(RMADisable, "RMA_DISABLE", 0);

extern struct ncclDevCommCompat ncclDevCommCompat_v22902, ncclDevCommCompat_v22907, ncclDevCommCompat_v23000;

// 数组中各条目的顺序应当无关紧要(以 nullptr 结尾的终止项除外)
static struct ncclDevCommCompat* devCommCompat[] = {&ncclDevCommCompat_v22902, &ncclDevCommCompat_v22907,
                                                    &ncclDevCommCompat_v23000, nullptr};

// 用侵入式地址映射实现的全局窗口表
// 直接使用 ncclDevrWindow(以显存地址为键，下一个 指针内嵌于结构体内)
static std::mutex ncclWindowMapMutex;
static ncclIntruAddressMap<ncclDevrWindow, struct ncclWindow_vidmem*, &ncclDevrWindow::vidmem, &ncclDevrWindow::next>
  ncclWindowMap;
static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream);

struct ncclDevrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclDevrWindow* win;
};

struct ncclDevrTeam {
  struct ncclDevrTeam* next;
  struct ncclTeam team;
  CUmemGenericAllocationHandle mcHandle;
  void* mcBasePtr;
  int worldRankList[];
};

////////////////////////////////////////////////////////////////////////////////
// 下方辅助函数：

// 找到满足 arg < sorted[i].key 的最小下标(即最小上界)
template <typename Obj, typename Key>
static int listFindSortedLub(Key Obj::* key, Obj* sorted, int count, Key arg);

template <typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);

template <typename Obj>
static void listRemove(Obj* list, int* count, int index);

////////////////////////////////////////////////////////////////////////////////

NCCL_PARAM(LsaTeamSize, "LSA_TEAM_SIZE", 0)

// 从通信域拓扑计算 LSA 团队大小，无任何副作用。
static int computeLsaSize(struct ncclComm* comm) {
  if (comm->devrState.bigSize != 0) return comm->devrState.lsaSize;

  // LSA 对所有 rank 必须大小一致，并且它要代表
  // 一组连续的 rank。
  int lsaSize = ncclParamLsaTeamSize();
  if (comm->p2pCrossClique && comm->nvlDomainSize == comm->nRanks) {
    // 单个 NVLD：所有 rank 通过 fabric 句柄共享内存。把 LSA 扩展到整个域。
    // 多 NVLD(各域大小可能不等)暂不支持跨 clique 的 LSA
    lsaSize = comm->nRanks;
    INFO(NCCL_INIT, "LSA extended to full NVL domain: lsaSize=%d (cross-clique P2P)", lsaSize);
  } else {
    // 基于节点的标准 gcd LSA 计算
    int nodeSize = 1;
    for (int r = 1; r < comm->nRanks; r++) {
      if (comm->rankToNode[r] == comm->rankToNode[r - 1]) {
        nodeSize += 1;
      } else {
        lsaSize = gcd(lsaSize, nodeSize);
        nodeSize = 1;
      }
    }
    lsaSize = gcd(lsaSize, nodeSize);
  }

  return lsaSize;
}

bool ncclDevrIsOneLsaTeam(struct ncclComm* comm) {
  int lsaSize = computeLsaSize(comm);
  return lsaSize == comm->nRanks; // Same as comm->nRanks / comm->devrState.lsaSize == 1
}

ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->bigSize != 0) return ncclSuccess;

  // LSA 对所有 rank 必须大小一致，并且它要代表
  // 一组连续的 rank。
  int lsaSize = computeLsaSize(comm);
  devr->lsaSize = lsaSize;
  devr->lsaSelf = comm->rank % lsaSize;
  devr->lsaRankList = (int*)malloc(devr->lsaSize * sizeof(int));
  for (int i = 0; i < devr->lsaSize; i++) {
    devr->lsaRankList[i] = comm->rank + (i - devr->lsaSelf);
  }
  devr->nLsaTeams = comm->nRanks / devr->lsaSize;

  CUmemAllocationProp memProp = {};
  memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes = ncclCuMemHandleType;
  memProp.location.id = comm->cudaDev;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&devr->granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret,
              fail_lsaRankList);

  devr->bigSize = ncclParamWinStride();
  if (-devr->bigSize <= 1) {
    devr->bigSize = 1;
    for (int r = 0; r < comm->nRanks; ++r) {
      devr->bigSize = std::max<size_t>(devr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  devr->bigSize = alignUp(devr->bigSize, size_t(1) << 32);
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)devr->bigSize >> 30);

  ncclSpaceConstruct(&devr->bigSpace);
  ncclShadowPoolConstruct(&devr->shadows);
  return ncclSuccess;

fail_lsaRankList:
  free(devr->lsaRankList);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm); // Further down

ncclResult_t ncclDevrFinalize(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  cudaStream_t stream;
  ncclResult_t ret = ncclSuccess;
  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
  if (devr->bigSize == 0) return ncclSuccess;

  CUDACHECKIGNORE(cudaThreadExchangeStreamCaptureMode(&captureMode));

  while (!ncclIntruQueueEmpty(&devr->regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&devr->regTaskQueue);
    free(task);
  }

  // 在 中止 或其它情形下，用户可能未调用 deregister API 来
  // 注销对称窗口对象，因此我们需要销毁所有未被用户注销、
  // 残留下来的窗口对象，以避免内存泄漏。
  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  while (devr->winSortedCount > 0) {
    struct ncclDevrWindow* win = devr->winSorted[0].win;
    NCCLCHECKIGNORE(symWindowDestroy(comm, win->vidmem, stream), ret);
  }
  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  symTeamDestroyAll(comm);

  // 删除窗口表
  struct ncclDevCommWindowTable* tableDev;
  tableDev = devr->windowTable;
  while (tableDev != nullptr) {
    struct ncclDevCommWindowTable* tableHost;
    if (ncclSuccess != ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost)) break;
    struct ncclDevCommWindowTable* next = tableHost->next;
    ncclShadowPoolFree(&devr->shadows, tableDev, stream);
    tableDev = next;
  }
  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  if (devr->lsaFlatBase != nullptr) {
    CUdeviceptr flatAddr = reinterpret_cast<CUdeviceptr>(devr->lsaFlatBase);
    CUCHECKIGNORE(cuMemUnmap(flatAddr, devr->lsaSize * devr->bigSize));
    CUCHECKIGNORE(cuMemAddressFree(flatAddr, devr->lsaSize * devr->bigSize));
  }
  ncclShadowPoolDestruct(&devr->shadows, stream);
  CUDACHECKIGNORE(cudaStreamDestroy(stream));
  CUDACHECKIGNORE(cudaThreadExchangeStreamCaptureMode(&captureMode));
  ncclSpaceDestruct(&devr->bigSpace);
  free(devr->lsaRankList);
  free(devr->winSorted);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

// LSA 团队 所有-收集 所使用的消息布局(每个 rank 每个段各一条)。
struct symLsaMessage {
  union {
    CUmemGenericAllocationHandle memHandle;
    CUmemFabricHandle fabricHandle;
  };
  CUmemLocationType type;
  size_t segmentSize;
};

static ncclResult_t symMemorySetAccessForVASegment(struct ncclComm* comm, symLsaMessage* message, CUdeviceptr addr) {
  CUmemAccessDesc accessDesc;
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess(addr, message->segmentSize, &accessDesc, 1));
  return ncclSuccess;
}

static ncclResult_t symMemoryExportSegmentHandle(struct ncclComm* comm, symLsaMessage* msg,
                                                 CUmemGenericAllocationHandle memHandle, size_t segmentSize) {
  ncclResult_t ret = ncclSuccess;
  CUmemAllocationProp prop;
  CUCHECKGOTO(cuMemGetAllocationPropertiesFromHandle(&prop, memHandle), ret, fail);
  msg->type = prop.location.type;
  msg->segmentSize = segmentSize;
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    msg->memHandle = memHandle;
  } else {
    CUCHECKGOTO(cuMemExportToShareableHandle(&msg->fabricHandle, memHandle, ncclCuMemHandleType, 0), ret, fail);
  }
fail:
  return ret;
}

static ncclResult_t symMemoryImportAndMapSegmentHandle(struct ncclComm* comm, int r, CUdeviceptr addr,
                                                       symLsaMessage* msg, CUmemGenericAllocationHandle memHandle,
                                                       bool reuseLocal) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  CUmemGenericAllocationHandle impHandle;
  if (reuseLocal) {
    impHandle = memHandle;
  } else {
    if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      int fd = -1;
      NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r], msg, &fd), ret, fail);
      CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, reinterpret_cast<void*>((uintptr_t)fd),
                                                 ncclCuMemHandleType),
                  ret, fail);
      SYSCHECKGOTO(close(fd), "close", ret, fail);
    } else {
      CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&msg->fabricHandle, ncclCuMemHandleType), ret,
                  fail);
    }
  }
  CUCHECKGOTO(cuMemMap(addr, msg->segmentSize, 0, impHandle, 0), ret, fail);
  NCCLCHECKGOTO(symMemorySetAccessForVASegment(comm, msg, addr), ret, fail);
  if (!reuseLocal) {
    CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);
  }
fail:
  return ret;
}

static ncclResult_t symMemoryImportAndMapSegmentsForRank(struct ncclComm* comm, int r, symLsaMessage* messages,
                                                         int maxSegments, int numSegments,
                                                         CUmemGenericAllocationHandle* memHandles, size_t bigOffset) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r * devr->bigSize + bigOffset);
  for (int segment = 0; segment < numSegments; segment++) {
    symLsaMessage* msg = messages + r * maxSegments + segment;
    bool reuseLocal =
      (r == devr->lsaSelf) || (ncclParamSymReuseSysmemHandles() && msg->type == CU_MEM_LOCATION_TYPE_HOST_NUMA);
    CUmemGenericAllocationHandle handle = reuseLocal ? memHandles[segment] : (CUmemGenericAllocationHandle)0ULL;
    NCCLCHECKGOTO(symMemoryImportAndMapSegmentHandle(comm, r, addr, msg, handle, reuseLocal), ret, fail);
    addr += msg->segmentSize;
  }
fail:
  return ret;
}

static ncclResult_t symMemoryMapLsaTeam(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  symLsaMessage* messages = nullptr;
  int* segmentCounts = mem->lsaNumSegments;  // filled from global allgather in symMemoryObtain
  int maxSegments = 0;
  const int numSegments = mem->numSegments;
  size_t* segmentSizes = mem->segmentSizes;

  for (int rank = 0; rank < devr->lsaSize; rank++) {
    maxSegments = std::max(maxSegments, segmentCounts[rank]);
  }

  NCCLCHECKGOTO(ncclCalloc(&messages, (size_t)devr->lsaSize * maxSegments), ret, fail);

  for (int segment = 0; segment < numSegments; segment++) {
    symLsaMessage* msg = messages + devr->lsaSelf * maxSegments + segment;
    NCCLCHECKGOTO(symMemoryExportSegmentHandle(comm, msg, mem->memHandles[segment], segmentSizes[segment]), ret, fail);
    INFO(NCCL_REG, "[%d] Segment %d, Type : %d, numSegments : %d, Segment size : %ld, memHandle : %lld", devr->lsaSelf,
         segment, msg->type, numSegments, msg->segmentSize, msg->memHandle);
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, messages,
                                            sizeof(symLsaMessage) * maxSegments),
                ret, fail);

  if (devr->lsaFlatBase == nullptr) {
    // 在首次需要时创建。
    CUdeviceptr addr;
    CUCHECKGOTO(cuMemAddressReserve(&addr, devr->lsaSize * devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr);
  }

  for (int r = 0; r < devr->lsaSize; r++) {
    NCCLCHECKGOTO(symMemoryImportAndMapSegmentsForRank(comm, r, messages, maxSegments, segmentCounts[r],
                                                       mem->memHandles, mem->bigOffset),
                  ret, fail);
  }
  // 确保所有人都已导入我的内存句柄。
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, 0xbeef),
                ret, fail);
leave:
  free(messages);
  return ret;
fail:
  goto leave;
}

static ncclResult_t symBindTeamMemory(struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
#if CUDART_VERSION >= 12010
    // 含 CPU 后端物理段的 mem 当前不支持 multimem 团队
    if (mem->globalHasSysmemSegment) {
      INFO(NCCL_NVLS, "Skipping bind multicast for maxGlobalNumSegments = %d, big=%lx, team {%d x %d}",
           mem->maxGlobalNumSegments, mem->bigOffset, tm->team.nRanks, tm->team.stride);
    } else {
      INFO(NCCL_NVLS, "Binding multicast memory at big=%lx size=%zu to team {%d x %d}", mem->bigOffset, mem->lsaMinSize,
           tm->team.nRanks, tm->team.stride);
      CUCHECK(cuMulticastBindAddr(tm->mcHandle, mem->bigOffset, reinterpret_cast<CUdeviceptr>(mem->primaryAddr),
                                  mem->lsaMinSize, 0));
    }
#endif
  }
  return ncclSuccess;
}

static ncclResult_t symUnbindTeamMemory(struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr && !mem->globalHasSysmemSegment) {
#if CUDART_VERSION >= 12010
    CUCHECK(cuMulticastUnbind(tm->mcHandle, comm->cudaDev, mem->bigOffset, mem->lsaMinSize));
#endif
  }
  return ncclSuccess;
}

// 调用者必须在此之后对团队做屏障同步。
static ncclResult_t symTeamObtain(struct ncclComm* comm, struct ncclTeam team, bool multimem,
                                  struct ncclDevrTeam** outTeam) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrTeam* t = devr->teamHead;
  bool teamIsNew = false;
  while (true) {
    if (t == nullptr) {
      teamIsNew = true;
      t = (struct ncclDevrTeam*)malloc(sizeof(struct ncclDevrTeam) + team.nRanks * sizeof(int));
      t->team = team;
      t->mcHandle = 0x0;
      t->mcBasePtr = nullptr;
      for (int i = 0; i < team.nRanks; i++) {
        t->worldRankList[i] = comm->rank + (i - team.rank) * team.stride;
      }
      break;
    } else if (t->team.rank == team.rank && t->team.nRanks == team.nRanks && t->team.stride == team.stride) {
      if (!multimem || t->mcBasePtr != nullptr) {
        // 匹配的团队即已足够
        if (outTeam) *outTeam = t;
        return ncclSuccess;
      }
      break; // Need to enable multimem
    } else {
      t = t->next;
    }
  }

  if (multimem) {
    if (!comm->nvlsSupport) {
      WARN("Multicast support requested for team but none available on system.");
      ret = ncclInvalidArgument;
      goto fail;
    } else {
#if CUDART_VERSION >= 12010
      CUmemGenericAllocationHandle mcHandle = 0;
      CUdeviceptr mcAddr = 0;
      CUmulticastObjectProp mcProp = {};
      char shareableHandle[NVLS_HANDLE_SIZE];

      mcProp.numDevices = team.nRanks;
      mcProp.handleTypes = ncclCuMemHandleType;
      mcProp.flags = 0;
      mcProp.size = devr->bigSize;
      if (team.rank == 0) {
        NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcProp, team.rank, team.nRanks, &mcHandle, shareableHandle), ret,
                      fail);
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0,
                                                  shareableHandle, NVLS_HANDLE_SIZE),
                      ret, fail_mcHandle);
      } else {
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0,
                                                  shareableHandle, NVLS_HANDLE_SIZE),
                      ret, fail);
        NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, t->worldRankList[0], &mcHandle), ret, fail);
      }

      CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->cudaDev), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemAddressReserve(&mcAddr, devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemMap(mcAddr, devr->bigSize, 0, mcHandle, 0), ret, fail_mcHandle_mcAddr);
      {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = comm->cudaDev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECKGOTO(cuMemSetAccess(mcAddr, devr->bigSize, &accessDesc, 1), ret, fail_mcHandle_mcAddr_unmap);
      }
      t->mcHandle = mcHandle;
      t->mcBasePtr = reinterpret_cast<void*>(mcAddr);

      // 把新团队与所有已存在的内存绑定。
      for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
        NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mcHandle_mcAddr_unmap_mems);
      }

      if (false) {
        // 错误标签：
      fail_mcHandle_mcAddr_unmap_mems:
        for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
          symUnbindTeamMemory(comm, t, mem);
        }
      fail_mcHandle_mcAddr_unmap:
        CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));
        goto fail_mcHandle_mcAddr; // silence unused label warning
      fail_mcHandle_mcAddr:
        CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));
        goto fail_mcHandle; // silence unused label warning
      fail_mcHandle:
        CUCHECKIGNORE(cuMemRelease(mcHandle));
        goto fail; // silence unused label warning
      }
#else
      goto fail; // silence unused label warning
#endif
    }
  }

  if (teamIsNew) {
     // 加入列表
    t->next = devr->teamHead;
    devr->teamHead = t;
  }
  if (outTeam) *outTeam = t;
  return ret;

fail:
  if (teamIsNew) free(t);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  while (devr->teamHead != nullptr) {
    struct ncclDevrTeam* t = devr->teamHead;
    devr->teamHead = t->next;
    if (t->mcBasePtr != nullptr) {
      for (struct ncclDevrMemory* m = devr->memHead; m != nullptr; m = m->next) {
        symUnbindTeamMemory(comm, t, m);
      }
      CUdeviceptr mcAddr = reinterpret_cast<CUdeviceptr>(t->mcBasePtr);
      CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));
      CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));
      CUCHECKIGNORE(cuMemRelease(t->mcHandle));
    }
    free(t);
  }
}

static ncclResult_t symMemoryRegisterGin(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  ncclResult_t ret = ncclSuccess;
  int numSegmentsRegistered = 0;
  size_t offset = 0;

  NCCLCHECKGOTO(ncclDevrVerifySegmentLayouts(mem, comm), ret, fail);
  NCCLCHECKGOTO(ncclDevrBuildGinSegmentInfos(mem), ret, fail);

  for (int segment = 0; segment < mem->numGinSegments; segment++) {
    int cuMemLocType = mem->ginSegmentInfos[segment].memType;
    int ptrType = (cuMemLocType == CU_MEM_LOCATION_TYPE_HOST_NUMA) ? NCCL_PTR_HOST : NCCL_PTR_CUDA;
    NCCLCHECKGOTO(ncclGinRegister(comm, (char*)mem->primaryAddr + offset, mem->ginSegmentInfos[segment].segmentSize,
                                  mem->ginSegmentInfos[segment].ginHostWins, mem->ginSegmentInfos[segment].ginDevWins,
                                  mem->winFlags, mem->maxGlobalNumSegments > 1, ptrType),
                  ret, fail);
    numSegmentsRegistered++;
    offset += mem->ginSegmentInfos[segment].segmentSize;
  }

  // 为单段情形缓存 ginWins，避免在设备上多做一次指针解引用
  if (mem->numGinSegments == 1) {
    for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
      mem->ginDevWins[i] = mem->ginSegmentInfos[0].ginDevWins[i];
      mem->ginHostWins[i] = mem->ginSegmentInfos[0].ginHostWins[i];
    }
  }
exit:
  return ret;
fail:
  for (int i = 0; i < numSegmentsRegistered; i++) {
    ncclGinDeregister(comm, mem->ginSegmentInfos[i].ginHostWins);
  }
  free(mem->ginSegmentInfos);
  mem->ginSegmentInfos = nullptr;
  goto exit;
}

static ncclResult_t symMemoryRegisterRma(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  NCCLCHECK(ncclRmaProxyConnectOnce(comm));
  NCCLCHECK(ncclRmaProxyRegister(comm, mem->primaryAddr, mem->size, mem->rmaHostWins));
  return ncclSuccess;
}

// 成功时，接管调用者持有的 memHandle 引用。
// 由于要对每个已存在的团队做多播绑定，本函数要求
// 调用者在返回用户之前做一次全局(world)屏障。
static ncclResult_t symMemoryObtain(struct ncclComm* comm, CUmemGenericAllocationHandle* memHandles, int numSegments,
                                    void* memAddr, size_t size, int winFlags, struct ncclDevrMemory** outMem,
                                    bool hasSysmemSegment = false) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  int64_t bigOffset = 0;
  struct segmentInfo {
    int numSegments;
    bool hasSysmemSegment;
    size_t totalSize;
  };
  struct segmentInfo* globalSegmentInfo = nullptr;
  const int globalLsaTeamBaseIdx = devr->lsaSize * (comm->rank / devr->lsaSize);

  struct ncclDevrMemory* mem = nullptr;
  // 新的内存。
  NCCLCHECKGOTO(ncclCalloc(&mem, 1), ret, fail_mem);
  NCCLCHECKGOTO(ncclCalloc(&mem->memHandles, numSegments), ret, fail_mem);
  memcpy(mem->memHandles, memHandles, sizeof(*mem->memHandles) * numSegments);
  mem->primaryAddr = memAddr;
  mem->size = size;
  mem->winFlags = winFlags;
  mem->hasSysmemSegment = hasSysmemSegment;
  mem->numSegments = numSegments;

  NCCLCHECKGOTO(ncclCalloc(&mem->segmentSizes, numSegments), ret, fail_mem);
  NCCLCHECKGOTO(ncclCalloc(&globalSegmentInfo, comm->nRanks), ret, fail_mem);

  NCCLCHECKGOTO(ncclDevrPopulateSegmentSizes(mem, numSegments), ret, fail_mem);

  // 我们需要最大段数与全局 sysmem 信息，以便有选择地禁用某些特性
  globalSegmentInfo[comm->rank].numSegments = numSegments;
  globalSegmentInfo[comm->rank].hasSysmemSegment = hasSysmemSegment;
  globalSegmentInfo[comm->rank].totalSize = size;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, globalSegmentInfo, sizeof(*globalSegmentInfo)), ret, fail_mem);
  mem->globalHasSysmemSegment = false;
  for (int r = 0; r < comm->nRanks; r++) {
    if (mem->maxGlobalNumSegments < globalSegmentInfo[r].numSegments) {
      mem->maxGlobalNumSegments = globalSegmentInfo[r].numSegments;
    }
    if (globalSegmentInfo[r].hasSysmemSegment) mem->globalHasSysmemSegment = true;
  }

  NCCLCHECKGOTO(ncclCalloc(&mem->lsaNumSegments, devr->lsaSize), ret, fail_mem);
  mem->lsaMinSize = size;
  mem->lsaMaxSize = size;
  for (int r = 0; r < devr->lsaSize; r++) {
    int rank = globalLsaTeamBaseIdx + r;
    mem->lsaNumSegments[r] = globalSegmentInfo[rank].numSegments;
    mem->lsaMinSize = std::min(mem->lsaMinSize, globalSegmentInfo[rank].totalSize);
    mem->lsaMaxSize = std::max(mem->lsaMaxSize, globalSegmentInfo[rank].totalSize);
  }

  // 在大地址空间中抢占偏移。使用 lsaMaxSize(LSA 各 rank 的最大值)以支持非对称大小。
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, mem->lsaMaxSize, devr->granularity, &bigOffset), ret,
                fail_mem);
  mem->bigOffset = bigOffset;

  // 把单播地址映射到 LSA 团队的扁平 VA 空间。
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, mem), ret, fail_mem_space);

  // 若调用者没有 VA，则使用 LSA 映射。
  if (mem->primaryAddr == nullptr) {
    mem->primaryAddr = (char*)devr->lsaFlatBase + devr->lsaSelf * devr->bigSize + mem->bigOffset;
  }

  // 把新内存与每个已存在的团队绑定。
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }

  if (devr->ginEnabled) {
    NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail_mem_space_teams);
  } else {
    // GIN 尚未启用时，默认使用单段。
    // 这会在 GIN 激活时由 ncclDevrCommCreateInternal 重新计算。
    mem->numGinSegments = 1;
  }

  // ginEnabled 在 ncclDevrCommCreateInternal 中设置，而 RMA 代理 场景下可能不会被调用，
  // 因此我们引入 rmaProxyEnabled 来跟踪 RMA 代理 是否已启用
  devr->rmaProxyEnabled =
    devr->nLsaTeams > 1 && comm->config.numRmaCtx > 0 && comm->globalRmaProxySupport && !ncclParamRMADisable();
  if (devr->rmaProxyEnabled && mem->maxGlobalNumSegments == 1) {
    NCCLCHECKGOTO(symMemoryRegisterRma(comm, mem), ret, fail_mem_space_teams);
  }

  // 加入内存列表。
  mem->next = devr->memHead;
  devr->memHead = mem;

  *outMem = mem;
  free(globalSegmentInfo);
  return ret;

fail_mem_space_teams:
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    symUnbindTeamMemory(comm, t, mem);
  }
fail_mem_space:
  ncclSpaceFree(&devr->bigSpace, bigOffset, mem->lsaMaxSize);
fail_mem:
  if (mem != nullptr) {
    free(mem->memHandles);
    free(mem->segmentSizes);
    free(mem->lsaNumSegments);
  }
  free(mem);
  free(globalSegmentInfo);
// 失败:
  return ret;
}

static void symMemoryDestroy(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  if (mem != nullptr) {
    struct ncclDevrState* devr = &comm->devrState;
    if (devr->ginEnabled && mem->ginSegmentInfos != nullptr) {
      for (int segment = 0; segment < mem->numGinSegments; segment++) {
        ncclGinDeregister(comm, mem->ginSegmentInfos[segment].ginHostWins);
      }
    }
    if (devr->rmaProxyEnabled && mem->maxGlobalNumSegments == 1) {
      ncclRmaProxyDeregister(comm, mem->rmaHostWins);
    }
    for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
      symUnbindTeamMemory(comm, t, mem);
    }
    for (int r = 0; r < devr->lsaSize; r++) {
      CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r * devr->bigSize + mem->bigOffset);
      for (int idx = 0; idx < mem->lsaNumSegments[r]; idx++) {
        CUdeviceptr tmpBase;
        size_t tmpBaseSize;
        CUCHECKIGNORE(cuMemGetAddressRange(&tmpBase, &tmpBaseSize, addr));
        CUCHECKIGNORE(cuMemUnmap(addr, tmpBaseSize));
        addr = addr + tmpBaseSize;
      }
    }

    ncclSpaceFree(&devr->bigSpace, mem->bigOffset, mem->lsaMaxSize);
    for (int segment = 0; segment < mem->numSegments; segment++) {
      CUCHECKIGNORE(cuMemRelease(mem->memHandles[segment]));
    }

    struct ncclDevrMemory** ptr = &devr->memHead;
    while (*ptr != mem) ptr = &(*ptr)->next;
    *ptr = mem->next; // Remove from list.

    free(mem->ginSegmentInfos);
    free(mem->lsaNumSegments);
    free(mem->segmentSizes);
    free(mem->memHandles);
    free(mem);
  }
}

static ncclResult_t symWindowTableInitOnce(struct ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  if (tableDev == nullptr) {
    // 在首次需要时创建。
    NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableDev, nullptr, stream));
    devr->windowTable = tableDev;
  }
  return ncclSuccess;
}

// 成功时，接管调用者持有的 mem 引用。
static ncclResult_t symWindowCreate(struct ncclComm* comm, struct ncclDevrMemory* mem, size_t memOffset, void* userPtr,
                                    size_t userSize, int winFlags, void* localReg, struct ncclWindow_vidmem** outWinDev,
                                    struct ncclDevrWindow** outWin, cudaStream_t stream) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrWindow* win;

  win = (struct ncclDevrWindow*)malloc(sizeof(struct ncclDevrWindow));
  memset(win, 0, sizeof(*win));
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;
  if (userPtr == nullptr) {
    // Null 表示调用者没有 VA，将使用 LSA 团队的扁平 VA 地址。
    win->userPtr = userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf * devr->bigSize) + mem->bigOffset;
    userAddr = reinterpret_cast<uintptr_t>(userPtr);
  } else {
    win->userPtr = userPtr;
  }

  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;
  winDevHost->mcOffset4K = win->bigOffset >> 12;
  winDevHost->stride4G = devr->bigSize >> 32;
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  winDevHost->ginOffset4K = memOffset >> 12;
  winDevHost->numSegments = mem->numGinSegments;
  for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
    winDevHost->ginWins[i] = mem->ginDevWins[i];
  }
  struct ncclSegmentWindow* segmentWindowsDev;
  NCCLCHECK(ncclDevrAllocAndPopulateSegmentWindows(devr, mem, stream, &segmentWindowsDev));
  winDevHost->ginMultiSegmentWins = segmentWindowsDev;
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));

  NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure devr->windowTable exists
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  while (true) {
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]),
                                cudaMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next, nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), cudaMemcpyHostToDevice,
                                stream));
    }
    tableDev = tableHost->next;
  }

  { // insert into winSorted[]
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
    struct ncclDevrWindowSorted winSort;
    winSort.userAddr = userAddr;
    winSort.size = userSize;
    winSort.win = win;
    listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);
  }

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}

static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclDevrWindow* winHost;

  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrWindow*)winDevHost->winHost;

  symMemoryDestroy(comm, winHost->memory);

  {
    struct ncclDevCommWindowTable* tableDev = devr->windowTable;
    while (true) {
      struct ncclDevCommWindowTable* tableHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret,
                      remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      tableDev = tableHost->next;
    }
  }

  if (winDevHost->ginMultiSegmentWins != nullptr) {
    NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDevHost->ginMultiSegmentWins, stream), ret, remove_winSorted);
  }

  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:
  {
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount,
                              reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.
    listRemove(devr->winSorted, &devr->winSortedCount, i);
  }
  // 从保存通信域指针的表中移除刚被释放的窗口
  {
    std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
    NCCLCHECKGOTO(ncclIntruAddressMapRemove(&ncclWindowMap, winDev), ret, fail);
  }

  free(winHost);
fail:
  return ret;
}

ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* userPtr, size_t userSize, int winFlags,
                                           ncclWindow_t* outWinDev) {
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  CUmemGenericAllocationHandle* memHandles = nullptr;
  size_t memOffset;
  struct ncclDevrMemory* mem = nullptr;
  cudaStream_t stream = nullptr;
  void* localRegHandle = nullptr;
  struct ncclDevrWindow* winHost = nullptr;
  int numSegments = 0;
  size_t offset = 0;
  bool hasSysmemSegment = false;
  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&captureMode));

  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    // 延迟对称 内核 初始化，直到至少存在一个带该标志的窗口。
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail_locReg);
  }

  // 获取底层 cumem 基址，以及 userPtr 所跨越的已映射物理段数量
  NCCLCHECKGOTO(ncclCuMemGetAddressRange(reinterpret_cast<CUdeviceptr>(userPtr), userSize, &memAddr, &memSize,
                                         &numSegments, &hasSysmemSegment),
                ret, fail_locReg);
  NCCLCHECKGOTO(ncclCalloc(&memHandles, numSegments), ret, fail_locReg);

  NCCLCHECKGOTO(ncclDevrCheckRegistrationSupport(userPtr, userSize, comm, hasSysmemSegment), ret, fail_locReg);

  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;
  if (memOffset % NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail_locReg;
  }

  // 保留所有句柄并校验各段的物理位置类型
  for (int segment = 0; segment < numSegments; segment++) {
    size_t baseSendSize;
    CUCHECK(cuMemGetAddressRange(nullptr, &baseSendSize, memAddr + offset));
    CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandles[segment], (void*)(reinterpret_cast<char*>(memAddr) + offset)),
                ret, fail_locReg);
    NCCLCHECKGOTO(ncclDevrValidateHandleLocationType(memHandles[segment], segment), ret, fail_locReg);
    offset += baseSendSize;
  }

  // 用 cumem 句柄换取 ncclDevrMemory*
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandles, numSegments, (void*)memAddr, memSize, winFlags, &mem,
                                hasSysmemSegment),
                ret, fail_locReg_memHandle);
  memset(memHandles, 0, numSegments * sizeof(*memHandles)); // symMemoryObtain took our reference

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_locReg_memHandle_mem);

  NCCLCHECKGOTO(symWindowCreate(comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, outWinDev, &winHost,
                                stream),
                ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference

  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // symWindowCreate 需要屏障。
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret,
                fail_locReg_memHandle_mem_stream_win);

  {
    std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
    // 直接在 winHost 上设置侵入式映射字段
    winHost->comm = comm;
    winHost->next = nullptr;  // Initialize next pointer
    // 由于窗口是唯一的、且只属于单个通信域，
    // 因此无需检查插入是否会覆盖已有条目
    NCCLCHECKGOTO(ncclIntruAddressMapInsert(&ncclWindowMap, *outWinDev, winHost), ret,
                  fail_locReg_memHandle_mem_stream_win);
    INFO(NCCL_ALLOC, "Inserted window %p into address map, ret=%d", *outWinDev, ret);
  }

  cudaStreamDestroy(stream);
  free(memHandles);
  cudaThreadExchangeStreamCaptureMode(&captureMode);
  return ret;

fail_locReg_memHandle_mem_stream_win:
  symWindowDestroy(comm, *outWinDev, stream);
  *outWinDev = nullptr;
  cudaStreamSynchronize(stream);
fail_locReg_memHandle_mem_stream:
  cudaStreamDestroy(stream);
fail_locReg_memHandle_mem:
  symMemoryDestroy(comm, mem);
fail_locReg_memHandle:
  for (int idx = 0; idx < numSegments; idx++) {
    if (memHandles[idx] != 0x0ULL) CUCHECKIGNORE(cuMemRelease(memHandles[idx]));
  }
  free(memHandles);
fail_locReg:
  ncclCommDeregister(comm, localRegHandle);
fail:
  cudaThreadExchangeStreamCaptureMode(&captureMode);
  *outWinDev = nullptr;
  return ret;
}

static ncclResult_t deepCopyDevCommRequirements(struct ncclDevCommRequirements const* src,
                                                struct ncclDevCommRequirements** dst) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevResourceRequirements** dstRes;
  struct ncclTeamRequirements** dstTeam;

  NCCLCHECK(ncclCalloc(dst, 1));
  **dst = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

  // 先整体拷贝结构体、稍后更新链表。出于向后兼容考虑，源结构体可能
  // 实际比类型应有的大小要小。
  memcpy(*dst, src, src->size);

  dstRes = &(*dst)->resourceRequirementsList;
  for (struct ncclDevResourceRequirements* rr = src->resourceRequirementsList; rr != nullptr; rr = rr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstRes, 1), ret, fail);
    (*dstRes)->bufferSize = rr->bufferSize;
    (*dstRes)->bufferAlign = rr->bufferAlign;
    (*dstRes)->outBufferHandle = rr->outBufferHandle;
    dstRes = &(*dstRes)->next;
  }

  dstTeam = &(*dst)->teamRequirementsList;
  for (struct ncclTeamRequirements* tr = src->teamRequirementsList; tr != nullptr; tr = tr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstTeam, 1), ret, fail);
    (*dstTeam)->team = tr->team;
    (*dstTeam)->multimem = tr->multimem;
    (*dstTeam)->outMultimemHandle = tr->outMultimemHandle;
    dstTeam = &(*dstTeam)->next;
  }

exit:
  return ret;
fail:
  freeDevCommRequirements(*dst);
  *dst = nullptr;
  goto exit;
}

void freeDevCommRequirements(struct ncclDevCommRequirements* reqs) {
  if (reqs) {
    while (reqs->resourceRequirementsList) {
      struct ncclDevResourceRequirements* rr_next = reqs->resourceRequirementsList->next;
      free(reqs->resourceRequirementsList);
      reqs->resourceRequirementsList = rr_next;
    }

    while (reqs->teamRequirementsList) {
      struct ncclTeamRequirements* tr_next = reqs->teamRequirementsList->next;
      free(reqs->teamRequirementsList);
      reqs->teamRequirementsList = tr_next;
    }

    free(reqs);
  }
}

bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs) {
  bool requestedGinResources = reqs->ginSignalCount > 0 || reqs->ginCounterCount > 0 || reqs->barrierCount > 0 ||
                               reqs->railGinBarrierCount > 0 || reqs->worldGinBarrierCount > 0;

  struct ncclDevResourceRequirements* node = reqs->resourceRequirementsList;
  while (!requestedGinResources && node != nullptr) {
    requestedGinResources = node->ginSignalCount > 0 || node->ginCounterCount > 0;
    node = node->next;
  }

  return requestedGinResources;
}

#if defined(NCCL_OS_LINUX)
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
static void ncclDevCommGdakiDump(void* handle) {
  struct ncclGinGdakiGPUContext ctx;
  if (cudaMemcpy(&ctx, handle, sizeof(ctx), cudaMemcpyDeviceToHost) == cudaSuccess) {
    printf("    GDAKI qp %p companion qp %p sink buffer lkey %x\n", ctx.gdqp, ctx.companion_gdqp, ctx.sink_buffer_lkey);
    printf("    GDAKI counters %p rkeys %p lkey %x offset %d\n", ctx.counters_table.buffer, ctx.counters_table.rkeys,
           ctx.counters_table.lkey, ctx.counters_table.offset);
    printf("    GDAKI signals  %p rkeys %p lkey %x offset %d\n", ctx.signals_table.buffer, ctx.signals_table.rkeys,
           ctx.signals_table.lkey, ctx.signals_table.offset);
  }
}

#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
static void ncclDevCommProxyDump(void* handle) {
  ncclGinProxyGpuCtx_t ctx;
  if (cudaMemcpy(&ctx, handle, sizeof(ctx), cudaMemcpyDeviceToHost) == cudaSuccess) {
    printf("    PROXY nranks %d queue size %d queues %p\n", ctx.nranks, ctx.queueSize, ctx.queues);
    printf("    PROXY pis %p cis %p counters %p signals %p\n", ctx.pis, ctx.cis, ctx.counters, ctx.signals);
  }
}
#elif defined(NCCL_OS_WINDOWS)
static void ncclDevCommGdakiDump(void* handle) {
  printf("    GDAKI handle %p (detailed dump not available on Windows)\n", handle);
}
static void ncclDevCommProxyDump(void* handle) {
  printf("    PROXY handle %p (detailed dump not available on Windows)\n", handle);
}
#endif /* !NCCL_OS_WINDOWS */

void ncclDevCommDump(struct ncclDevComm* devComm) {
  printf("**** Dev Comm Dump %p ****\n", devComm);
  printf(" Rank %d/%d CPC32 %d\n", devComm->rank, devComm->nRanks, devComm->nRanks_rcp32);
  printf(" LSA Rank %d/%d CPC32 %d\n", devComm->lsaRank, devComm->lsaSize, devComm->lsaSize_rcp32);
  printf("\n");
  printf(" GIN\n");
  printf("  Mode %s\n", devComm->ginConnectionsRailed ? "Rail" : "Full");
  printf("  Connections %d\n", devComm->ginConnectionCount);
  for (int c = 0; c < devComm->ginConnectionCount; c++) {
    printf("   [%d] %d %p\n", c, devComm->ginNetDeviceTypes[c], devComm->ginHandles[c]);
    if (devComm->ginNetDeviceTypes[c] == NCCL_GIN_TYPE_GDAKI) ncclDevCommGdakiDump(devComm->ginHandles[c]);
    if (devComm->ginNetDeviceTypes[c] == NCCL_GIN_TYPE_PROXY) ncclDevCommProxyDump(devComm->ginHandles[c]);
  }
  printf("  Signals  %d shadows %p\n", devComm->ginSignalCount, devComm->ginSignalShadows);
  printf("  Contexts %d\n", devComm->ginContextCount);
  printf("\n");
  printf(" Abort flag %p\n", devComm->abortFlag);
  printf(" LSA Barriers count %d handle %d\n", devComm->lsaBarrier.nBarriers, devComm->lsaBarrier.bufHandle);
  printf(" Hybrid Barriers count %d LSA handle %d GIN Rail Barrier signal0 %d GIN World Barrier signal0 %d\n",
         devComm->hybridLsaBarrier.nBarriers, devComm->hybridLsaBarrier.bufHandle,
         devComm->hybridRailGinBarrier.signal0, devComm->hybridWorldGinBarrier.signal0);
  printf(" GIN Rail Barrier signal0 %d\n", devComm->railGinBarrier.signal0);
  printf(" GIN World Barrier signal0 %d\n", devComm->worldGinBarrier.signal0);
}

ncclResult_t ncclDevrCommCreateInternal(struct ncclComm* comm, struct ncclDevCommRequirements* reqs,
                                        struct ncclDevComm* outDevComm, bool isInternal,
                                        struct ncclDevCommCompat* devCompat) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclTeam world = ncclTeamWorld(comm);
  struct ncclTeam lsa = ncclTeamInnerFactor(world, devr->lsaSize);
  bool ginActivated = false;
  struct ncclDevrTeam* tmLsa;
  size_t bufSizeTotal;
  int nGinContexts = reqs->ginContextCount;
  int ginSignalTotal = 0, ginCounterTotal = 0;
  struct ncclDevResourceRequirements* resReqsHead = reqs->resourceRequirementsList;
  struct ncclDevResourceRequirements lsaBarReq;
  struct ncclDevResourceRequirements hybridLsaBarrierReq;
  cudaStream_t stream = nullptr;
  struct ncclDevResourceRequirements railGinBarrierReq;
  struct ncclDevResourceRequirements hybridRailGinBarrierReq;
  struct ncclDevResourceRequirements hybridWorldGinBarrierReq;
  struct ncclDevResourceRequirements worldGinBarrierReq;
  CUmemGenericAllocationHandle memHandle = 0x0;
  struct ncclDevrMemory* mem = nullptr;
  struct ncclDevrWindow* win = nullptr;
  struct ncclWindow_vidmem* winHost = nullptr;
  size_t ginSignalShadowsOffset = 0;
  void* outDevCommPreserve = nullptr;
  struct ncclDevComm outDevCommTmp;
  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;

  // 本函数始终操作 ncclDevResourceRequirements 结构的当前版本，因为
  // 有 deepCopyDevCommRequirements() 函数，所以无需版本检查。reqs 中的数据也可
  // 被本函数修改，这也没问题，因为它不是用户提供的结构体。
  ncclGinConnectionType_t requestedConnectionType = reqs->ginConnectionType;

  if (reqs->ginForceEnable) {
    INFO(NCCL_INIT, "ginForceEnable set to true, defaulting ginConnectionType to NCCL_GIN_CONNECTION_FULL");
    INFO(NCCL_INIT, "ginForceEnable is being deprecated in favor of explicitly setting ginConnectionType!");
    requestedConnectionType = NCCL_GIN_CONNECTION_FULL;
  }

  bool requestedGinResources = ncclGinResourcesRequested(reqs);
  if (requestedGinResources && requestedConnectionType == NCCL_GIN_CONNECTION_NONE) {
    WARN("User requested GIN resources but did not request GIN to be enabled!");
    return ncclInvalidArgument;
  }

  if (requestedConnectionType != NCCL_GIN_CONNECTION_NONE) {
    if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE) {
      WARN("User requested GIN but not all ranks in the communicator support GIN");
      return ncclInvalidArgument;
    }
    if (requestedConnectionType == NCCL_GIN_CONNECTION_FULL) {
      if (comm->globalGinSupport == NCCL_GIN_CONNECTION_RAIL) {
        WARN("User requested GIN connection type NCCL_GIN_CONNECTION_FULL but the communicator supports only "
             "NCCL_GIN_CONNECTION_RAIL");
        return ncclInvalidArgument;
      }
    }

    ginActivated = !devr->ginEnabled;
    devr->ginEnabled = true;
  }

  if (reqs->worldGinBarrierCount > 0 && requestedConnectionType == NCCL_GIN_CONNECTION_RAIL) {
    WARN("Cannot create worldGinBarrier with NCCL_GIN_CONNECTION_RAIL.");
    return ncclInvalidArgument;
  }

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&captureMode));

  if (ginActivated) {
    NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, fail);
    // 用 GIN 注册所有已存在的内存。稍后、待有流可用时再更新窗口。
    // 
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail);
    }
  }

  // 若出于兼容保留了一个拷贝回调，我们用临时缓冲区存放 devComm，完成后
  // 再让回调把数据拷过去。
  if (devCompat && devCompat->devCommCopyNewToOld) {
    outDevCommPreserve = outDevComm;
    outDevComm = &outDevCommTmp;
  }

  memset(outDevComm, 0, sizeof(*outDevComm));
  outDevComm->magic = NCCL_API_MAGIC;
  outDevComm->version = reqs->version;
  outDevComm->rank = comm->rank;
  outDevComm->nRanks = comm->nRanks;
  outDevComm->nRanks_rcp32 = idivRcp32(comm->nRanks);
  outDevComm->lsaRank = devr->lsaSelf;
  outDevComm->lsaSize = devr->lsaSize;
  outDevComm->lsaSize_rcp32 = idivRcp32(devr->lsaSize);
  outDevComm->ginConnectionsRailed = comm->sharedRes->ginState.ginConnectionType == NCCL_GIN_CONNECTION_RAIL;
  outDevComm->ginContextsRailed = requestedConnectionType == NCCL_GIN_CONNECTION_RAIL;
  if (isInternal) outDevComm->abortFlag = comm->abortFlagDev;

  NCCLCHECKGOTO(symTeamObtain(comm, lsa, /*multicast=*/reqs->lsaMultimem, &tmLsa), ret, fail);
  outDevComm->lsaMultimem.mcBasePtr = tmLsa->mcBasePtr;

  {
    struct ncclTeamRequirements* tr = reqs->teamRequirementsList;
    while (tr != nullptr) {
      if (tr->multimem) {
        struct ncclDevrTeam* tm;
        NCCLCHECKGOTO(symTeamObtain(comm, tr->team, tr->multimem, &tm), ret, fail);
        if (tr->outMultimemHandle != nullptr) tr->outMultimemHandle->mcBasePtr = tm->mcBasePtr;
      }
      tr = tr->next;
    }
  }

  resReqsHead = reqs->resourceRequirementsList;

  // 为混合(hybrid)屏障初始化资源
  ncclLsaBarrierCreateRequirement(lsa, reqs->barrierCount, &outDevComm->hybridLsaBarrier, &hybridLsaBarrierReq);
  hybridLsaBarrierReq.next = resReqsHead;
  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm), reqs->barrierCount, &outDevComm->hybridRailGinBarrier,
                                  &hybridRailGinBarrierReq);
  hybridRailGinBarrierReq.next = &hybridLsaBarrierReq;
  ncclGinBarrierCreateRequirement(comm, ncclTeamWorld(comm),
                                  requestedConnectionType == NCCL_GIN_CONNECTION_RAIL ? 0 : reqs->barrierCount,
                                  &outDevComm->hybridWorldGinBarrier, &hybridWorldGinBarrierReq);
  hybridWorldGinBarrierReq.next = &hybridRailGinBarrierReq;
  resReqsHead = &hybridWorldGinBarrierReq;

  ncclLsaBarrierCreateRequirement(lsa, reqs->lsaBarrierCount, &outDevComm->lsaBarrier, &lsaBarReq);
  lsaBarReq.next = resReqsHead;
  resReqsHead = &lsaBarReq;

  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm), reqs->railGinBarrierCount, &outDevComm->railGinBarrier,
                                  &railGinBarrierReq);
  railGinBarrierReq.next = resReqsHead;
  resReqsHead = &railGinBarrierReq;

  ncclGinBarrierCreateRequirement(comm, ncclTeamWorld(comm), reqs->worldGinBarrierCount, &outDevComm->worldGinBarrier,
                                  &worldGinBarrierReq);
  worldGinBarrierReq.next = resReqsHead;
  resReqsHead = &worldGinBarrierReq;

  {
    struct ncclDevResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0;
    ginSignalTotal = reqs->ginSignalCount;
    ginCounterTotal = reqs->ginCounterCount;
    while (rr != nullptr) {
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      if (rr->outBufferHandle != nullptr) *rr->outBufferHandle = bufSizeTotal / 128;
      if (rr->outGinSignalStart != nullptr) *rr->outGinSignalStart = ginSignalTotal;
      if (rr->outGinCounterStart != nullptr) *rr->outGinCounterStart = ginCounterTotal;
      bufSizeTotal += rr->bufferSize;
      ginSignalTotal += rr->ginSignalCount;
      ginCounterTotal += rr->ginCounterCount;
      rr = rr->next;
    }
    bufSizeTotal = alignUp(bufSizeTotal, 128);
    ginSignalShadowsOffset = bufSizeTotal;
    bufSizeTotal += nGinContexts * ginSignalTotal * sizeof(uint64_t); // include signal shadows
    bufSizeTotal = alignUp(bufSizeTotal, devr->granularity);
  }

  if (devr->ginEnabled) {
    reqs->ginSignalCount = ginSignalTotal;
    reqs->ginCounterCount = ginCounterTotal;
    NCCLCHECK(ncclGinDevCommSetup(comm, reqs, outDevComm));
  }

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  if (ginActivated) {
    // 现在更新所有已有窗口中的 GIN 句柄。内存的注册已在前面完成。
    for (int i = 0; i < devr->winSortedCount; i++) {
      struct ncclDevrWindow* win = devr->winSorted[i].win;
      struct ncclWindow_vidmem* winHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream);
      winHost->ginOffset4K = (win->bigOffset - win->memory->bigOffset) >> 12;
      for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
        winHost->ginWins[i] = win->memory->ginDevWins[i];
      }
      winHost->numSegments = win->memory->numGinSegments;

      NCCLCHECKGOTO(ncclDevrReplaceSegmentWindowsIfNeeded(devr, win->memory, winHost, stream), ret, fail_stream);
      CUDACHECKGOTO(cudaMemcpyAsync(win->vidmem, winHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice,
                                    stream),
                    ret, fail_stream);
    }
  }

  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail_stream); // ensure devr->windowTable exists
  outDevComm->windowTable = devr->windowTable;

  if (bufSizeTotal == 0) {
    outDevComm->resourceWindow = nullptr;
    outDevComm->resourceWindow_inlined = {};
  } else {
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.requestedHandleTypes = ncclCuMemHandleType;
    // 我们必须假定：只要 GIN 有可能启用，将来就可能被请求，
    // 即便在单节点上也不例外。
    memProp.allocFlags.gpuDirectRDMACapable = comm->sharedRes->ginState.ncclGin != nullptr ? 1 : 0;
    memProp.location.id = comm->cudaDev;

    CUCHECKGOTO(cuMemCreate(&memHandle, bufSizeTotal, &memProp, 0), ret, fail_stream);

    NCCLCHECKGOTO(symMemoryObtain(comm, &memHandle, 1, NULL, bufSizeTotal, /*winFlags=*/0, &mem), ret, fail_stream_mem);
    memHandle = 0x0; // Reference given to symMemoryObtain

    NCCLCHECKGOTO(symWindowCreate( // Requires world barrier afterward.
                    comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
                    /*localReg=*/nullptr, &outDevComm->resourceWindow, &win, stream),
                  ret, fail_stream_mem);
    mem = nullptr; // Reference given to symWindowCreate
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream_mem_win);
    outDevComm->resourceWindow_inlined.lsaFlatBase = winHost->lsaFlatBase;
    outDevComm->resourceWindow_inlined.stride4G = winHost->stride4G;
    outDevComm->resourceWindow_inlined.mcOffset4K = winHost->mcOffset4K;
    outDevComm->ginSignalShadows =
      (uint64_t*)add4G((char*)winHost->lsaFlatBase + ginSignalShadowsOffset, winHost->lsaRank * winHost->stride4G);

    CUDACHECKGOTO(cudaMemsetAsync(win->userPtr, 0, bufSizeTotal, stream), ret, fail_stream_mem_win);
  }

  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_stream_mem_win);

  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_stream_mem_win);
  CUDACHECKGOTO(cudaStreamDestroy(stream), ret, fail_stream_mem_win);

  // ncclDevCommDump(outDevComm);
  if (outDevCommPreserve) {
    NCCLCHECKGOTO(devCompat->devCommCopyNewToOld(comm, outDevCommPreserve, outDevComm), ret, fail_stream_mem_win);
  }
  cudaThreadExchangeStreamCaptureMode(&captureMode);
  return ret;

fail_stream_mem_win:
  symWindowDestroy(comm, win->vidmem, stream);
  cudaStreamSynchronize(stream);
fail_stream_mem:
  if (memHandle != 0x0) CUCHECKIGNORE(cuMemRelease(memHandle));
  symMemoryDestroy(comm, mem);
fail_stream:
  cudaStreamDestroy(stream);
fail:
  cudaThreadExchangeStreamCaptureMode(&captureMode);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win,
         int winFlags);
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(win, __func__, "win"));
  *win = nullptr;
  if (buff == nullptr || size <= 0) {
    WARN("invalid pointer %p / size %zu", buff, size);
    return ncclInvalidArgument;
  }

  if (!comm->symmetricSupport) {
    return ncclSuccess;
  }

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrRegTask* task;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = buff;
  task->userSize = size;
  task->winFlags = winFlags;
  task->outWinDev = win;
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclCommWindowDeregister(struct ncclComm* comm, struct ncclWindow_vidmem* winDev) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  cudaStream_t stream;
  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;

  if (winDev == nullptr) goto exit;

  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&captureMode));
  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_dev);
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream);
fail_dev_stream:
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
fail_dev:
  cudaSetDevice(saveDev);
fail:
  cudaThreadExchangeStreamCaptureMode(&captureMode);
exit:
  return ret;
}

ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin) {
  struct ncclDevrState* devr = &comm->devrState;
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
  if (0 < i && (userAddr - devr->winSorted[i - 1].userAddr < devr->winSorted[i - 1].size)) {
    *outWin = devr->winSorted[i - 1].win;
  } else {
    *outWin = nullptr;
  }
  return ncclSuccess;
}

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win) {
  return win != NULL && win->memory->maxGlobalNumSegments > 1;
}

bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win) {
  return win != NULL && win->memory->globalHasSysmemSegment;
}

// 若编译版本高于运行时版本则返回 ncclInvalidUsage，
// 且未设置 NCCL_ENABLE_VERSION_CHECK=0
static ncclResult_t getNcclVersionCompat(int compiledVersion, struct ncclDevCommCompat** devCompatPtr) {
  *devCompatPtr = nullptr;

  if (compiledVersion > NCCL_VERSION_CODE && ncclParamEnableVersionCheck()) {
    char compiledBuf[16], runtimeBuf[16];
    WARN("NCCL library is too old. This application was compiled with NCCL version %s, but is running with NCCL "
         "library version %s.",
         ncclVersionToString(compiledVersion, compiledBuf, sizeof(compiledBuf)),
         ncclVersionToString(NCCL_VERSION_CODE, runtimeBuf, sizeof(runtimeBuf)));
    return ncclInvalidUsage;
  }

  struct ncclDevCommCompat* devCompat = nullptr;
  for (int i = 0; devCommCompat[i]; i++) {
    if (compiledVersion >= devCommCompat[i]->minVersion && compiledVersion <= devCommCompat[i]->maxVersion) {
      devCompat = devCommCompat[i];
      break;
    }
  }
  if (devCompat == nullptr) {
    char compiledBuf[16], runtimeBuf[16];
    WARN("NCCL library is not backwards compatible. This application was compiled with NCCL version %s, but is running "
         "with NCCL library version %s.",
         ncclVersionToString(compiledVersion, compiledBuf, sizeof(compiledBuf)),
         ncclVersionToString(NCCL_VERSION_CODE, runtimeBuf, sizeof(runtimeBuf)));
    return ncclInvalidUsage;
  }
  *devCompatPtr = devCompat;

  return ncclSuccess;
}

void ncclDevCommCopyLsaData(void* dstRankPtr, void const* srcRankPtr) {
  memcpy(dstRankPtr, srcRankPtr, offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank));
}

NCCL_API(ncclResult_t, ncclCommQueryProperties, ncclComm_t, ncclCommProperties_t*);
ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(props, __func__, "props"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  if (props->magic != NCCL_API_MAGIC) {
    WARN("Cannot get communicator properties: ncclCommProperties_t argument must be initialized via "
         "NCCL_COMM_PROPERTIES_INITIALIZER");
    return ncclInvalidUsage;
  }

  struct ncclDevCommCompat* devCompat = nullptr;
  NCCLCHECK(getNcclVersionCompat(props->version, &devCompat));

  props->rank = comm->rank;
  props->nRanks = comm->nRanks;
  props->cudaDev = comm->cudaDev;
  props->nvmlDev = comm->nvmlDev;
  props->deviceApiSupport = comm->symmetricSupport;
  // NVLS 多播在 clique 之间不可用
  props->multimemSupport = comm->nvlsSupport && !comm->p2pCrossClique;

  if (props->version > NCCL_VERSION(2, 29, 3)) {
    props->hostRmaSupport = comm->hostRmaSupport;
    NCCLCHECK(ncclGetGinType(comm, &props->ginType));
    NCCLCHECK(ncclGetRailedGinType(comm, &props->railedGinType));

    // 倾向于直接调用 ncclDevrInitOnce 而非 ncclTeam* 函数，因为
    // 这样可以把 ncclDevrInitOnce 的结果传播回调用者。
    NCCLCHECK(ncclDevrInitOnce(comm));
    props->nLsaTeams = comm->devrState.nLsaTeams;
  }

  if (devCompat->commPropertiesFilter) {
    NCCLCHECK(devCompat->commPropertiesFilter(comm, props));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclDevCommCreate, ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
         ncclDevComm_t* outDevComm);
ncclResult_t ncclDevCommCreate(ncclComm_t comm, struct ncclDevCommRequirements const* reqs,
                               struct ncclDevComm* outDevComm) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(reqs, __func__, "reqs"));
  if (reqs->magic != NCCL_API_MAGIC) {
    WARN("Cannot create device communicator: ncclDevCommRequirements_t argument must be initialized via "
         "NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER");
    return ncclInvalidUsage;
  }

  struct ncclDevCommCompat* devCompat = nullptr;
  NCCLCHECK(getNcclVersionCompat(reqs->version, &devCompat));

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrCommCreateTask* task = nullptr;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  // reqs 必须深拷贝到任务中，以便后台线程能安全访问
  NCCLCHECKGOTO(deepCopyDevCommRequirements(reqs, &task->reqs), ret, fail);
  if (devCompat->devCommRequirementsFilter) {
    NCCLCHECKGOTO(devCompat->devCommRequirementsFilter(comm, task->reqs), ret, fail);
  }
  task->outDevComm = outDevComm;
  task->devCompat = devCompat;
  ncclIntruQueueEnqueue(&comm->devrState.commCreateTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  free(task);
  goto exit;
}

NCCL_API(ncclResult_t, ncclDevCommDestroy, ncclComm_t comm, ncclDevComm_t const* devComm);
ncclResult_t ncclDevCommDestroy(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(devComm, __func__, "devComm"));
  int saveDev;
  ncclResult_t ret = ncclSuccess;
  struct ncclDevCommCompat* devCompat = nullptr;
  ncclDevComm_t devCommTmp;

  if (devComm->magic == NCCL_API_MAGIC) {
    NCCLCHECK(getNcclVersionCompat(devComm->version, &devCompat));
  } else {
    // 无版本号的 devComm——必须是 v22902 或 v22907。v22902 同时处理二者的 devCommDestroy。
    devCompat = &ncclDevCommCompat_v22902;
  }
  if (devCompat->devCommCopyOldToNew) {
    memset(&devCommTmp, 0, sizeof(devCommTmp));
    devCommTmp.magic = NCCL_API_MAGIC;
    devCommTmp.version = NCCL_VERSION_CODE;
    NCCLCHECK(devCompat->devCommCopyOldToNew(comm, &devCommTmp, devComm));
    devComm = &devCommTmp;
  }

  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev)); // This is needed at least for cuMem memory freeing in GDAKI

  if (devComm->resourceWindow != nullptr) {
    NCCLCHECKGOTO(ncclCommWindowDeregister(comm, devComm->resourceWindow), ret, end);
  }
  if (devComm->ginContextCount) {
    NCCLCHECKGOTO(ncclGinDevCommFree(comm, devComm), ret, end);
  }

end:
  cudaSetDevice(saveDev);
  return ret;
}

NCCL_API(ncclResult_t, ncclWinGetUserPtr, ncclComm_t comm, ncclWindow_t win, void** outUserPtr);
ncclResult_t ncclWinGetUserPtr(struct ncclComm* comm, struct ncclWindow_vidmem* win, void** outUserPtr) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(outUserPtr, __func__, "outUserPtr"));

  if (!comm->symmetricSupport) {
    INFO(NCCL_INIT, "Symmetric registration is not supported in this communicator.");
    *outUserPtr = nullptr;
    return ncclSuccess;
  }

  NCCLCHECK(PtrCheck(win, __func__, "win"));

  struct ncclDevrWindow* winHost = nullptr;
  struct ncclWindow_vidmem* winDevHost = nullptr;
  NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, win, &winDevHost));

  winHost = (struct ncclDevrWindow*)winDevHost->winHost;
  if (winHost == nullptr) {
    WARN("window has a NULL user pointer");
    return ncclInternalError;
  }

  *outUserPtr = winHost->userPtr;
  return ncclSuccess;
}

ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm* comm, int peerWorldRank, int* peerLsaRank) {
  ncclTeam_t worldTeam = ncclTeamWorld(comm);
  ncclTeam_t lsaTeam = ncclTeamLsa(comm);
  if (!ncclTeamRankIsMember(lsaTeam, worldTeam, peerWorldRank)) {
    WARN("ncclDevrWorldToLsaRank: world rank %d is not a member of the LSA team", peerWorldRank);
    return ncclInternalError;
  }
  *peerLsaRank = ncclTeamRankToTeam(lsaTeam, worldTeam, peerWorldRank);
  return ncclSuccess;
}

// 获取在另一个 lsa rank 的对称内存窗口中对应的指针
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank,
                                   void** outPtr) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  struct ncclDevrState* devr = &comm->devrState;

  // 校验 lsaRank 在界限内
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    return ncclInvalidArgument;
  }

  // 校验 偏移 在界限内
  if (offset < 0 || offset >= winHost->size) {
    return ncclInvalidArgument;
  }

  // 为指定的 lsa rank 计算加上 偏移 后的地址
  *outPtr = (void*)((uintptr_t)devr->lsaFlatBase + lsaRank * devr->bigSize + winHost->bigOffset + offset);
  return ncclSuccess;
}

// 获取特定上下文的 RMA 设备窗口句柄
void* ncclDevrGetRmaWin(struct ncclDevrWindow* winHost, int ctx) {
  if (winHost == nullptr || winHost->memory == nullptr) {
    return nullptr;
  }
  if (ctx < 0 || ctx >= NCCL_GIN_MAX_CONNECTIONS) {
    return nullptr;
  }
  return winHost->memory->rmaHostWins[ctx];
}

// 获取给定团队的多播地址
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset,
                                     struct ncclTeam lsaTeam, void** outPtr) {
  if (winHost == nullptr || outPtr == nullptr) return ncclInternalError;

  if (!comm->nvlsSupport) {
    WARN("Multimem pointer requested but system does not support multimem.");
    return ncclInvalidUsage;
  }

  bool multimem = true;
  struct ncclDevrTeam* tm;
  NCCLCHECK(symTeamObtain(comm, lsaTeam, multimem, &tm));

  // 返回本团队带 偏移 的基址多播地址
  *outPtr = (void*)((uintptr_t)tm->mcBasePtr + winHost->bigOffset + offset);
  return ncclSuccess;
}

static ncclResult_t findCommAndHostWindowFromDeviceWindow(ncclWindow_t devWindow, ncclComm_t* foundComm,
                                                          ncclDevrWindow** hostWindow) {
  struct ncclDevrWindow* winHost = nullptr;
  std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
  NCCLCHECK(ncclIntruAddressMapFind(&ncclWindowMap, devWindow, &winHost));
  if (winHost == nullptr) {
    WARN("Could not find communicator matching window %p (map hbits=%d count=%d)", devWindow, ncclWindowMap.base.hbits,
         ncclWindowMap.base.count);
    return ncclInvalidArgument;
  }

  *foundComm = winHost->comm;
  *hostWindow = winHost;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetMultimemDevicePointer, ncclWindow_t window, size_t offset, ncclMultimemHandle multimem,
         void** outPtr);
ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle multimem,
                                          void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));
  if (multimem.mcBasePtr == nullptr) {
    WARN("MCBasePtr %p needs to be valid.", multimem.mcBasePtr);
    return ncclInvalidArgument;
  }

  ncclComm_t comm = nullptr;
  struct ncclDevrWindow* winHost = nullptr;

  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  if (!comm->nvlsSupport) {
    *outPtr = nullptr;
    return ncclSuccess;
  }
  *outPtr = (void*)((uintptr_t)multimem.mcBasePtr + winHost->bigOffset + offset);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetLsaMultimemDevicePointer, ncclWindow_t window, size_t offset, void** outPtr);
ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  ncclComm_t comm = nullptr;
  struct ncclDevrWindow* winHost = nullptr;

  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  if (comm->nvlsSupport == 0) {
    *outPtr = nullptr;
    return ncclSuccess;
  }

  NCCLCHECK(ncclDevrGetLsaTeamPtrMC(comm, winHost, offset, ncclTeamLsa(comm), outPtr));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetLsaDevicePointer, ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  ncclComm_t comm = nullptr;
  struct ncclDevrState* devr;
  struct ncclDevrWindow* winHost = nullptr;

  // 获取设备窗口的主机侧版本
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  devr = &comm->devrState;
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    WARN("The provided lsaRank %d is not in the valid lsaSize of [0,%d] for the provided window %p.", lsaRank,
         devr->lsaSize, window);
    return ncclInvalidArgument; // In this case the user should know what the lsa size is.
  }

  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetPeerDevicePointer, ncclWindow_t window, size_t offset, int peer, void** outPtr);
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  ncclComm_t comm = nullptr;
  struct ncclDevrState* devr;
  struct ncclDevrWindow* winHost = nullptr;
  int lsaRank;
  ncclTeam_t worldTeam;
  ncclTeam_t lsaTeam;

  // 获取设备窗口的主机侧版本
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));
  // 校验对端 rank 在界限内
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("peer %d is not within valid range of ranks %d.", peer, comm->nRanks);
    return ncclInvalidArgument;
  }

  devr = &comm->devrState;
  worldTeam = ncclTeamWorld(comm);
  lsaTeam = ncclTeamLsa(comm);

  // 把全局 world rank 转换为 LSA 团队 rank
  lsaRank = ncclTeamRankToTeam(lsaTeam, worldTeam, peer);

  // 校验转换后的 LSA rank 在界限内
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    // 若对端不可达则返回 nullptr。与设备侧行为一致
    *outPtr = nullptr;
    return ncclSuccess;
  }

  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr));

  return ncclSuccess;
}
////////////////////////////////////////////////////////////////////////////////

// 找到严格大于 arg 的最小下标。
template <typename Obj, typename Key>
static int listFindSortedLub(Key Obj::* key, Obj* sorted, int count, Key arg) {
  int lo = 0, hi = count;
  while (lo + 16 < hi) {
    int i = (lo + hi) / 2;
    if (sorted[i].*key <= arg) lo = i + 1;
    else hi = i;
  }
  int i = lo;
  while (i < hi && sorted[i].*key <= arg) i++;
  return i;
}

template <typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val) {
  if (*capacity < *count + 1) {
    *capacity *= 2;
    if (*capacity == 0) *capacity = 16;
    *list = (Obj*)realloc(*list, (*capacity) * sizeof(Obj));
  }
  for (int j = *count; j != index; j--) {
    (*list)[j] = (*list)[j - 1];
  }
  (*list)[index] = val;
  *count += 1;
}

template <typename Obj>
static void listRemove(Obj* list, int* count, int index) {
  for (int i = index; i + 1 < *count; i++) {
    list[i] = list[i + 1];
  }
  *count -= 1;
}
