/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * enqueue.cc —— NCCL 调度/启动层（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 本文件在 AllReduce 全链路中的定位：
 *   graph(拓扑构建) ──► 【enqueue.cc 调度/启动层】 ──► device kernel(计算)
 *
 * 职责：当用户调用 ncclAllReduce(...) 时，collectives.cc 把参数打包成 struct
 * ncclInfo，调用 ncclEnqueueCheck()。本文件负责把这一次 collective 操作
 * “组织成若干 kernel 启动任务”：
 *   1) 参数校验与 group 语义（ncclEnqueueCheck / ncclGroupStart-End）。
 *   2) 把 task 暂存进 comm->planner（taskAppend / collTaskAppend）。
 *   3) group 结束时计算算法(algo: ring/tree)与协议(protocol: LL/LL128/Simple)，
 *      并把数据切分到多个 channel。
 *   4) 为每 channel 生成 ncclDevWorkColl、选择 kernel(devFuncId)、设置 send/recv
 *      连接（来自 graph 阶段建好的 channel 拓扑）。
 *   5) 把 proxy 任务(proxyOp)入队，供后台 proxy 线程执行网络/跨卡搬运。
 *   6) 真正启动 CUDA kernel（ncclLaunchKernel）以及 CUDA graph 处理。
 *
 * 本仓库仅保留 AllReduce：凡涉及 P2P/Bcast/CE/RMA/NVLS/CollNet 的分支均为
 * “死代码/不会走到的分支”，已在相应处标注说明。
 * ============================================================================
 */

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "channel.h"
#include "cudawrap.h"
#include "profiler.h"
#include "transport.h"
#include "register_inline.h"
#include "ce_coll.h"
#include "nvtx.h"
#include "scheduler.h"
#include "compiler.h"
#include "rma/rma.h"

#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64
#include <cassert>
#include <cfloat> // FLT_MAX

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);
NCCL_PARAM(AllgathervEnable, "ALLGATHERV_ENABLE", 1);
NCCL_PARAM(SymCeThreshold, "SYM_CE_THRESHOLD", 8 * 1024 * 1024);

// 初始化本设备上的所有 CUDA 内核：探测每个 内核 需要的 driver 版本、
// 共享内存上限，并设置 L1/shared 分配策略(carveout)。内核 启动前需保证
// 已就绪——与 全规约 启动间接相关（内核 属性要先查好才能 launch）。
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();
  int maxDynamicSmem = 1 << 30;
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));

  for (int sym = 0; sym <= 1; sym++) {
    int kcount = sym == 0 ? ncclDevKernelCount : ncclSymkKernelCount;
    void** kptrs = sym == 0 ? ncclDevKernelList : ncclSymkKernelList;
    int* krequires = sym == 0 ? ncclDevKernelRequirements : ncclSymkKernelRequirements;
    for (int k = 0; k < kcount; k++) {
      if (kptrs[k] != nullptr && driverVersion < krequires[k]) {
        INFO(NCCL_INIT, "Skipping %skernel %d which requires driver %d", sym ? "symmetric " : "", k, krequires[k]);
        kptrs[k] = nullptr;
      }
      void* fn = kptrs[k];
      cudaFuncAttributes attr = {0};
      if (fn == nullptr) continue;

      if (!CUDASUCCESS(cudaFuncGetAttributes(&attr, fn))) continue; // Silently ignore failures

      if (maxStackSize) {
        if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
      }
      if (carveout) {
        CUDACHECKGOTO(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, carveout), result,
                      ignore1);
      ignore1:;
      }
      {
        int dynSmem = maxSharedMem - attr.sharedSizeBytes;
        if (sym) {
          ncclSymkKernelMaxDynamicSmem[k] = dynSmem;
        } else {
          maxDynamicSmem = std::min(maxDynamicSmem, dynSmem);
        }
        CUDACHECKGOTO(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmem), result,
                      next_kernel);
      }
    next_kernel:;
    }
  }

  if (ncclShmemDynamicSize(cudaArch) > maxDynamicSmem) {
    WARN("cudaArch %d dynamic smem %d exceeds device/fn maxSharedMem %d", cudaArch, ncclShmemDynamicSize(cudaArch),
         maxDynamicSmem);
    return ncclSystemError;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// 数据搬运量的统计指标。

static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce:
    return 2;
  case ncclFuncAllGather:
    return nRanks;
  case ncclFuncReduceScatter:
    return nRanks;
  default:
    return 1;
  }
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pEpochEnable, "P2P_EPOCH_ENABLE", 1);

void ncclAddWorkBatchToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
                            enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset, int p2pEpoch,
                            int p2pRound, bool newBatch) {
  size_t workSize = ncclDevWorkSize(workType);
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  // 以下这些条件会促使我们新建一个空白批次(batch)。
  newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    // 以下是所有“无法继续追加到当前批次”的判定条件。
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // 下面这些检查用于确保设备能够承载这么大的批次。它们必须
    // 把所有扩展批次融合后的总量一并计入，这也正是为什么
    // 在创建新的扩展批次时，wipBatch.workBytes 与 wipBatch.nP2ps 不会被清零
    // (见下方相关逻辑)。
    if (workType == ncclDevWorkTypeP2p) {
      if (ncclParamP2pEpochEnable()) newBatch |= chan->wipBatch.p2pEpoch != p2pEpoch;
      // 每个批次最多只允许 NCCL_MAX_DEV_WORK_P2P_PER_BATCH 个操作。
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      for (int i = 0; i < chan->wipBatch.nP2ps; i++) {
        // 同一批次中不允许出现相同的轮次，否则会重复占用同一条连接。
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
        // 确保只在同一个 p2p 分组内部聚合 p2p 操作(一个分组包含
        // NCCL_MAX_DEV_WORK_P2P_PER_BATCH 个操作)。
        // 这样可以强制通信域内各 rank 采用一致的分批方式，从而避免挂死。
        newBatch |= (p2pRound / NCCL_MAX_DEV_WORK_P2P_PER_BATCH) !=
                    (chan->wipBatch.p2pRounds[i] / NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
      }
    }
    if (workType == ncclDevWorkTypeBcast) {
      int maxitem = ncclMaxDevWorkBatchBytes(comm->cudaArch) / sizeof(ncclDevWorkBcast);
      newBatch |= chan->wipBatch.nBcasts == maxitem;
    } else {
      newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    }
  }
  // 以下条件会促使我们创建一个扩展批次(prev->nextExtends=1)
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  bool extendBatch = 63 * workSize < offset;
  extendBatch |= 0 != offset % workSize;
  if (newBatch || extendBatch) {
    if (!newBatch) batch->nextExtends = extendBatch; // Extending the previous batch.
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity 认为 ncclIntruQueueEnqueue 会访问 chan->workBatchQueue->尾，而它可能
    // 为 NULL。但那段代码有 chan->workBatchQueue->头 非空作为前提保护，在此前提下
    // 尾 也必然不会为 NULL。
    // coverity[var_deref_model:假]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->nextExtends = 0;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
    offset = 0;
    if (newBatch) {
      // 由于扩展批次会在设备端被融合在一起，而这些数值
      // 反映的是融合后批次的约束，因此我们只在创建
      // 全新批次时才重置这些数值
      chan->wipBatch.workBytes = 0;
      chan->wipBatch.nP2ps = 0;
      chan->wipBatch.nBcasts = 0;
      // 这里不统计扩展批次，因为该值用于推导 proxyOpCount，
      // 而我们希望所有被融合在一起的操作都拥有相同的取值。
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
      chan->nWorkBatchesBcast += (workType == ncclDevWorkTypeBcast ? 1 : 0);
    }
    plan->nWorkBatches += 1;
  }
  batch->offsetBitset |= 1ull << (offset / workSize);
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {
    chan->wipBatch.p2pEpoch = p2pEpoch;
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound;
  }
  if (workType == ncclDevWorkTypeBcast) {
    chan->wipBatch.nBcasts += 1;
  }
}

static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches * sizeof(struct ncclDevWorkBatch);

  if (plan->isSymColl) return;
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // 如果全部内容能塞进 内核 参数里，就直接放进去(省去额外的显存访问)。
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  plan->kernelArgs =
    (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;

  // 把各批次放入 内核 参数。每个 通道 的首个批次
  // 必须位于 batchZero[blockIdx.x] 处。为此我们按升序对各 通道
  // 做轮转分配，直到全部处理完毕。
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs + 1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull << c;
      }
    } while (tmpMask != 0);
  }

  // 把各 通道 的 代理-操作 链表按 opCount 做归并排序，合并进 plan->proxyOpQueue
  // 第一阶段：扫描每个 通道 的首个操作，把其 opCount 存入 headIds[c]。
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c = 0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c + 1;
  }
  // 第二阶段：从 planner->通道[c] 出队，按归并顺序入队到 plan 中
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // 找出 代理-操作 id 最小的 通道。我们把 heads[c]->opCount 缓存在
    // headIds[c] 中，以消除该循环里的间接内存访问(提升性能)。
    for (int c1 = 0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id >> 1 | id << 63); // Move tag bit to order collectives before p2p's
      if (id < minId) {
        c = c1;
        minId = id;
      }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
}

NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

static ncclResult_t calcCollChunking(struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
                                     /*outputs*/ uint32_t* outChunkSize, uint32_t* outDirectFlags,
                                     struct ncclProxyOp* proxyOp);

struct ncclKernelPlanBudget {
  ssize_t inArgsBytes; // Space available within kernel args struct
  ssize_t outArgsBytes; // Space available outside of args struct (fifo or persistent buf)
};

bool ncclTestBudget(struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes) {
  ssize_t batchBytes = nWorkBatches * sizeof(struct ncclDevWorkBatch);
  bool ok = false;
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;
}

ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task;
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // 为每个任务构建一个 ncclDevWorkColl[Reg?] 结构体。
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    struct ncclWorkList* workNode = NULL;
    struct ncclDevWorkColl devWork = {};

    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      workNode = ncclIntruQueueDequeue(&planner->tmpCollWorkQueue);
      goto next;
    }
    ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    devWork.sendbuff = (void*)task->sendbuff;
    devWork.recvbuff = (void*)task->recvbuff;
    devWork.sendbuffOffset = task->sendbuffOffset;
    devWork.recvbuffOffset = task->recvbuffOffset;
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    devWork.root = task->root;
    devWork.nWarps = task->nWarps;
    devWork.redOpArg = task->opDev.scalarArg;
    devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
    devWork.oneNode = (comm->nNodes == 1);
    devWork.isOneRPN = comm->isOneRPN;
    devWork.netRegUsed = devWork.regUsed = 0;
    devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
    if (task->regBufType & NCCL_NET_REG_BUFFER) devWork.netRegUsed = 1;
    if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER)) devWork.regUsed = 1;

    if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
      struct ncclDevWorkCollReg workReg = {};
      workReg.coll = devWork; // C++ struct assignment
      /* NVLS only has one send and recv buffer registered */
      workReg.dnInputs[0] = regBufSend[0];
      workReg.dnOutputs[0] = regBufRecv[0];
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeCollReg;
      workNode->size = sizeof(struct ncclDevWorkCollReg);
      memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size);
    } else {
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeColl;
      workNode->size = sizeof(struct ncclDevWorkColl);
      memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size);
    }
  next:
    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
    task = task->next;
  }
  assert(ncclIntruQueueEmpty(&planner->tmpCollWorkQueue));
  return ncclSuccess;
}

// 每个 ncclGroup 调用一次，用于把用户提交的任务在
// 通信域->planner 中组织好，以便后续拆分成一个个执行计划(plan)。
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);

  // 若广播对端只有一个，则把广播任务放入 collSorter 统一排序
  if (planner->bcast_info.BcastPeers == 1) {
    while (!ncclIntruQueueEmpty(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue)) {
      struct ncclTaskBcast* bcastTask =
        ncclIntruQueueDequeue(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue);
      struct ncclTaskColl* t =
        ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      t->func = ncclFuncBroadcast;
      t->sendbuff = bcastTask->sendbuff;
      t->recvbuff = bcastTask->recvbuff;
      t->count = bcastTask->count;
      t->root = bcastTask->root;
      t->datatype = bcastTask->datatype;
      t->trafficBytes = t->count * ncclFuncTrafficPerByte(t->func, comm->nRanks);
      t->chunkSteps = BROADCAST_CHUNKSTEPS;
      t->sliceSteps = BROADCAST_SLICESTEPS;
      ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
      planner->nTasksColl += 1;
      ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bcastTask);
    }
    // 重置广播相关信息
    planner->nTasksBcast = 0;
    planner->bcast_info.BcastPeers = 0;
  }

  // 从排序器取出的任务是按大小降序排列的。
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // 任务按 (算子, 规约操作, 数据类型) 分组，并按大小升序组装。
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));
  int fnOpTyIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes];
  int fnOpTyCount = 0;

  // 跨 clique 场景下跳过对称 内核
  if (comm->symmetricSupport && !comm->p2pCrossClique) {
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, task, &planner->collSymTaskQueue, &task));
  }

  // 遍历按大小排序后的任务，按 (算子, 规约操作, 数据类型) 分箱归类。
  while (task != nullptr) {
    struct ncclTaskColl* next = task->next;
    int index = ((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype;
    // 首次出现时，把该 (算子,操作,类型) 组合加入索引集合
    if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
    // 加入该 (算子,操作,类型) 对应的后进先出栈
    task->next = tasksByFnOpTy[index];
    tasksByFnOpTy[index] = task;
    // 处理下一个任务
    task = next;
  }

  // 遍历各个 (算子,操作,类型) 分箱，计算算法与协议等参数。然后再按它们的
  // 调度约束条件(collnet × NVLS)进行二次分箱。
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
  for (int cursor = 0; cursor < fnOpTyCount; cursor++) {
    struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]];
    int collNetSupport = 0;
    NCCLCHECK(ncclGetCollNetSupport(comm, aggBeg, &collNetSupport));
    int nvlsSupport =
      comm->nvlsSupport && (ncclNvlsSupported(aggBeg->opDev.op, aggBeg->datatype) || aggBeg->func == ncclFuncAllGather);
    // 粗略估算每个 通道 上的任务数量。对 NVLS 类算法而言这里用的 通道 数
    // 并不准确，但要确定算法本身又必须先有这个值(存在循环依赖)，
    // 因此要么粗略估算、要么迭代到收敛，我们选择了前者。
    int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);
    do {
      struct ncclTaskColl* aggEnd = aggBeg->next;
      struct ncclTaskColl agg = *aggBeg;
      // 我们会把大小相差在 4 倍以内的操作聚合到一起。
      while (aggEnd != nullptr && aggEnd->trafficBytes < 4 * aggBeg->trafficBytes) {
        agg.count += aggEnd->count;
        agg.trafficBytes += aggEnd->trafficBytes;
        aggEnd = aggEnd->next;
      }

      NCCLCHECK(ncclGetAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
      agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);

      int isCollnet = 0, isNvls = 0;
      switch (agg.algorithm) {
      case NCCL_ALGO_NVLS:
      case NCCL_ALGO_NVLS_TREE:
        isNvls = 1;
        isCollnet = agg.algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1;
        break;
      case NCCL_ALGO_COLLNET_CHAIN:
      case NCCL_ALGO_COLLNET_DIRECT:
        isCollnet = 1;
        break;
      }
      // 用计算得到的结果更新这些已聚合的任务。
      do {
        struct ncclTaskColl* next = aggBeg->next;
        aggBeg->algorithm = agg.algorithm;
        aggBeg->protocol = agg.protocol;
        if (aggBeg->protocol == NCCL_PROTO_LL) aggBeg->trafficBytes *= 4;
        aggBeg->nMaxChannels = agg.nMaxChannels;
        aggBeg->nWarps = agg.nWarps;
        aggBeg->devFuncId = agg.devFuncId;
        aggBeg->isCollnet = isCollnet;
        aggBeg->isNvls = isNvls;
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
        aggBeg = next;
      } while (aggBeg != aggEnd);
    } while (aggBeg != nullptr);
  }

  // 把 collBins[*][*] 拼接成最终的任务列表 planner->collTaskQueue。
  // Collnet 作为外层维度，因为它会影响我们如何在各 通道 之间
  // 划分任务。
  for (int isCollnet = 0; isCollnet <= 1; isCollnet++) {
    for (int isNvls = 0; isNvls <= 1; isNvls++) {
      ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
    }
  }

  // 再次遍历任务，以完成以下工作：
  // 1. 视情况注册缓冲区。
  // 2. 构建 ncclDevWorkColl 结构体。
  // 3. 根据这些工作结构体可被分配到的有效 通道 数量，把它们
  //    可能为 assigned to {collnet, NVLS, 标准}
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // 为每个任务构建一个 ncclDevWorkColl[Reg?] 结构体。
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      if (task->algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false &&
          regNeedConnect == true) {
        comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
        algoNeedConnect[NCCL_ALGO_NVLS] = true;
      }
      if (task->algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    }

    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      struct ncclDevWorkColl devWork = {};
      devWork.sendbuff = (void*)task->sendbuff;
      devWork.recvbuff = (void*)task->recvbuff;
      devWork.sendbuffOffset = task->sendbuffOffset;
      devWork.recvbuffOffset = task->recvbuffOffset;
      devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
      devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
      devWork.root = task->root;
      devWork.nWarps = task->nWarps;
      devWork.redOpArg = task->opDev.scalarArg;
      devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
      devWork.oneNode = (comm->nNodes == 1);
      devWork.netRegUsed = devWork.regUsed = 0;
      devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
      if (task->regBufType & NCCL_NET_REG_BUFFER) devWork.netRegUsed = 1;
      if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER)) devWork.regUsed = 1;

      struct ncclWorkList* workNode;
      if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
        struct ncclDevWorkCollReg workReg = {};
        workReg.coll = devWork; // C++ struct assignment
        /* NVLS only has one send and recv buffer registered */
        workReg.dnInputs[0] = regBufSend[0];
        workReg.dnOutputs[0] = regBufRecv[0];
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeCollReg;
        workNode->size = sizeof(struct ncclDevWorkCollReg);
        memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size);
      } else {
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeColl;
        workNode->size = sizeof(struct ncclDevWorkColl);
        memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size);
      }

      ncclIntruQueueEnqueue(&planner->tmpCollWorkQueue, workNode);
    }
    task = task->next;
  }

  // 处理 runtimeConn 相关的广播任务
  if (comm->runtimeConn && planner->nTasksBcast > 0) {
    for (int peer = planner->bcast_info.minBcastPeer; peer <= planner->bcast_info.maxBcastPeer; peer++) {
      struct ncclTaskBcast* bcastTask = ncclIntruQueueHead(&planner->peers[peer].bcastQueue);
      while (bcastTask != nullptr) {
        if (comm->initAlgoChannels[NCCL_ALGO_RING] == false) {
          comm->initAlgoChannels[NCCL_ALGO_RING] = true;
          algoNeedConnect[NCCL_ALGO_RING] = true;
          *needConnect = true;
        }
        bcastTask = bcastTask->next;
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t addProfilerProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                               struct ncclProxyOp* op) {
  int tmp = op->pattern;
  op->pattern = ncclPatternProfiler;
  ncclResult_t ret = ncclAddProxyOpIfNeeded(comm, plan, op);
  op->pattern = tmp;
  return ret;
}

static ncclResult_t scheduleCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                            struct ncclKernelPlanBudget* budget) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // 估算本计划中能够容纳多少个任务。
  int nPlanColls = 0;
  size_t trafficBytes[2 * 2] = {0, 0, 0, 0}; // [collnet][nvls]
  int nChannels[2 * 2] = {0, 0, 0, 0}; // [collnet][nvls]
  int const nMaxChannels[2 * 2] = {comm->nChannels, comm->nvlsChannels, // [collnet][nvls]
                                   comm->nChannels, std::min(comm->nChannels, comm->nvlsChannels)};
  constexpr size_t MinTrafficPerChannel = 32 << 10; // 32K traffic as minimal
  do {
    size_t workBytes = 0;
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    while (task != nullptr) {
      int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
      if (!ncclTestBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;

      nPlanColls += 1;
      workBytes += workNode->size;
      int kind = 2 * task->isCollnet + task->isNvls;
      trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);
      nChannels[kind] += task->nMaxChannels;
      nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);
      task = task->next;
      workNode = workNode->next;
    }
  plan_full:;
  } while (0);

  int kindPrev = -1;
  size_t trafficPerChannel = 0;
  int channelId = 0;
  size_t currentTraffic = 0;
  while (nPlanColls != 0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode + 1);
    size_t elementSize = ncclTypeSize(task->datatype);

    int kind = 2 * task->isCollnet + task->isNvls;
    if (kind != kindPrev) {
      trafficPerChannel = divUp(trafficBytes[kind] / nChannels[kind], 16) * 16;
      kindPrev = kind;
      channelId = 0;
      currentTraffic = 0;
    }

    if (task->isCollnet) {
      int nChannels = task->nMaxChannels;
      // 预留出最坏情况所需的空间：即每个 通道 都新增一个批次
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      size_t globalBytesPerElement = elementSize * ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOp;
      uint32_t chunkSize, directFlags = 0;
      NCCLCHECK(calcCollChunking(comm, task, nChannels, globalBytesPerElement * task->count, &chunkSize, &directFlags,
                                 &proxyOp));
      devWork->channelLo = 0;
      devWork->channelHi = nChannels - 1;
      devWork->collnet.count = task->count;
      devWork->collnet.chunkCount = chunkSize / ncclTypeSize(task->datatype);
      devWork->direct = directFlags;

      uint64_t proxyOpId = uint64_t(plan->collOpCount++) << 1 | 0;
      for (int c = devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        proxyOp.channelId = c;
        proxyOp.opCount = proxyOpId;
        proxyOp.task.coll = task;
        proxyOp.rank = comm->rank;
        proxyOp.eActivationMask = task->eActivationMask;
        proxyOp.incWorkCounter = true;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // 向 剖析器 设置模式，以便为 内核 事件添加一个 代理 性能分析器
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, &proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    } else {
      // 非 collnet 任务(即 task->isCollnet 为假)
      int trafficPerByte = ncclFuncTrafficPerByte(task->func, comm->nRanks);
      if (task->protocol == NCCL_PROTO_LL) trafficPerByte *= 4;
      size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
      int elementsPerCell = cellSize / elementSize;
      size_t cells = divUp(task->count * elementSize, cellSize);
      size_t trafficPerElement = elementSize * trafficPerByte;
      size_t trafficPerCell = cellSize * trafficPerByte;
      size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));
      size_t cellsLo;
      if (channelId + 1 == nMaxChannels[kind]) {
        // 在最后一个 通道 上，所有内容都归入 lo(低区)
        cellsLo = cells;
      } else {
        cellsLo = std::min(cells, divUp((trafficPerChannel - currentTraffic), trafficPerCell));
      }
      int nMidChannels = (cells - cellsLo) / cellsPerChannel;
      size_t cellsHi = (cells - cellsLo) % cellsPerChannel;
      int nChannels = (cellsLo != 0 ? 1 : 0) + nMidChannels + (cellsHi != 0 ? 1 : 0);
      if (nMaxChannels[kind] < channelId + nChannels) {
        // 已超出可用 通道 的数量
        nMidChannels = nMaxChannels[kind] - channelId - 2;
        cellsPerChannel = (cells - cellsLo) / (nMidChannels + 1);
        cellsHi = cellsPerChannel + (cells - cellsLo) % (nMidChannels + 1);
      }
      if (cellsHi == 0 && nMidChannels != 0) {
        cellsHi = cellsPerChannel;
        nMidChannels -= 1;
      }
      if (cellsLo == 0) {
        // 最小的那个 通道 已被跳过。把下一个 通道 设为新的最小值。
        channelId += 1;
        if (nMidChannels == 0) {
          cellsLo = cellsHi;
          cellsHi = 0;
        } else {
          cellsLo = cellsPerChannel;
          nMidChannels -= 1;
        }
      }
      size_t countMid = nMidChannels != 0 ? cellsPerChannel * elementsPerCell : 0;
      size_t countLo = cellsLo * elementsPerCell;
      size_t countHi = cellsHi * elementsPerCell;
      (countHi != 0 ? countHi : countLo) -= cells * elementsPerCell - task->count;

      nChannels = (countLo != 0 ? 1 : 0) + nMidChannels + (cellsHi != 0 ? 1 : 0);

      // 更新传递给 剖析器 的 通道 数量
      task->nChannels = (uint8_t)nChannels;

      // 预留出最坏情况所需的空间：即每个 通道 都新增一个批次
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      devWork->channelLo = channelId;
      devWork->channelHi = channelId + nChannels - 1;
      devWork->cbd.countLo = countLo;
      devWork->cbd.countMid = countMid;
      devWork->cbd.countHi = countHi;

      // calcCollChunking() 使用的是全局字节数而非流量，二者的差别在于
      // 全规约 的字节数不会乘以 2。
      size_t globalBytesPerElement = elementSize * ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi;

      uint32_t chunkSize, directFlags = 0;
      size_t grainSize = ncclProtoGrainSize(task->protocol);
      if (countLo != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countLo, &chunkSize,
                                   &directFlags, &proxyOpLo));
        devWork->cbd.chunkGrainsLo = chunkSize / grainSize;
      }
      if (countHi != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countHi, &chunkSize,
                                   &directFlags, &proxyOpHi));
        devWork->cbd.chunkGrainsHi = chunkSize / grainSize;
      }
      if (nMidChannels != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countMid, &chunkSize,
                                   &directFlags, &proxyOpMid));
        devWork->cbd.chunkGrainsMid = chunkSize / grainSize;
      }
      devWork->direct = directFlags;

      // 更新当前 通道 以及剩余的流量预算。
      if (countHi != 0) {
        channelId += nChannels - 1;
        currentTraffic = cellsHi * elementsPerCell * trafficPerElement;
      } else if (nMidChannels != 0) {
        channelId += nChannels;
        currentTraffic = 0;
      } else {
        currentTraffic += cellsLo * elementsPerCell * trafficPerElement;
      }

      if (currentTraffic >= trafficPerChannel && channelId + 1 != nMaxChannels[kind]) {
        channelId += 1;
        currentTraffic = 0;
      }

      uint64_t proxyOpId = uint64_t(plan->collOpCount++) << 1 | 0;
      for (int c = devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        struct ncclProxyOp* proxyOp;
        if (c == (int)devWork->channelLo) {
          proxyOp = &proxyOpLo;
          proxyOp->loopOffset = 0;
          proxyOp->channelSize = countLo * elementSize;
        } else if (c == (int)devWork->channelHi) {
          proxyOp = &proxyOpHi;
          proxyOp->loopOffset = (countLo + nMidChannels * countMid) * elementSize;
          proxyOp->channelSize = countHi * elementSize;
        } else {
          proxyOp = &proxyOpMid;
          proxyOp->loopOffset = (countLo + (c - devWork->channelLo - 1) * countMid) * elementSize;
          proxyOp->channelSize = countMid * elementSize;
        }
        proxyOp->channelId = c;
        proxyOp->opCount = proxyOpId;
        proxyOp->task.coll = task;
        proxyOp->rank = comm->rank;
        proxyOp->ringAlgo = NULL;
        if (proxyOp->reg && task->algorithm == NCCL_ALGO_RING && (task->recvNetHandles[c] || task->sendNetHandles[c])) {
          if (task->func == ncclFuncAllGather) {
            proxyOp->ringAlgo =
              new RingAGAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.userRanks,
                                  proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize,
                                  proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->count * elementSize,
                                  task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncAllReduce) {
            proxyOp->ringAlgo =
              new RingARAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.index,
                                  proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize,
                                  proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->sendNetHandles[c],
                                  task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncBroadcast) {
            proxyOp->ringAlgo =
              new RingBCAlgorithm(task->sendbuff, task->recvbuff, comm->rank, task->root, comm->nRanks,
                                  comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps,
                                  proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize,
                                  task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          }
          proxyOp->ringAlgo->incRefCount();
        }
        proxyOp->eActivationMask = task->eActivationMask;
        proxyOp->incWorkCounter = true;
        proxyOp->nChannels = nChannels;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Coverity 报告 proxyOp->连接 可能未初始化。目前难以
        // 确认该判断是否属实，同时也不清楚即便属实是否真会引发问题。
        // coverity[uninit_use_in_call:假]
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, proxyOp));
      }
    }

    plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo);
    plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE);
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
    }
    // 性能分析器(剖析器)相关处理
    plan->groupApiEventHandle = task->groupApiEventHandle;

    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype), ncclAlgoToString(task->algorithm),
           ncclProtoToString(task->protocol), devWork->channelLo, devWork->channelHi);

      if (task->isCollnet) {
        TRACE(NCCL_COLL,
              "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count=%ld chunkCount=%d",
              ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op), ncclDatatypeToString(task->datatype),
              ncclAlgoToString(task->algorithm), ncclProtoToString(task->protocol), (long)task->count, task->devFuncId,
              devWork->channelLo, devWork->channelHi, (long)devWork->collnet.count, devWork->collnet.chunkCount);
      } else {
        TRACE(NCCL_COLL,
              "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} "
              "count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
              ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op), ncclDatatypeToString(task->datatype),
              ncclAlgoToString(task->algorithm), ncclProtoToString(task->protocol), (long)task->count, task->devFuncId,
              devWork->channelLo, devWork->channelHi, (long)devWork->cbd.countLo, (long)devWork->cbd.countMid,
              (long)devWork->cbd.countHi, int(devWork->cbd.chunkGrainsLo * ncclProtoGrainSize(task->protocol)),
              int(devWork->cbd.chunkGrainsMid * ncclProtoGrainSize(task->protocol)),
              int(devWork->cbd.chunkGrainsHi * ncclProtoGrainSize(task->protocol)));
      }
    }

    for (int i = 0; i < task->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);
    nPlanColls -= 1;
    planner->nTasksColl -= 1;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

// 把 p2p 操作放入 plan，前提是批次预算中还剩 sizeof(ncclDevWorkBatch)、
// 工作预算中还剩 sizeof(ncclDevWorkP2p)。sendRank 与 recvRank 必须
// 与本轮 p2p 调度表中的对应值一致(不允许出现 -1)。
// 空操作(无-操作)用 -1 作为 大小 来表示。
static ncclResult_t addP2pToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, int nChannelsMin, int nChannelsMax,
                                 int p2pEpoch, int p2pRound, int sendRank, void* sendAddr, ssize_t sendBytes,
                                 int recvRank, void* recvAddr, ssize_t recvBytes, const int planTotalTasks[],
                                 struct ncclTaskP2p** p2pTasks) {
  ncclResult_t ret = ncclSuccess;
  constexpr int connIndex = 1;
  bool selfSend = (sendRank == comm->rank);
  // 方向约定：接收 dir=0，发送 dir=1
  void* addrs[2] = {recvAddr, sendAddr};
  ssize_t bytes[2] = {recvBytes, sendBytes};
  bool protoLL[2] = {!selfSend, !selfSend};
  bool network[2] = {false, false};
  bool proxySameProcess[2] = {true, true};
  void** handles[2] = {NULL, NULL};
  uint8_t base = ncclP2pChannelBaseForRound(comm, p2pRound);
  struct ncclProxyOp proxyOps[2] = {};
  int nProxyOps = selfSend ? 0 : 2;
  if (!selfSend) {
    for (int part = 0; part < nChannelsMax; part++) {
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
      for (int dir = 0; dir <= 1; dir++) {
        int peerRank = dir ? sendRank : recvRank;
        struct ncclConnector* conn =
          dir ? &channelPeers[peerRank]->send[connIndex] : &channelPeers[peerRank]->recv[connIndex];
        protoLL[dir] &= conn->conn.buffs[NCCL_PROTO_LL] != nullptr;
        network[dir] |= conn->transportComm == (dir ? &netTransport.send : &netTransport.recv);
        proxySameProcess[dir] &= conn->proxyConn.sameProcess;
      }
    }
  }

  ssize_t paramChunkSize = ncclParamChunkSize();
  // 以 dir 为下标的数组，其中接收为 0，发送为 1：
  int nChannels[2];
  int protocol[2];
  int stepSize[2];
  int chunkSize[2];
  int chunkDataSize[2];
  int chunkDataSize_u32fp8[2];
  bool netRegistered[2] = {false, false};
  bool ipcRegistered[2] = {false, false};

  for (int dir = 0; dir < 2; dir++) {
    // 0 表示接收，1 表示发送
    // 先假定使用 SIMPLE 协议，以此估算出 通道 的数量
    stepSize[dir] = comm->p2pChunkSize;

    if (bytes[dir] == -1) {
      nChannels[dir] = 0;
    } else if (bytes[dir] == 0) {
      nChannels[dir] = 1;
    } else {
      ssize_t minPartSize = comm->nNodes > 1 ? stepSize[dir] / 2 : stepSize[dir] / 8;
      ssize_t maxPartSize = comm->nNodes > 1 ? stepSize[dir] : stepSize[dir] * 32;
      nChannels[dir] = std::min<int>(nChannelsMin, divUp(bytes[dir], minPartSize));
      size_t partSize = std::max(minPartSize, divUp(bytes[dir], nChannels[dir]));
      while (partSize > maxPartSize && nChannels[dir] <= nChannelsMax / 2) {
        nChannels[dir] *= 2;
        partSize = divUp(bytes[dir], nChannels[dir]);
      }
    }
    // 更新传递给 剖析器 的 通道 数量
    if (p2pTasks[dir]) p2pTasks[dir]->nChannels = nChannels[dir];

    // 再根据每个 通道 实际承载的数据量，选择使用 LL 还是 SIMPLE 协议
    if (bytes[dir] != -1) protoLL[dir] &= bytes[dir] <= nChannels[dir] * ncclParamP2pLLThreshold();
    protocol[dir] = protoLL[dir] ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

    stepSize[dir] = comm->buffSizes[protocol[dir]] / NCCL_STEPS;
    if (protocol[dir] == NCCL_PROTO_SIMPLE) stepSize[dir] = comm->p2pChunkSize;
    chunkSize[dir] = stepSize[dir];
    if (paramChunkSize != 0) {
      chunkSize[dir] = paramChunkSize;
    } else if (network[dir]) {
      // 针对网络传输调整 块 大小
      if (protocol[dir] == NCCL_PROTO_SIMPLE && bytes[dir] < stepSize[dir]) chunkSize[dir] /= 4;
      else if (bytes[dir] < 8 * stepSize[dir]) chunkSize[dir] /= 2;
    }

    chunkDataSize[dir] = chunkSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkDataSize[dir] /= 2;
    chunkDataSize_u32fp8[dir] = u32fp8Encode(chunkDataSize[dir]);
    chunkDataSize[dir] = u32fp8Decode(chunkDataSize_u32fp8[dir]);
    chunkSize[dir] = chunkDataSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkSize[dir] *= 2;

    if (p2pTasks[dir] && p2pTasks[dir]->allowUB) {
      if (network[dir]) {
        bool pxnUsed = !ncclPxnDisable(comm) && comm->isAllNvlink && comm->maxLocalRanks > 1;
        if (bytes[dir] > 0 && proxySameProcess[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && (!pxnUsed)) {
          int regFlag = 0;
          NCCLCHECKGOTO(ncclCalloc(&handles[dir], nChannelsMax), ret, cleanup);
          for (int part = 0; part < nChannelsMax; part++) {
            int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
            struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
            int peerRank = dir ? sendRank : recvRank;
            struct ncclConnector* conn =
              dir ? &channelPeers[peerRank]->send[connIndex] : &channelPeers[peerRank]->recv[connIndex];
            if (conn->conn.flags & NCCL_DIRECT_NIC) {
              ncclRegisterP2pNetBuffer(comm, addrs[dir], bytes[dir], conn, &regFlag, &handles[dir][part],
                                       &plan->cleanupQueue);
            }
            if (!regFlag) break;
          }
          netRegistered[dir] = regFlag ? true : false;
        }
      } else if (bytes[dir] > 0 && addrs[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && !selfSend) {
        int peerRank = dir ? sendRank : recvRank;
        int regFlag = 0;
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, 0);
        struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
        struct ncclConnector* conn =
          dir ? &channelPeers[peerRank]->send[connIndex] : &channelPeers[peerRank]->recv[connIndex];
        void* regAddr = NULL;
        if (conn->conn.flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
          // 要求用户在收发两侧都完成缓冲区注册
          NCCLCHECKGOTO(ncclRegisterP2pIpcBuffer(comm, addrs[dir], bytes[dir], peerRank, &regFlag, &regAddr,
                                                 &plan->cleanupQueue),
                        ret, cleanup);
          if (regFlag) {
            if (dir == 0 && (conn->conn.flags & NCCL_P2P_WRITE)) recvAddr = regAddr;
            else if (dir == 1 && (conn->conn.flags & NCCL_P2P_READ)) sendAddr = regAddr;
          }
        }
        ipcRegistered[dir] = regFlag ? true : false;
      }
    }
  }

  struct ncclWorkList* workNode;
  workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkP2p>(&comm->memScoped, 1);
  workNode->workType = ncclDevWorkTypeP2p;
  workNode->size = sizeof(struct ncclDevWorkP2p);
  ncclIntruQueueEnqueue(&plan->workQueue, workNode);
  uint32_t workOffset;
  workOffset = plan->workBytes;
  plan->workBytes += sizeof(struct ncclDevWorkP2p);

  struct ncclDevWorkP2p* work;
  work = (struct ncclDevWorkP2p*)(workNode + 1);
  work->nP2pChannels = comm->p2pnChannels;
  work->channelBase = base;
  work->nSendChannels = nChannels[1];
  work->sendProtoLL = protoLL[1];
  work->sendNetReg = netRegistered[1];
  work->sendIpcReg = ipcRegistered[1];
  work->sendChunkSize_u32fp8 = chunkDataSize_u32fp8[1];
  work->sendRank = sendRank;
  work->sendAddr = sendAddr;
  work->sendBytes = sendBytes == -1 ? 0 : sendBytes;
  work->nRecvChannels = nChannels[0];
  work->recvProtoLL = protoLL[0];
  work->recvNetReg = netRegistered[0];
  work->recvIpcReg = ipcRegistered[0];
  work->recvChunkSize_u32fp8 = chunkDataSize_u32fp8[0];
  work->recvRank = recvRank;
  work->recvAddr = recvAddr;
  work->recvBytes = recvBytes == -1 ? 0 : recvBytes;
  work->profilerEnabled =
    ncclProfilerPluginLoaded() && ((p2pTasks[0] ? p2pTasks[0] : p2pTasks[1])->eActivationMask & ncclProfileKernelCh);

  for (int dir = 0; dir < nProxyOps; dir++) {
    struct ncclProxyOp* op = &proxyOps[dir];
    op->root = dir ? sendRank : recvRank;
    op->sliceSteps = 1;
    op->chunkSteps = 1;
    op->dtype = ncclInt8;
    op->redOp = ncclSum;
    op->protocol = protocol[dir];
    op->pattern = dir ? ncclPatternSend : ncclPatternRecv;
    op->chunkSize = chunkSize[dir];
    op->reg = netRegistered[dir];
    op->coll = p2pTasks[dir] ? p2pTasks[dir]->func : 0;
    op->collAPI = p2pTasks[dir] ? p2pTasks[dir]->collAPI : 0;
    op->task.p2p = p2pTasks[dir];
    op->rank = comm->rank;
    op->eActivationMask = p2pTasks[dir] ? p2pTasks[dir]->eActivationMask : 0;
    // 以下字段会在 addWorkToChannels() 中按每个 通道 分片分别修改：
    // 操作->缓冲区、操作->nbytes、操作->nsteps 等字段在此赋值……
  }

  nChannelsMax = std::max(nChannels[0], nChannels[1]);
  // 确定本计划会并发面向多少个对端。这里做一个
  // 简化假设：每个任务面向的对端各不相同。
  // 每个任务会被条带化(stripe)分散到 p2pnChannels 中的 nChannelsMax 个 通道 上。
  // 每个 通道 最多并发执行 NCCL_MAX_DEV_WORK_P2P_PER_BATCH 个任务。
  int maxConcurrent;
  int concurrentTasks[2];
  maxConcurrent = comm->p2pnChannels / nChannelsMax * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  concurrentTasks[0] = std::min(planTotalTasks[0], maxConcurrent);
  concurrentTasks[1] = std::min(planTotalTasks[1], maxConcurrent);
  for (int part = 0; part < nChannelsMax; part++) {
    int incWorkCounter = -1;
    int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
    plan->channelMask |= uint64_t(1) << channelId;
    // 先添加批次。
    ncclAddWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeP2p, ncclDevFuncId_P2p(), workOffset, p2pEpoch,
                           p2pRound);
    for (int dir = 0; dir < nProxyOps; dir++) {
      // 把各步骤(步骤)划分到各个 通道 上。
      int nParts = dir ? work->nSendChannels : work->nRecvChannels;
      void* addr = dir ? work->sendAddr : work->recvAddr;
      size_t bytes = dir ? work->sendBytes : work->recvBytes;

      proxyOps[dir].recvbuff = nullptr;
      if (nParts <= part) {
        proxyOps[dir].nsteps = 0;
      } else if (bytes == 0) {
        proxyOps[dir].nsteps = 1;
        proxyOps[dir].nbytes = 0;
      } else {
        size_t chunkDataSize = u32fp8Decode(dir ? work->sendChunkSize_u32fp8 : work->recvChunkSize_u32fp8);
        size_t partBeg, partEnd;
        ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
        if (proxyOps[dir].reg) {
          (dir ? proxyOps[dir].sendbuff : proxyOps[dir].recvbuff) = (uint8_t*)addr + partBeg;
          (dir ? proxyOps[dir].sendMhandle : proxyOps[dir].recvMhandle) = handles[dir][part];
          proxyOps[dir].nbytes = partEnd - partBeg;
          proxyOps[dir].nsteps = DIVUP(proxyOps[dir].nbytes, NCCL_MAX_NET_SIZE);
        } else {
          proxyOps[dir].nsteps = divUp(partEnd - partBeg, chunkDataSize);
          proxyOps[dir].nbytes = std::min(partEnd - partBeg, chunkDataSize);
        }
        if (proxyOps[dir].protocol == NCCL_PROTO_LL) {
          proxyOps[dir].nbytes *= 2;
          proxyOps[dir].nbytes = roundUp(proxyOps[dir].nbytes, sizeof(union ncclLLFifoLine));
        }
      }

      // 按 <发送, 接收> 配对来递增工作计数器，而不是按单个 p2p 操作递增
      if (proxyOps[dir].nsteps && incWorkCounter < 0) {
        proxyOps[dir].incWorkCounter = true;
        incWorkCounter = dir;
      }

      if (proxyOps[dir].nsteps != 0) {
        // 在添加批次之后再计算 opCount，因为那时批次计数正好
        // 等于该 p2p 所落入的批次索引加一。
        proxyOps[dir].channelId = channelId;
        proxyOps[dir].opCount = uint64_t(comm->planner.wipPlan.channels[channelId].nWorkBatchesP2p) << 1 | 1;
        proxyOps[dir].nChannels = nChannels[dir];
        proxyOps[dir].nPeers = concurrentTasks[dir];
        NCCLCHECKGOTO(ncclAddProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
        NCCLCHECKGOTO(addProfilerProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
      }
    }
  }
cleanup:
  free(handles[0]);
  free(handles[1]);
  return ret;
}

static int calcP2pChannelCount(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels / 2) {
    nChannels *= 2;
    size = divUp(totalSize, nChannels);
  }
  return nChannels;
}

static ncclResult_t scheduleP2pTasksToPlan(struct ncclComm* comm, int* p2pEpoch, int* p2pRound,
                                           struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  int nRanks = comm->nRanks;
  struct ncclKernelPlanner::Peer* peers = comm->planner.peers;

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
  }

  // 计算操作需要拆分的粒度
  // 尽量使用全部 通道
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // 尽量用满所有 通道，但每个操作只占用一个 通道。
  while (nChannelsMin * nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

  // 在 plan 中保存收发任务的总数
  int planTotalTasks[2] = {comm->planner.nTasksP2pRecv, comm->planner.nTasksP2pSend};
  while (comm->planner.nTasksP2p != 0) {
    for (; *p2pRound < nRanks; (*p2pRound)++) {
      int sendRank = comm->p2pSchedule[*p2pRound].sendRank;
      int recvRank = comm->p2pSchedule[*p2pRound].recvRank;
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
      if (send == nullptr && recv == nullptr) continue;

      if (sendRank == comm->rank) {
        if (send != nullptr && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == nullptr && recv != nullptr) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      ssize_t sendBytes = send ? send->bytes : -1;
      ssize_t recvBytes = recv ? recv->bytes : -1;
      void* sendBuff = send ? send->buff : nullptr;
      void* recvBuff = recv ? recv->buff : nullptr;

      if (sendRank == comm->rank && send->buff == recv->buff) {
        // 跳过“原地自发自收”的情况(无需支持这种用法)。
        ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
        ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, send);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, recv);
        comm->planner.nTasksP2p -= 2;
        comm->planner.nTasksP2pSend -= 1;
        comm->planner.nTasksP2pRecv -= 1;
      } else {
        // 预留最坏情况所需空间：每个 通道 各新增一个批次。
        if (!ncclTestBudget(budget, plan->nWorkBatches + nChannelsMax,
                            plan->workBytes + sizeof(struct ncclDevWorkP2p))) {
          return ncclSuccess;
        }
        struct ncclTaskP2p* p2pTasks[2] = {recv, send};
        NCCLCHECK(addP2pToPlan(comm, plan, nChannelsMin, nChannelsMax, *p2pEpoch, *p2pRound, sendRank, sendBuff,
                               sendBytes, recvRank, recvBuff, recvBytes, planTotalTasks, p2pTasks));
        if (send != nullptr) {
          ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
          // 性能分析器 —— 此处可以直接覆盖 groupAPI 事件句柄，因为这里的所有操作都属于同一个 组
          plan->groupApiEventHandle = send->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pSend -= 1;
        }
        if (recv != nullptr) {
          ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
          // 性能分析器 —— 此处可以直接覆盖 groupAPI 事件句柄，因为这里的所有操作都属于同一个 组
          plan->groupApiEventHandle = recv->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pRecv -= 1;
        }
      }
    }
    *p2pRound = 0;
    (*p2pEpoch)++;
  }
  return ncclSuccess;
}

// 自旋等待，直到可以安全地把 通信域->workFifoProduced 推进到 desiredProduced 为止。
static ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  if (!hasRoom) {
    while (true) {
      // 检查中止标志：一旦收到中止信号就跳出，避免死锁
      if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
        return ncclInternalError;
      }

      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
      if (hasRoom) break;
      std::this_thread::yield();
    }
  }
  return ncclSuccess;
}

namespace {
struct uploadWork_cleanup_t {
  struct ncclCommEventCallback base;
  void* hostBuf;
};
ncclResult_t uploadWork_cleanup_fn(struct ncclComm* comm, struct ncclCommEventCallback* cb) {
  struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
  ncclOsAlignedFree(me->hostBuf);
  CUDACHECK(cudaEventDestroy(me->base.event));
  free(me);
  return ncclSuccess;
}
} // namespace

static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (plan->isSymColl || plan->isCeColl || plan->isRma) return ncclSuccess;

  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches * sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  case ncclDevWorkStorageTypeFifo:
    fifoBufHost = comm->workFifoBuf;
    fifoCursor = comm->workFifoProduced;
    fifoMask = comm->workFifoBytes - 1;
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
    plan->kernelArgs->workBuf = comm->workFifoBufDev;
    break;
  case ncclDevWorkStorageTypePersistent:
    {
      size_t hostAllocBytes = workBytes;
// 这里依赖 16 字节对齐。若环境支持则使用对齐分配(C++11 及以上，或带 /std:c++11+ 的 MSVC)。
// MSVC 会把 __cplusplus 固定报告为 199711L(需特殊判断)
#if (__cplusplus >= 201103L) || (defined(_MSC_VER) && _MSVC_LANG >= 201103L)
      hostAllocBytes = ROUNDUP(workBytes, 16);
      fifoBufHost = ncclOsAlignedAlloc(16, hostAllocBytes);
#else
      static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
      fifoBufHost = malloc(workBytes);
#endif
      INFO_LOC(NCCL_ALLOC_HOST, "Persistent host work buf Size %zu pointer %p", hostAllocBytes, fifoBufHost);
      fifoCursor = 0;
      fifoMask = ~0u;
      break;
    }
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // finishPlan() 已经把各批次放在 kernelArgs 之后。现在剩下要做的
  // 只是把工作偏移量从“plan 内以 0 为基准”转换为：
  //  ncclDevWorkStorageTypeArgs: 偏移 from beginning of 内核 args
  //  ncclDevWorkStorageTypeFifo: 偏移 from base of fifo
  //  ncclDevWorkStorageTypePersistent: 无 translation 自 our dedicated 缓冲区 will 也 开始 at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs + 1);
  for (int b = 0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // 写入各 通道 共享的工作结构体。
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode + 1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(COMPILER_ASSUME_ALIGNED(dst + (fifoCursor & fifoMask), 16), COMPILER_ASSUME_ALIGNED(src, 16), 16);
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    comm->workFifoProduced = fifoCursor;
    if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    break;
  case ncclDevWorkStorageTypePersistent:
    {
      ncclResult_t result = ncclSuccess;
      struct uploadWork_cleanup_t* cleanup = nullptr;
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      void* fifoBufDev = nullptr;
      cudaStream_t deviceStream;

      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), result, fail);

      // 获取 deviceStream。由于用户的 图 会在稍后启动，而它同样会
      // 获取该 deviceStream，因此它必然能观察到本次上传的数据。
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
                                            &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream),
                    result, fail);

      CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, deviceStream), result, fail);
      INFO_LOC(NCCL_ALLOC, "Persistent cudaMallocAsync work buf Size %zu pointer %p", workBytes, fifoBufDev);
      plan->workBufPersistent = fifoBufDev;
      plan->kernelArgs->workBuf = fifoBufDev;

      // coverity[uninit_use_in_call:假] => fifoBufHost is never NULL
      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
      cudaEvent_t memcpyDone;
      CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, fail);
      CUDACHECKGOTO(cudaEventRecord(memcpyDone, deviceStream), result, fail);

      NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
      cleanup->base.fn = uploadWork_cleanup_fn;
      cleanup->base.event = memcpyDone;
      cleanup->hostBuf = fifoBufHost;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, (struct ncclCommEventCallback*)cleanup);

      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
                                            &comm->sharedRes->deviceStream, /*concurrent=*/false),
                    result, fail);
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), result, fail);

    finish_scope:
      if (mode != cudaStreamCaptureModeRelaxed) (void)cudaThreadExchangeStreamCaptureMode(&mode);
      return result;
    fail:
      if (!cleanup) ncclOsAlignedFree(fifoBufHost);
      goto finish_scope;
    }
    break;
  default:
    break;
  }
  return ncclSuccess;
}

static int geteActivationMask(struct ncclProxyOp* op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eActivationMask;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return 0;
  }
  return op->task.coll->eActivationMask;
}

static void* gettaskEventHandle(struct ncclProxyOp* op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eventHandle;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return nullptr;
  }
  return op->task.coll->eventHandle;
}

static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // 按本计划中集合通信操作的数量，推进 通信域 的 collOpCount。
  int hasp2p = 0;
  comm->sharedRes->collOpCount += plan->collOpCount;
  comm->collOpCount += plan->collOpCount;

  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    op->profilerContext = comm->profilerContext;
    op->eActivationMask = geteActivationMask(op);
    op->taskEventHandle = gettaskEventHandle(op);
    ncclProfilerAddPidToProxyOp(op);

    uint64_t oldId = op->opCount;
    // 忽略最低位的 tag 标志位后，opCount 在 plan 内部是从 0 开始计数的，因此
    // 需要把它们平移到 通信域 历史计数的末端。
    if (oldId & 1) {
      // p2p
      // 在同一 plan 的同一 通道 内 opCount 是单调递增的，所以只需
      // 记住最后一个值即可得到最大值。
      p2pOpBump[op->channelId] = (oldId >> 1) + 1; // +1 to ensure next plan doesn't collide
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId] << 1) + oldId;
      hasp2p = 1;
    } else {
      // 集合通信
      op->opCount = (collOpCount << 1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    op->opCount = oldId; // Restore for next uploadProxyOps()
    op = op->enqNext;
  }

  if (hasp2p) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      // 按本 plan 该 通道 中的 p2p 数量，推进 通道 的 p2pOpCount。
      comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
    }
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  if (ncclIntruQueueHead(&plan->proxyOpQueue)) {
    NCCLCHECK(uploadProxyOps(comm, plan));
    NCCLCHECK(ncclProxyStart(comm));
  }
  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  if (!plan->persistent) {
    // 通知主线程我们正在回收，主线程会并发地回收该 plan。
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static void CUDART_CB hostStreamPlanCallback(void* plan_) {
  NCCL_NVTX3_FUNC_RANGE;
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  if (result != ncclSuccess) {
    WARN("hostStreamPlanCallback() failed : %s", ncclGetErrorString(result));
  }
  return;
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->sharedRes->persistentRefs -= 1;
    comm->localPersistentRefs -= 1;
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      CUDACHECK(cudaFree(plan->workBufPersistent));
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    }
  }
  if (plan->isSymColl) {
    free(plan->kernelSymArgs);
  }
  // 释放集合(集合)任务
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct != nullptr) {
    struct ncclTaskColl* ct1 = ct->next;
    free(ct->sendNetHandles);
    free(ct->recvNetHandles);
    free(ct->srecvNetHandles);
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
    ct = ct1;
  }
  // 释放 p2p 任务
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt != nullptr) {
    struct ncclTaskP2p* pt1 = pt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, pt);
    pt = pt1;
  }
  // 释放广播(广播)任务
  struct ncclTaskBcast* bt = ncclIntruQueueHead(&plan->bcastTaskQueue);
  while (bt != nullptr) {
    struct ncclTaskBcast* bt1 = bt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bt);
    bt = bt1;
  }
  // 释放 代理 操作
  struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (q != nullptr) {
    struct ncclProxyOp* q1 = q->enqNext;
    if (q->ringAlgo && q->ringAlgo->decRefCount() == 0) delete q->ringAlgo;
    ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    q = q1;
  }
  // 释放 RMA 持久描述符(图 模式)
  // 纯 RMA 的 plan 总是创建持久描述符；CE 的 plan 仅在分层(多节点)路径下才会创建，
  // 即 ncclHierCeAllGather 使用 RMA 代理 的情形。
  if (plan->persistent && (plan->isRma || (plan->isCeColl && comm->nNodes > 1))) {
    NCCLCHECK(ncclRmaProxyReclaimPlan(comm, plan));
  }
  // 运行其它释放回调
  ncclResult_t result = ncclSuccess;
  while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
    ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
  }
  NCCLCHECK(result);
  // 释放 plan 结构体
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

static void persistentDestructor(void* plans_) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plans_;
  struct ncclComm* comm = plan->comm;
  while (plan != nullptr) {
    struct ncclKernelPlan* next = plan->next;
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    plan = next;
  }
}

NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0);
NCCL_PARAM(GraphStreamOrdering, "GRAPH_STREAM_ORDERING", NCCL_CONFIG_UNDEF_INT);

namespace {
enum ncclImplicitOrder {
  ncclImplicitOrderNone,
  ncclImplicitOrderSerial,
  ncclImplicitOrderLaunch
};

// 为真时，NCCL 在捕获期对通信 内核 做内部串行化(走 captureStream 路径)。
static bool ncclGraphStreamOrderingSerialize(struct ncclComm* comm) {
  return comm->config.graphStreamOrdering != 0;
}
} // namespace

static ncclResult_t getImplicitOrder(enum ncclImplicitOrder* mode, bool capturing, int driver = -1) {
  if (ncclParamLaunchOrderImplicit()) {
    if (driver < 0) NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (capturing && driver < 12090) {
      *mode = ncclImplicitOrderSerial;
      return ncclSuccess;
    }
    *mode = 12030 <= std::min<int>(CUDART_VERSION, driver) ? ncclImplicitOrderLaunch : ncclImplicitOrderSerial;
    return ncclSuccess;
  }
  *mode = ncclImplicitOrderNone;
  return ncclSuccess;
}

ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  planner->persistent = persistent;
  // 不同 plan 的操作不会被合并到同一批次。每个新 plan 在调度其操作时都会创建新批次(见
  // ncclAddWorkBatchToPlan)。
  // 对 p2p 操作，我们进一步保证不同 epoch 的操作不会被合并到同一批次(以免挂死)。
  // p2pEpoch 在 scheduleP2pTasksToPlan 中递增，其值会在各 plan 之间延续(即便并非严格必要)
  // 
  int nPlans = 0, p2pEpoch = 0, p2pRound = 0;

  if (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
      !ncclIntruQueueEmpty(&planner->collSymTaskQueue) || !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
      planner->nTasksRma != 0) {
    do {
      memset(&planner->wipPlan, 0, sizeof(planner->wipPlan));

      struct ncclKernelPlan* plan =
        ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan;
      plan->persistent = persistent;
      // 若工作能装下，finishPlan() 会把 ncclDevWorkStorageType[Fifo|Persistent] 提升为 Args 类型。
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent : ncclDevWorkStorageTypeFifo;

      if (planner->nTasksRma != 0) {
        NCCLCHECKGOTO(scheduleRmaTasksToPlan(comm, plan), result, failure);
        if (plan->isRma && plan->rmaArgs != NULL && plan->rmaArgs->nRmaTasks > 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      } else if (!ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
        NCCLCHECKGOTO(scheduleCeCollTaskToPlan(comm, plan), result, failure);
        nPlans += 1;
      } else {
        if (!ncclIntruQueueEmpty(&planner->collSymTaskQueue)) {
          NCCLCHECKGOTO(ncclSymmetricTaskScheduler(comm, &planner->collSymTaskQueue, plan), result, failure);
        } else {
          struct ncclKernelPlanBudget budget;
          budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
          // 非持久的 内核 每次最多只占用 fifo 的一半。
          budget.outArgsBytes = plan->persistent ? (1 << 30) : comm->workFifoBytes / 2;

          // 先排空集合(集合)任务。这一步很关键：因为我们是基于
          // 工作预算来切分任务的，而 p2p 工作不是集合通信。如果先排空 p2p，
          // 各 rank 切分 内核 的位置就可能不一致，进而导致
          // “最短 通道 优先”选择器在不同 rank 上产生不一致的结果。
          if (planner->nTasksColl != 0) {
            NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
          }
          if (planner->nTasksColl == 0 && planner->nTasksBcast != 0) {
            NCCLCHECKGOTO(ncclScheduleBcastTasksToPlan(comm, plan, &budget), result, failure);
          }
          // 只有在集合任务耗尽后，才去排空 p2p 任务。
          if (planner->nTasksColl == 0 && planner->nTasksBcast == 0 && planner->nTasksP2p != 0) {
            NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &p2pEpoch, &p2pRound, plan, &budget), result, failure);
          }
        }

        finishPlan(comm, plan);
        if (plan->workBytes != 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      }
    } while (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
             !ncclIntruQueueEmpty(&planner->collSymTaskQueue) || !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
             planner->nTasksRma != 0);

    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
    planner->unlaunchedPlansHead = planHead;

    if (nPlans == 0) return ncclSuccess;

    cudaStream_t launchStream = planner->streams->stream;
    cudaStream_t deviceStream, launchOrder;
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    bool useLaunchStream = capturing && !ncclGraphStreamOrderingSerialize(comm);

    if (useLaunchStream) {
      // GRAPH_STREAM_ORDERING=0：在 图 的起点(launchStream)上运行 内核，不使用
      // 次级 captureStream。通过在 origin 流上等待 serialEvent 来实现 图 启动串行化，
      // CUDA 在捕获期间允许在 origin 流上做 ExternalWait。
      struct ncclStrongStream* ss = &comm->sharedRes->deviceStream;
      bool firstCapture = !COMPILER_ATOMIC_LOAD(&ss->everCaptured, std::memory_order_relaxed);
      COMPILER_ATOMIC_STORE(&ss->everCaptured, true, std::memory_order_relaxed);
      if (firstCapture) {
        // Bootstrap：在 live 流上给 serialEvent 发信号，使第一个 图 的 ExternalWait
        // 节点能立即触发。这保证了所有 图 的结构完全一致(ExternalWait 始终存在)，
        // 从而使 cudaGraphExecUpdate 能够成功。
        CUDACHECKGOTO(cudaEventRecord(ss->serialEvent, ss->liveStream), result, failure);
      }
      CUDACHECKGOTO(cudaStreamWaitEvent(launchStream, ss->serialEvent, cudaEventWaitExternal), result, failure);
      deviceStream = launchStream;
    } else {
      NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream,
                                            /*concurrent=*/false, &deviceStream),
                    result, failure);
    }

    // userStream[0] 等待每一个 userStream[i]……
    for (struct ncclCudaStreamList* l = planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, l->stream), result, failure);
      CUDACHECKGOTO(cudaStreamWaitEvent(launchStream, comm->sharedRes->scratchEvent, 0), result, failure);
    }
    // userStream[0] 等待 deviceStream(若相同则跳过，避免在 CUDA 图 中产生自环)
    if (deviceStream != launchStream) {
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, deviceStream, comm->sharedRes->scratchEvent), result, failure);
    }

    enum ncclImplicitOrder implicitOrder;
    cudaError_t status = cudaSuccess;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, capturing), result, failure);

    if (implicitOrder != ncclImplicitOrderNone) {
      // userStream[0] 等待每设备(上下文)的 launchOrder。若这是 图 捕获，则需要并发的强流访问；
      // 非捕获场景不能并发，因为那会破坏启动的确定性程序顺序。
      // 
      bool concurrent = capturing;
      if (useLaunchStream) {
        launchOrder = planner->capturingGraph.origin;
      } else {
        NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->context->launchOrder, concurrent,
                                              &launchOrder),
                      result, failure);
      }
      if (launchOrder != launchStream) {
        NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, launchOrder, comm->sharedRes->scratchEvent), result, failure);
      }
    }

    if (!persistent && comm->sharedRes->persistentRefs) {
      status = CUDACLEARERROR(cudaEventQuery(comm->sharedRes->hostStream.serialEvent));
    }
    if (persistent || ncclCudaLaunchBlocking || status == cudaErrorNotReady) {
      // 我们必须启动主机(hos)任务来推送 代理 参数。我们只在确有必要时才这么做，
      // 因为主机任务在 CUDA 中开销很高。
      bool acquired = false;
      cudaStream_t hostStream;
      for (struct ncclKernelPlan* plan = planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream,
                                                  /*concurrent=*/false, &hostStream),
                          result, failure);
          }
          plan->isHostCbEnq = true;
          CUDACHECKGOTO(cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // 让即将启动的 内核 依赖于刚启动的主机流任务。
        NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, hostStream, comm->sharedRes->scratchEvent), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream,
                                              /*concurrent=*/false),
                      result, failure);
      }
    }

    if (persistent) {
      comm->sharedRes->persistentRefs += nPlans;
      comm->localPersistentRefs += nPlans;
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result,
                    failure);
    }
  }
failure:
  return result;
}

ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // 本段代码在进程内屏障登记之后、内核 启动之前被调用。除非 内核 启动被捕获，
  // 否则不允许调用 CUDA。
  // 
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

#if CUDART_VERSION >= 12000
// NCCL 默认使用 "远端" 内存同步域
NCCL_PARAM(MemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);
#endif

// 真正启动 CUDA 内核 的函数（在 组 结束时被调用）。它做三件事：
//   1) ncclLaunchKernelBefore_NoUncapturedCuda：设置 CUDA 图 capture 模式下所需的资源配置；
//   2) 为 plan 中的每个 内核 调用 cudaLaunchKernel（连同 grid/块/共享内存配置），
//      并启动 代理 线程的任务，使 GPU 内核 与后台 代理 协同搬运数据；
//   3) ncclLaunchKernelAfter_NoCuda：记录当次启动的 代理 进度链，供同步/等待。
// 这就是 全规约 真正“跑起来”的那一刻。
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  int nChannels = countOneBits(plan->channelMask);
  void* sym = plan->kernelFn;
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
  cudaStream_t launchStream = planner->streams->stream;

  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs, CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
                   CU_LAUNCH_PARAM_END};

  int driverVersion;
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, do_return);

  CUfunction fn;
  CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), ret, do_return);

  if (CUDART_VERSION >= 11080 && driverVersion >= 11080) {
#if CUDART_VERSION >= 11080
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;

    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[6] = {};
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // grid 维度必须能被 clusterSize 整除
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
#if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // 在 CUDA 12.0 及以上(sm90)上，设置 NCCL 的内存同步域
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain)ncclParamMemSyncDomain();
    }
#endif
#if CUDART_VERSION >= 12030
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, plan->persistent, driverVersion), ret, do_return);
    if (implicitOrder == ncclImplicitOrderLaunch) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
    }
    if (plan->isSymColl && compCap >= 90 && driverVersion >= 12030) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launchAttrs[attrs].value.programmaticStreamSerializationAllowed = 1;
      attrs++;
    }
#endif
#if CUDART_VERSION >= 13000
    if (compCap >= 100 && driverVersion >= 13000) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING;
      launchAttrs[attrs].value.nvlinkUtilCentricScheduling = comm->config.nvlinkCentricSched;
      attrs++;
    }
#endif
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    launchConfig.sharedMemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = launchStream;
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
#endif
  } else {
    // 标准的 内核 启动
    CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr,
                               extra),
                ret, do_return);
  }

do_return:
  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ret;
}

ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!plan->isHostCbEnq) {
    // 我们不使用主机流来提交 代理 操作与回收任务，因此直接调用
    // hostStreamPlanTask
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

namespace {
struct KernelFinishCallback {
  struct ncclCommEventCallback base;
  uint32_t workFifoConsumed;
};
ncclResult_t KernelFinishCallback_fn(struct ncclComm* comm, struct ncclCommEventCallback* cb) {
  struct KernelFinishCallback* me = (struct KernelFinishCallback*)cb;
  comm->workFifoConsumed = me->workFifoConsumed;
  CUDACHECK(cudaEventDestroy(me->base.event));
  free(me);
  return ncclSuccess;
}
} // namespace

ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  if (!ncclIntruQueueEmpty(&planner->planQueue)) {
    // 把队列重置为空，但不销毁 plan —— 这些 plan 会通过 callbackQueue
    // 发回给我们做回收。
    ncclIntruQueueConstruct(&planner->planQueue);

    cudaStream_t launchStream = planner->streams->stream; // First user stream gets launch
    cudaStream_t deviceStream, launchOrder;
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    CUDACHECK(cudaEventRecord(finishedEvent, launchStream));

    if (comm->workFifoProduced - comm->workFifoProducedLastRecorded > comm->workFifoBytes / 8) {
      comm->workFifoProducedLastRecorded = comm->workFifoProduced;
      struct KernelFinishCallback* cb;
      NCCLCHECK(ncclCalloc(&cb, 1));
      cb->base.event = finishedEvent;
      cb->base.fn = KernelFinishCallback_fn;
      cb->workFifoConsumed = comm->workFifoProduced;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cb->base);
      // 我们刚刚“借用”了 scratchEvent，因此必须新建一个。
      CUDACHECK(cudaEventCreateWithFlags(&comm->sharedRes->scratchEvent, cudaEventDisableTiming));
    }

    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    bool useLaunchStream = capturing && !ncclGraphStreamOrderingSerialize(comm);

    if (!useLaunchStream) {
      // deviceStream 等待 userStream[0]
      NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->sharedRes->deviceStream,
                                                   /*concurrent=*/false, &deviceStream));

      // 我们知道 deviceStream 严格落后于 launchStream，因为 launchStream 在启动 内核 前
      // 已与它同步。这使我们能把 deviceStream 等待 launchStream 视为一次“快进”。
      // 在构建 CUDA 图 时，快进需要特殊处理，以免 图 边数爆炸式膨胀。
      // 
      // 因此我们本可以这样做：
      //   CUDACHECK(cudaStreamWaitEvent(deviceStream, finishedEvent, 0));
      // 但我们改用了：
      NCCLCHECK(ncclStreamAdvanceToEvent(planner->capturingGraph, deviceStream, finishedEvent));
    }

    // 每个 userStream[i] 都等待 userStream[0]
    for (struct ncclCudaStreamList* l = planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECK(cudaStreamWaitEvent(l->stream, finishedEvent, 0));
    }
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECK(getImplicitOrder(&implicitOrder, capturing));
    if (implicitOrder != ncclImplicitOrderNone) {
      // 与 ncclLaunchPrepare 中一样，非捕获场景下强流可以非并发。
      bool concurrent = capturing;
      // 把启动事件并入每设备(上下文)的启动顺序。
      // 注意：即便 NCCL_GRAPH_STREAM_ORDERING=0，launchOrder 也不能去掉。
      // 通信域->sharedRes->launchEvent 由 CUDA 通过 CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT 在
      // cuLaunchKernelEx 返回时填充。用户无法查询某条流的“最后一个 内核 的启动事件”，
      // 因此这个顺序依赖永远不能交给用户的流来处理。
      if (useLaunchStream) {
        launchOrder = planner->capturingGraph.origin;
      } else {
        NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->context->launchOrder, concurrent,
                                                     &launchOrder));
      }
      // 若没有启动事件(需 CUDA 12.3)则退而用完成事件(执行串行化)。
      CUDACHECK(cudaStreamWaitEvent(
        launchOrder, implicitOrder == ncclImplicitOrderLaunch ? comm->sharedRes->launchEvent : finishedEvent));
      if (!useLaunchStream) {
        // 释放 ncclLaunchPrepare() 中获取的 launchOrder
        NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->context->launchOrder, concurrent));
      }
    }
    if (!useLaunchStream) {
      NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false));
    } else {
      NCCLCHECK(ncclCudaGraphRecordEvent(planner->capturingGraph, comm->sharedRes->deviceStream.serialEvent,
                                         launchStream));
    }
  }
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

ncclResult_t ncclGetCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* info, int* collNetSupport) {
  // 转换 ncclAvg 与 PreMulSum 算子
  ncclRedOp_t netOp = info->opHost;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    netOp = ncclSum;
  }
  *collNetSupport = comm->config.collnetEnable;
  switch (info->func) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    *collNetSupport &= comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

static void initCollCostTable(float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;
    }
  }
}

// numPipeOps：流水线化的操作数。聚合模式下可大于 1，用于调整延迟。
static ncclResult_t updateCollCostTable(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
                                        int collNetSupport, int nvlsSupport, int numPipeOps, float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  if (comm->nRanks == 1) {
    table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
    return ncclSuccess;
  }

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    // CollNetDirect 最多只支持 8 张本地 GPU
    if (a == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY + 1) continue;
    // 超过 8 张本地 GPU 时禁用 CollNet Chain
    if (a == NCCL_ALGO_COLLNET_CHAIN && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY + 1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) &&
        (!nvlsSupport || (info->func != ncclFuncAllReduce && comm->localRanks > NCCL_MAX_NVLS_ARITY))) {
      continue;
    }
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1) continue;
    /* Tree reduceScatter doesn't support scaling yet */
    if (a == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter &&
        (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv))
      continue;
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &table[a][p]));
      // 把深度足够、会带来精度损失的 fp8 规约树降级为最不优先，
      // 
      if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
        if (a == NCCL_ALGO_RING && comm->nRanks > 8) {
          table[a][p] *= 1024.0; // Any factor large enough to act as a partition between lossy and non-lossy algos.
        }
      }
    }
  }

  return ncclSuccess;
}

/*
 * topoGetAlgoInfo —— 基于拓扑代价模型，为一次集合通信选定“算法 + 协议 + 并行度”
 * ----------------------------------------------------------------------------
 * 这是 NCCL 调度决策的核心。它要回答三个问题：
 *   1) 用哪种算法？ Ring / Tree / CollNet / NVLS / PAT ...
 *   2) 用哪种协议？ Simple / LL / LL128
 *   3) 开多少并行度？ nChannels(线程块数) 与 nThreads(每块线程数)
 *
 * 决策依据是 collCostTable：一张 [算法][协议] 的二维预估耗时表，
 * 由 updateCollCostTable() 根据消息大小、拓扑带宽、延迟模型填充。
 * 本函数只需从中挑出**耗时最小**的那个组合即可。
 *
 * 选定算法后，再根据数据量动态“缩减”并行度：数据量小的时候开太多 channel/线程
 * 反而会因为启动开销和同步开销而变慢，因此要逐步降档。
 */
static ncclResult_t topoGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
                                    float** collCostTable, ncclSimInfo_t* simInfo) {
  // 把一维指针还原成二维数组视图，方便用 table[算法][协议] 的形式访问
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  /* 第一步：遍历整张代价表，选出预估耗时最小的 (算法, 协议) 组合 */
  float minTime = FLT_MAX;
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  int protocol = info->protocol = NCCL_PROTO_UNDEF;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      // IGNORE 表示该组合被显式屏蔽(例如用户用 NCCL_ALGO 环境变量限定了算法)
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
      // 负值表示该组合在当前拓扑/数据类型下不可用，只有非负值才是有效的耗时估计
      if (table[a][p] >= 0.0 && table[a][p] < minTime) {
        algorithm = a;
        protocol = p;
        minTime = table[a][p];
      }
    }
  }

  info->algorithm = algorithm;
  info->protocol = protocol;
  float time = minTime;

  // 确实是“先赋值后检查合法性”，但在这个场景下没有问题(赋的是 取消定义 哨兵值，随后立即校验)。
  // coverity[check_after_sink]
  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    char ncclAlgoEnvStr[1024] = "";
    char ncclProtoEnvStr[1024] = "";
    const char* algoEnv = ncclGetEnv("NCCL_ALGO");
    if (algoEnv) {
      snprintf(ncclAlgoEnvStr, 1023, " NCCL_ALGO was set to %s.", algoEnv);
    }
    const char* protoEnv = ncclGetEnv("NCCL_PROTO");
    if (protoEnv) {
      snprintf(ncclProtoEnvStr, 1023, " NCCL_PROTO was set to %s.", protoEnv);
    }
    WARN("No algorithm/protocol available for function %s with datatype %s.%s%s", ncclFuncToString(info->func),
         ncclDatatypeToString(info->datatype), ncclAlgoEnvStr, ncclProtoEnvStr);
    return (algoEnv || protoEnv) ? ncclInvalidUsage : ncclInternalError;
  }
  if (simInfo) simInfo->estimatedTime = time;
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", nBytes, info->algorithm, info->protocol, time);

  /* 第二步：根据数据量决定并行度(channel 数 nc 与每块线程数 nt)
   *
   * 核心判据是 nBytes < nc * nt * threadThreshold：
   *   等式右边表示“要喂饱当前并行度，至少需要这么多字节”。
   *   如果实际数据量还不到这个量，说明每个线程分到的数据太少，
   *   启动/同步开销会盖过传输收益，此时应当降低并行度。
   */
  int nc = comm->nChannels;                                          // 起始 channel 数 = 通信域可用的全部 channel
  int nt = comm->maxThreads[info->algorithm][info->protocol];        // 该算法/协议下每个 block 的最大线程数
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol]; // 单线程至少应处理的字节数
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet 的 通道 调优：采用“逐级折半”的台阶式收敛(16 -> 8 -> 4 -> 2 -> 1)
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      // 注意乘上了 nHeads：CollNet 每个 通道 有多个 头 并行收发，吞吐能力更强
      while ((flag = nBytes < nc * nt * comm->channels[0].collnetDirect.nHeads * threadThreshold) && nc > ncSwitch) {
        // 降到当前台阶的 1.5 倍位置时，把线程阈值也减半，让收敛更平滑
        if (nc == ncSwitch + ncSwitch / 2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // NVLS(NVLink SHARP)依靠交换机内的硬件规约单元，16 个 通道 就足以打满带宽，
    // 再多开 通道 只会增加开销而没有收益。
    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      nc = std::min(comm->nvlsChannels, comm->nChannels);
    } else {
      nc = comm->nvlsChannels;
    }
  } else {
    // 环/树 的 通道 调优：数据量不够就逐个减少 通道，但至少保留 1 个
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  // 第三步：通道 数减到头之后，如果数据量依然偏小，继续折半减少线程数。
  // 限制 nt % 128 == 0 是为了保证减半后线程数仍是 128 的整数倍(即 线程束 对齐)，
  // 避免出现不完整的 线程束 造成执行效率下降。
  if (info->algorithm != NCCL_ALGO_NVLS && info->algorithm != NCCL_ALGO_NVLS_TREE &&
      info->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  // 第四步：为 Simple 协议追加同步所需的额外线程
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // 额外增加一个 warp 专门负责同步
    // 树 采用了“线程分组(split)”模型：上行组与下行组各自需要同步 线程束，因此追加更多
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4 * WARP_SIZE;
  }
  // 兜底：无论怎么削减，至少保证 3 个 线程束，否则连基本的收/发/同步分工都无法完成
  nt = nt / WARP_SIZE < 3 ? 3 * WARP_SIZE : nt;
  if (info->algorithm == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree 现在恒定使用全部线程
  if (info->algorithm == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;  // PAT 同理，恒定用满
  info->nMaxChannels = nc;
  info->nWarps = nt / WARP_SIZE;   // 对外以 warp 数为单位记录，而非线程数
  return ncclSuccess;
}

/*
 * ncclGetAlgoInfo —— 算法选择的对外总入口
 * ----------------------------------------------------------------------------
 * 若调优插件(tuner plugin)未能给出结果，则退回使用默认的、基于拓扑的调优器。
 * 完整决策顺序为：
 *   1) 先调用插件，允许它设置 算法+协议，和/或 nChannels；
 *   2) 再交给 topoGetAlgoInfo：对插件没设置的部分补齐算法/协议，
 *      并依据最终选定的算法/协议推导出 nChannels 与 nThreads；
 *   3) 最后，如果插件明确指定了 nChannels，则用插件的值覆盖第 2 步的推导结果
 *      (插件的显式意图优先级最高)。
 */
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* info, int collNetSupport, int nvlsSupport,
                             int numPipeOps, ncclSimInfo_t* simInfo /* = NULL*/
) {
  size_t elementSize = ncclTypeSize(info->datatype);
  size_t nBytes = elementSize * ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  struct ncclReg* regSendBuf = NULL;
  struct ncclReg* regRecvBuf = NULL;
  int regBuff;
  bool isSendValid, isRecvValid;
  size_t sendbuffSize = elementSize * ncclFuncSendCount(info->func, comm->nRanks, info->count);
  size_t recvbuffSize = elementSize * ncclFuncRecvCount(info->func, comm->nRanks, info->count);
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  int nMaxChannels = 0;
  float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float**)collCostTable);
  NCCLCHECK(updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float**)collCostTable));
  if (comm->tuner != NULL) {
    NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
    NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
    NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
    NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
    regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) ||
              (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
    NCCLCHECK(comm->tuner->getCollInfo(comm->tunerContext, info->func, nBytes, numPipeOps, (float**)collCostTable,
                                       NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS, regBuff, &nMaxChannels));
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float**)collCostTable, simInfo));
  } else {
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float**)collCostTable, simInfo));
    // NCCL_CTA_POLICY_EFFICIENCY 需要用户(非对称)缓冲区注册(目前 MNNVL 下不支持)
    if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY) && ncclGetEnv("NCCL_ALGO") == NULL &&
        ncclGetEnv("NCCL_PROTO") == NULL && !comm->MNNVL) {
      // 基于缓冲区注册情况来做算法选择
      // 将来还可能有其它针对算法/协议选择的专用策略
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
      NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
      NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
      regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) ||
                (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
      if (regBuff && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
        if ((comm->nNodes > 1 && collNetSupport && nvlsSupport) || (comm->nNodes == 1 && nvlsSupport)) {
          int recChannels;
          NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
          if (recChannels <= info->nMaxChannels) {
            info->algorithm = NCCL_ALGO_NVLS;
            info->protocol = NCCL_PROTO_SIMPLE;
            info->nMaxChannels = recChannels;
            info->nWarps = comm->maxThreads[info->algorithm][info->protocol] / WARP_SIZE;
          }
        }
      }
    }
  }

  info->nMaxChannels = nMaxChannels == 0 ? info->nMaxChannels : nMaxChannels;
  return ncclSuccess;
}

static ncclResult_t calcCollChunking(struct ncclComm* comm, struct ncclTaskColl* info, int nChannels, size_t nBytes,
                                     /*outputs*/ uint32_t* outChunkSize, uint32_t* outDirectFlags,
                                     struct ncclProxyOp* proxyOp) {
  ncclPattern_t pattern;
  size_t grainSize = ncclProtoGrainSize(info->protocol);

  switch (info->func) {
  case ncclFuncBroadcast:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom;
    break;
  case ncclFuncReduce:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo;
    break;
  case ncclFuncReduceScatter:
    pattern = info->algorithm == NCCL_ALGO_PAT            ? ncclPatternPatUp :
              info->algorithm == NCCL_ALGO_NVLS           ? ncclPatternNvls :
              info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
                                                            ncclPatternRing;
    break;
  case ncclFuncAllGather:
    pattern = info->algorithm == NCCL_ALGO_PAT            ? ncclPatternPatDown :
              info->algorithm == NCCL_ALGO_NVLS           ? ncclPatternNvls :
              info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
                                                            ncclPatternRing;
    break;
  case ncclFuncAllReduce:
    pattern = info->algorithm == NCCL_ALGO_NVLS           ? ncclPatternNvls :
              info->algorithm == NCCL_ALGO_NVLS_TREE      ? ncclPatternNvlsTree :
              info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
              info->algorithm == NCCL_ALGO_COLLNET_CHAIN  ? ncclPatternCollnetChain :
              info->algorithm == NCCL_ALGO_TREE           ? ncclPatternTreeUpDown :
                                                            ncclPatternRingTwice;
    break;
  default:
    WARN("Unknown pattern for collective %d algorithm %d", info->func, info->algorithm);
    return ncclInternalError;
  }

  int nstepsPerLoop, nchunksPerLoop;
  size_t loopOffset = 0;
  int stepSize = comm->buffSizes[info->protocol] / NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize * chunkSteps;
  if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  // 基于缓冲区的上限；插件可以把 块 大小提高到此上限。
  int bufferMaxChunkSize = chunkSize;

  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // 优化 chunkSize / nSteps
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) <
             comm->channels[0].collnetDirect.depth * 64 &&
           chunkSize > 131072) {
      chunkSize /= 2;
    }
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) <
             comm->channels[0].collnetDirect.depth * 8 &&
           chunkSize > 65536) {
      chunkSize /= 2;
    }
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) <
             comm->channels[0].collnetDirect.depth * 8 &&
           chunkSize > 32768) {
      chunkSize /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
    stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    chunkSize = std::min(256 * 1024, stepSize * chunkSteps);
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 64 && chunkSize > 131072) {
      chunkSize /= 2;
    }
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 8 && chunkSize > 65536) {
      chunkSize /= 2;
    }
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2;
  } else if (info->algorithm == NCCL_ALGO_NVLS) {
    if ((info->regBufType & NCCL_NVLS_REG_BUFFER) &&
        (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
      chunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    } else {
      int maxChunkSize = comm->nvlsChunkSize;
      if (comm->nNodes > 1 && comm->bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] < 150) {
        maxChunkSize = 32768;
      }
      if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
      // 用 uint64_t 以防 concurrentOps*chunkSize*X 溢出。
      // 不过 nChannels * 通信域->通道[0].NVLS.nHeads 应能轻松装入 32 位。
      // coverity[overflow_before_widen]
      uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
      if ((nBytes < (64 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
      if ((nBytes < (8 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
      if ((nBytes < (2 * (concurrentOps * chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // 用 uint64_t 以防 concurrentOps*chunkSize*X 溢出。
    // 不过 nChannels * 通信域->通道[0].NVLS.nHeads 应能轻松装入 32 位。
    // coverity[overflow_before_widen]
    uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
    chunkSize = std::min(comm->nvlsChunkSize, comm->nvlsTreeMaxChunkSize);
    if ((nBytes < (32 * (concurrentOps * chunkSize))) && (chunkSize > 262144)) chunkSize = 262144;
    if ((nBytes < (16 * (concurrentOps * chunkSize))) && (chunkSize > 131072)) chunkSize = 131072;
    if ((nBytes < (4 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = comm->nNodes;
    float ppn = comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1 + log2i(nNodes) + 0.1 * ppn;
    // 是的，我们可以接受 < 运算符左侧做整数除法，这没有问题。
    // coverity[integer_division]
    while (nBytes / (nChannels * chunkSize) < nstepsLL128 * 64 / ppn && chunkSize > 131072) chunkSize /= 2;
    // coverity[integer_division]
    while (nBytes / (nChannels * chunkSize) < nstepsLL128 * 16 / ppn && chunkSize > 32768) chunkSize /= 2;
  } else if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize * nChannels * 32 > nBytes && chunkSize > 65536) chunkSize /= 2;
  } else if (info->func == ncclFuncReduceScatter && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize * nChannels * 16 > nBytes && chunkSize > 65536) chunkSize /= 2;
  }

  // 计算工作结构体的 directFlags。
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    *outDirectFlags = NCCL_P2P_WRITE;
  } else {
    *outDirectFlags = 0;
  }

  if (comm->tuner != nullptr && comm->tuner->getChunkSize != nullptr) {
    size_t tunerChunkSize = chunkSize;
    NCCLCHECK(comm->tuner->getChunkSize(comm->tunerContext, info->func, nBytes, info->algorithm, info->protocol,
                                        nChannels, &tunerChunkSize));
    if (tunerChunkSize > (size_t)bufferMaxChunkSize) {
      INFO(NCCL_TUNING, "%s: tuner chunk size %zu exceeds buffer max %d, clamping", ncclFuncToString(info->func),
           tunerChunkSize, bufferMaxChunkSize);
      tunerChunkSize = bufferMaxChunkSize;
    }
    chunkSize = (int)tunerChunkSize;
  }

  // 为 代理 计算所需的步数(nSteps)
  chunkSize = chunkSize / grainSize * grainSize; // align chunkSize to multiple grainSize
  switch (pattern) {
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternCollnetChain:
    nstepsPerLoop = nchunksPerLoop = 1;
    break;
  case ncclPatternNvls:
    nstepsPerLoop = 1;
    nchunksPerLoop = comm->channels[0].nvls.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].nvls.headRank;
    break;
  case ncclPatternCollnetDirect:
    nstepsPerLoop = 1;
    nchunksPerLoop = comm->channels[0].collnetDirect.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].collnetDirect.headRank;
    break;
  case ncclPatternRing:
    nstepsPerLoop = comm->nRanks - 1;
    nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternRingTwice:
    nstepsPerLoop = 2 * (comm->nRanks - 1);
    nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternNvlsTree:
    nstepsPerLoop = 1;
    nchunksPerLoop = comm->channels[0].nvls.nHeads;
    break;
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  // 为 代理 计算所需的步数(nSteps)
  size_t loopSize = size_t(nChannels) * nchunksPerLoop * chunkSize;
  int nLoops = (int)DIVUP(nBytes, loopSize);
  memset(proxyOp, 0, sizeof(*proxyOp));
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps;
  proxyOp->loopSize = loopSize;
  proxyOp->loopOffset = loopOffset;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->algorithm = info->algorithm;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    proxyOp->redOp = ncclSum; // Network sees avg as sum
  } else {
    proxyOp->redOp = info->opHost;
  }
  proxyOp->pattern = pattern;
  proxyOp->coll = info->func;
  proxyOp->collAPI = info->func;
  proxyOp->root = info->root;
  proxyOp->isOneRPN = comm->isOneRPN;
  // P2P 用它来缩减接收缓冲区大小。在集合通信里不用它，
  // 因为某些协议传输量会超过总大小，且有时会向上取整，
  // 
  proxyOp->nbytes = stepSize * sliceSteps;

  if (info->regBufType & NCCL_NET_REG_BUFFER) {
    proxyOp->reg = 1;
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT || info->algorithm == NCCL_ALGO_NVLS ||
        info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      if (proxyOp->isOneRPN) {
        proxyOp->nsteps = 1;
        proxyOp->loopOffset = 0;
        proxyOp->sendbuff = (uint8_t*)info->sendbuff;
        proxyOp->sendMhandle = info->sendMhandle;
      } else {
        if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
          proxyOp->nbytes = nBytes / nchunksPerLoop;
          proxyOp->loopSize = proxyOp->loopSize / nchunksPerLoop;
          proxyOp->loopOffset = 0;
          if (info->func == ncclFuncAllGather) {
            proxyOp->sendbuff = (uint8_t*)info->sendbuff;
            proxyOp->sendMhandle = info->sendMhandle;
          }
        } else {
          proxyOp->sendbuff = (uint8_t*)info->recvbuff;
          proxyOp->sendMhandle = info->recvMhandle;
        }
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {
      if (proxyOp->isOneRPN && info->func == ncclFuncAllGather) {
        proxyOp->chunkSize = NCCL_MAX_NET_SIZE;
        proxyOp->sliceSize = NCCL_MAX_NET_SIZE;
        proxyOp->chunkSteps = 1;
        proxyOp->sliceSteps = 1;
        proxyOp->loopSize = size_t(nChannels) * nchunksPerLoop * proxyOp->chunkSize;
        proxyOp->nsteps = DIVUP(nBytes, proxyOp->loopSize) * nstepsPerLoop;
        proxyOp->loopOffset = 0;
      }
    } else {
      WARN("Net registration invalid algorithm %s", ncclAlgoToString(info->algorithm));
      return ncclInternalError;
    }

    proxyOp->recvMhandle = info->recvMhandle;
    proxyOp->recvbuff = (uint8_t*)info->recvbuff;
    proxyOp->nbytes = nBytes;
  } else {
    proxyOp->reg = 0;
  }

  if (pattern == ncclPatternCollnetDirect || pattern == ncclPatternNvls) {
    proxyOp->specifics.collnetDirect.nNodes = comm->nNodes;
    proxyOp->specifics.collnetDirect.node = comm->node;
    if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
      proxyOp->specifics.collnetDirect.sizePerRank = info->count * ncclTypeSize(info->datatype);
    }
  }

  if (pattern == ncclPatternPatUp || pattern == ncclPatternPatDown) {
    proxyOp->nbytes = DIVUP(nBytes, nChannels);
  }

  // 设置供网络插件使用的对端数量提示
  switch (proxyOp->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
    proxyOp->nPeers = 1;
    break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternNvlsTree:
    proxyOp->nPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;
    break;
  case ncclPatternCollnetChain:
  case ncclPatternCollnetDirect:
  case ncclPatternNvls:
  case ncclPatternProfiler:
    // 对端数量提示未使用
    break;
  case ncclPatternSend:
  case ncclPatternRecv:
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  *outChunkSize = proxyOp->chunkSize;
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(ncclDevRedOpFull* opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm* comm) {
  union {
    int8_t i8;
    uint8_t u8;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    __half f16;
    float f32;
    double f64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    __nv_bfloat16 bf16;
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
    __nv_fp8_storage_t f8;
#endif
    void* ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8 * ncclTypeSize(datatype);
  if (nbits <= 0) return ncclInvalidArgument;
  uint64_t allBits = uint64_t(-1) >> (64 - nbits);
  uint64_t signBit = allBits ^ (allBits >> 1);
  bool datatype_signed = false;

  switch (int(op)) {
  case ncclSum:
    opFull->op = ncclDevSum;
    break;
  case ncclProd:
    opFull->op = ncclDevProd;
    break;
  case ncclMin:
  case ncclMax:
    opFull->op = ncclDevMinMax;
    opFull->scalarArg = 0;
    // ncclFuncMinMax<[u]整型> 所用的 xormask 是符号位的异或值：
    // 有符号类型取符号位，求最大值时取所有位(与求最小值相反)。
    if (datatype == ncclInt8 || datatype == ncclInt32 || datatype == ncclInt64) {
      opFull->scalarArg ^= signBit;
    }
    opFull->scalarArg ^= (op == ncclMax) ? allBits : 0;
    break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:
    case ncclInt32:
    case ncclInt64:
      datatype_signed = true;
      // 故意不写 break，让控制流 fall through……
    case ncclUint8:
    case ncclUint32:
    case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks << 1 | datatype_signed;
      break;
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0 / comm->nRanks), __NV_SATFINITE, __NV_E4M3);
      break;
    case ncclFloat8e5m2:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0 / comm->nRanks), __NV_SATFINITE, __NV_E5M2);
      break;
#endif
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0 / comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = __float2bfloat16(float(1.0 / comm->nRanks));
      break;
#endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0 / comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0 / comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp* user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

static ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner* planner = &comm->planner;
  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) {
        // 到了末尾，这必定是一条新的流。
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by "
               "the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // 把流加入列表
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream) break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pTaskAppend(struct ncclComm* comm, struct ncclInfo* info, ncclFunc_t coll, ncclFunc_t collAPI,
                                  void* buff, size_t count, ncclDataType_t datatype, int peer, bool allowUB) {
  struct ncclKernelPlanner* planner = &comm->planner;

  // 确定对端与基本参数。
  ssize_t nBytes = count * ncclTypeSize(datatype);
  bool isSendNotRecv = coll == ncclFuncSend;

  // 在 通信域->memScoped 中分配任务前，必须先进入线程局部 组。
  ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective);
  info->coll = coll;
  // 设置正在捕获的 图。在此调用，以便 剖析器 能发出带此信息的 组 API 事件
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));

  NCCLCHECK(ncclProfilerStartP2pApiEvent(info, isGraphCaptured));

  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);
  p2p->func = coll;
  p2p->collAPI = collAPI;
  p2p->buff = buff;
  p2p->count = count;
  p2p->datatype = datatype;
  p2p->root = peer;
  p2p->bytes = nBytes;
  p2p->allowUB = allowUB;
  p2p->eActivationMask = ncclProfilerApiState.eActivationMask;
  p2p->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  p2p->p2pApiEventHandle = ncclProfilerApiState.p2pApiEventHandle;
  ncclIntruQueueEnqueue(isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue, p2p);
  planner->nTasksP2p += 1;
  if (isSendNotRecv) planner->nTasksP2pSend += 1;
  else planner->nTasksP2pRecv += 1;

  // 标记需要做预连接的 通道
  if (comm->rank != peer) {
    if (!(isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen)) {
      // planner->对等端[对等端].发送/recvSeen 是每个 通信域 私有的，因此无论如何都要设置。
      (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true;
      int round = 0;
      while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank : comm->p2pSchedule[round].recvRank)) {
        round += 1;
      }
      uint8_t base = ncclP2pChannelBaseForRound(comm, round);
      for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, c);
        if (isSendNotRecv) {
          if (comm->channels[channelId].peers[peer]->send[1].hasSeen == 0) {
            // P2P 只使用 1 个连接器
            // 发送/接收 连接器在 split 共享的 通信域 间是共享的。我们需要把 hasSeen 设为 1，
            // 以避免用户在 组 中把 sendrecv 操作与 split 共享 通信域 一起使用时，重复建连。
            // 
            comm->channels[channelId].peers[peer]->send[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->send[1].p2pOnly = 1;
            comm->connectSend[peer] |= (1ULL << channelId);
            ncclGroupCommPreconnect(comm);
          }
        } else {
          if (comm->channels[channelId].peers[peer]->recv[1].hasSeen == 0) {
            // P2P 只使用 1 个连接器
            comm->channels[channelId].peers[peer]->recv[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->recv[1].p2pOnly = 1;
            comm->connectRecv[peer] |= (1ULL << channelId);
            ncclGroupCommPreconnect(comm);
          }
        }
      }
    }
  }
  ncclProfilerStopP2pApiEvent();
  return ncclSuccess;
}

// 把一次 集合（本仓库仅 全规约）任务追加进 通信域->planner。
// 它根据 信息 里的 func（=ncclFuncAllReduce）确定算法(环/树)与协议(LL/LL128/Simple)，
// 计算需要几个 通道、如何把 计数 切分到各 通道，并为每 通道 生成
// ncclDevWorkColl（设备 端工作描述）、选择 内核（devFuncId）以及配置 发送/接收 连接。
// 这是 全规约 从“API 参数”变成“可启动的 内核 任务”的核心函数。
static ncclResult_t collTaskAppend(struct ncclComm* comm, struct ncclInfo* info, struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner* planner = &comm->planner;

  // 在 通信域->memScoped 中分配任务前，必须先进入线程局部 组。
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // 设置正在捕获的 图。在此调用，以便 剖析器 能发出带此信息的 组 API 事件
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));

  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  if (info->coll == ncclFuncBroadcast && ncclParamAllgathervEnable() && !comm->ccEnable) {
    // 在 通信域->memScoped 中分配任务前，必须先进入线程局部 组。
    struct ncclTaskBcast* t =
      ncclMemoryPoolAlloc<struct ncclTaskBcast>(&comm->memPool_ncclTaskBcast, &comm->memPermanent);
    t->func = ncclFuncAllGatherV;
    t->sendbuff = info->sendbuff;
    t->recvbuff = info->recvbuff;
    t->count = info->count * ncclTypeSize(info->datatype);
    t->datatype = ncclInt8;
    t->root = info->root;

    // 更新广播的最小/最大对端
    planner->bcast_info.minBcastPeer = std::min(planner->bcast_info.minBcastPeer, info->root);
    planner->bcast_info.maxBcastPeer = std::max(planner->bcast_info.maxBcastPeer, info->root);
    if (ncclIntruQueueEmpty(&planner->peers[info->root].bcastQueue)) {
      planner->bcast_info.BcastPeers += 1;
    }

    // 入队到对端的广播队列，而非 collSorter
    ncclIntruQueueEnqueue(&planner->peers[info->root].bcastQueue, t);
    planner->nTasksBcast += 1;
  } else {
    struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
    t->func = info->coll;
    t->sendbuff = info->sendbuff;
    t->recvbuff = info->recvbuff;
    t->count = info->count;
    t->root = info->root;
    t->datatype = info->datatype;
    size_t elementSize = ncclTypeSize(t->datatype);
    if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
      t->count *= elementSize;
      t->datatype = ncclInt8;
      elementSize = 1;
    }
    t->trafficBytes = t->count * elementSize * ncclFuncTrafficPerByte(t->func, comm->nRanks);
    t->opHost = info->op;
    t->opDev = opDev; // C++ struct assignment
    t->chunkSteps = info->chunkSteps;
    t->sliceSteps = info->sliceSteps;
    t->eActivationMask = ncclProfilerApiState.eActivationMask;
    t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
    t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;

    planner->nTasksColl += 1;
    ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
  }
  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t ceCollTaskAppend(struct ncclComm* comm, struct ncclInfo* info, struct ncclDevrWindow* sendWin,
                                     struct ncclDevrWindow* recvWin, struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner* planner = &comm->planner;

  // 检查 CE 是否需要初始化
  if (comm->ceColl.baseUCSymReadyPtr == NULL && ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->ceInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // 在 通信域->memScoped 中分配任务前，必须先进入线程局部 组。
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // 设置正在捕获的 图。在此调用，以便 剖析器 能发出带此信息的 组 API 事件
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count * elementSize * ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;
  t->sendWin = sendWin;
  t->recvWin = recvWin;

  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);

  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t rmaTaskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner* planner = &comm->planner;

  void const* srcBuff = info->sendbuff;

  if (!comm->hostRmaSupport) {
    WARN("One sided RMA: host RMA is not supported in this communicator.");
    return ncclInvalidArgument;
  }

  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion < 12050) {
    WARN("One-sided RMA requires CUDA driver 12.5 or later (found %d.%d).", driverVersion / 1000,
         (driverVersion % 1000) / 10);
    return ncclInvalidUsage;
  }

  // 检查上下文是否有效(目前必须为 0)
  if (info->ctx != 0) {
    WARN("Context %d is invalid (must be 0)", info->ctx);
    return ncclInvalidArgument;
  }

  // 检查信号索引是否有效(目前必须为 0)
  if (info->sigIdx != 0) {
    WARN("Signal index %d is invalid (must be 0)", info->sigIdx);
    return ncclInvalidArgument;
  }

  // 检查 标志 是否有效
  if (info->flags != 0) {
    WARN("Flags %u is invalid (must be 0)", info->flags);
    return ncclInvalidArgument;
  }

  // 初始化窗口指针——仅 放置 与 信号 需要
  struct ncclDevrWindow* peerWinHost = NULL;
  struct ncclDevrWindow* srcWinHost = NULL;
  size_t srcWinOffset = 0;

  if (info->coll == ncclFuncPutSignal) {
    // 用详细调试信息校验对端窗口
    if (info->peerWin == NULL) {
      WARN("ncclPutSignal: peerWin is NULL");
      return ncclInvalidArgument;
    }

    struct ncclWindow_vidmem* peerWinDevHost = NULL;
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, info->peerWin, &peerWinDevHost));
    peerWinHost = (struct ncclDevrWindow*)peerWinDevHost->winHost;

    // 校验源缓冲区与窗口
    if (srcBuff == NULL) {
      WARN("ncclPutSignal: srcBuff is NULL");
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclDevrFindWindow(comm, srcBuff, &srcWinHost));
    if (srcWinHost == NULL || !(srcWinHost->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
      WARN("ncclPutSignal: srcWinHost is not in a valid symmetric window");
      return ncclInvalidArgument;
    }
    srcWinOffset = (char*)srcBuff - (char*)srcWinHost->userPtr;

    bool isMultiSegment = ncclDevrWindowIsMultiSegment(srcWinHost) || ncclDevrWindowIsMultiSegment(peerWinHost);
    bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(srcWinHost) || ncclDevrWindowHasSysmemSegment(peerWinHost);

    if (isMultiSegment) {
      WARN("ncclPutSignal currently does not support VAs backed by multiple physical cuMem segments");
      return ncclInvalidArgument;
    }
    if (hasSysmemSegment) {
      WARN("ncclPutSignal currently does not support VAs with host-backed cuMem segments");
      return ncclInvalidArgument;
    }
  } else if (info->coll == ncclFuncSignal) {
    // 检查 计数 是否有效
    if (info->count != 0) {
      WARN("ncclSignal: count must be 0");
      return ncclInvalidArgument;
    }
  } else if (info->coll == ncclFuncWaitSignal) {
    // 检查 signalDescs 是否有效
    if (info->signalDescs == NULL || info->nDesc == 0) {
      WARN("ncclWaitSignal: invalid arguments");
      return ncclInvalidArgument;
    }
    // 校验每个描述符
    for (int i = 0; i < info->nDesc; i++) {
      if (info->signalDescs[i].opCnt <= 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid opCnt %d", i, info->signalDescs[i].opCnt);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].sigIdx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid sigIdx %d (must be 0)", i, info->signalDescs[i].sigIdx);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].ctx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid context %d (must be 0)", i, info->signalDescs[i].ctx);
        return ncclInvalidArgument;
      }
    }
  }

  // 检查 RMA CE 是否需要初始化
  if (!comm->rmaState.rmaCeState.initialized && ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->rmaCeInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // 在 通信域->memScoped 中分配任务前，必须先进入线程局部 组。
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));

  // 单独处理 WaitSignal
  if (info->coll == ncclFuncWaitSignal) {
    struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

    t->func = ncclFuncWaitSignal;
    t->ctx = 0;
    t->count = 0;
    t->bytes = 0;
    t->srcBuff = NULL;
    t->srcWinOffset = 0;
    t->srcWinHost = NULL;
    t->peer = 0;
    t->peerWinOffset = 0;
    t->peerWinHost = NULL;
    t->signalMode = NCCL_SIGNAL;

    // 把描述符转换为 对等端 与 nsignals 数组
    t->npeers = info->nDesc;
    t->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);
    t->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);

    for (int i = 0; i < info->nDesc; i++) {
      t->peers[i] = info->signalDescs[i].peer;
      t->nsignals[i] = info->signalDescs[i].opCnt;
    }

    t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
    planner->nTasksRma++;
    ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);

  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal) {
    // 计算操作的总字节数
    size_t totalBytes = info->count * ncclTypeSize(info->datatype);

    // 定义 1GB 的分块大小，用于拆分大 放置 操作
    const size_t chunkSize = 1ULL << 30; // 1GB = 1073741824 bytes

    // 判断是否需要拆分该操作
    int numChunks = 1;
    if (info->coll == ncclFuncPutSignal && totalBytes > chunkSize) {
      numChunks = (totalBytes + chunkSize - 1) / chunkSize;
    }

    // 为每个分块创建任务
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

      // 计算各分块自身的尺寸与偏移
      size_t chunkBytes = (chunkIdx == numChunks - 1) ? (totalBytes - chunkIdx * chunkSize) : chunkSize;

      size_t chunkOffset = chunkIdx * chunkSize;

      t->func = info->coll;
      t->srcBuff = (const char*)srcBuff + chunkOffset;
      t->srcWinOffset = srcWinOffset + chunkOffset;
      t->srcWinHost = srcWinHost;
      t->count = chunkBytes / ncclTypeSize(info->datatype);
      t->datatype = info->datatype;
      t->bytes = chunkBytes;
      t->ctx = info->ctx;
      t->peer = info->root;
      t->peerWinOffset = info->peerWinOffset + chunkOffset;
      t->peerWinHost = peerWinHost;

      // 信号发送：只有最后一个分块才发信号
      bool isLastChunk = (chunkIdx == numChunks - 1);
      if (isLastChunk) {
        t->signalMode = NCCL_SIGNAL;
      } else {
        // 前面的分块：不发信号
        t->signalMode = NCCL_SIGNAL_NONE;
      }
      t->peers = NULL;
      t->nsignals = NULL;
      t->npeers = 0;

      t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);

      planner->nTasksRma++;
      // 把任务入队到对应的上下文队列
      ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }
  }

  return ncclSuccess;
}

// 把 信息 转换为任务并加入 通信域->planner。唯一的例外是：
// 单 rank 通信域中，集合通信被当作 ncclMemcpyAsync 直接发出，
// 因此不需要任务。
// taskAppend 是任务分类入口：按 信息->func 把请求分派给具体的 *TaskAppend。
// 本仓库只走 ncclFuncAllReduce 分支 -> collTaskAppend(信息, opDev)；
// 其余 P2P/SendRecv/CE/RMA 分支为死代码（不会被触发，保留以便阅读原结构）。
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  ncclFunc_t collAPI = info->coll;

  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    NCCLCHECK(p2pTaskAppend(comm, info, info->coll, collAPI, (void*)info->recvbuff, info->count, info->datatype,
                            info->root, true));
  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal || info->coll == ncclFuncWaitSignal) {
    NCCLCHECK(rmaTaskAppend(comm, info));
  } else {
    // 空的集合通信可以直接丢弃。
    if (info->count == 0) return ncclSuccess;

    if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
      if (comm->minCompCap < 90 && info->coll != ncclFuncAllGather && info->coll != ncclFuncBroadcast &&
          info->coll != ncclFuncAlltoAll && info->coll != ncclFuncScatter && info->coll != ncclFuncGather) {
        WARN("FP8 reduction support begins with sm90 capable devices.");
        return ncclInvalidArgument;
      }
    }

    // 在此把规约算子的状态从 操作 句柄 拷入 信息 结构体，因为
    // 操作 句柄 可能在 ncclGroupEnd() 之前就被销毁。
    struct ncclDevRedOpFull opDev;
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

    if (comm->nRanks == 1) {
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
      return ncclSuccess;
    } else {
      struct ncclDevrWindow* sendWin;
      struct ncclDevrWindow* recvWin;
      ncclDevrFindWindow(comm, info->sendbuff, &sendWin);
      ncclDevrFindWindow(comm, info->recvbuff, &recvWin);
      // 若 CE 受支持且用户请求，则追加 CE 集合任务
      ncclSymRegType_t winRegType;
      NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
      bool ceAvailable = ncclCeAvailable(comm, info->coll, info->op, info->datatype, winRegType);
      bool hierCeAvailable = ncclHierCeAvailable(comm, info->coll, info->op, info->datatype, winRegType);
      bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);

      if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) && (ceAvailable || hierCeAvailable) && !hasSysmemSegment) {
        NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
      }
      // 追加基于 内核 的集合任务
      else {
        // 当前 legacy sendrecv 要求 源 与 目标 缓冲区都已注册，
        // 因此当 alltoall/散播/收集 回退到 legacy sendrecv 时，不能允许 UB(用户缓冲区)。
        // 
        struct ncclReg* sendReg = NULL;
        struct ncclReg* recvReg = NULL;
        bool allowUB = false;
        bool captured = false;
        struct ncclCudaGraph graph;
        // 用于 CUDA 图 检查
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        captured = ncclCudaGraphValid(graph);
        if (info->coll == ncclFuncAlltoAll) {
          NCCLCHECK(ncclRegFind(comm, info->sendbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype),
                                &sendReg));
          NCCLCHECK(ncclRegFind(comm, info->recvbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype),
                                &recvReg));
          allowUB = captured || (sendReg != NULL && recvReg != NULL);
          for (int r = 0; r < comm->nRanks; r++) {
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI,
                                    (void*)((char*)info->sendbuff + r * info->count * ncclTypeSize(info->datatype)),
                                    info->count, info->datatype, r, allowUB));
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI,
                                    (void*)((char*)info->recvbuff + r * info->count * ncclTypeSize(info->datatype)),
                                    info->count, info->datatype, r, allowUB));
          }
        } else if (info->coll == ncclFuncGather) {
          size_t offset = 0;
          allowUB = captured;
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)info->sendbuff, info->count, info->datatype,
                                  info->root, allowUB));
          if (comm->rank == info->root) {
            for (int r = 0; r < comm->nRanks; r++) {
              void* buff = (void*)((char*)info->recvbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, buff, info->count, info->datatype, r,
                                      allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
        } else if (info->coll == ncclFuncScatter) {
          size_t offset = 0;
          allowUB = captured;
          if (comm->rank == info->root) {
            for (int r = 0; r < comm->nRanks; r++) {
              void* buff = (void*)((char*)info->sendbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, buff, info->count, info->datatype, r,
                                      allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)info->recvbuff, info->count, info->datatype,
                                  info->root, allowUB));
        } else if (ceAvailable && comm->symmetricSupport && info->coll == ncclFuncAllGather &&
                   info->count > ncclParamSymCeThreshold() && comm->minCompCap >= 100 && comm->isAllDirectNvlink) {
          // 在 Blackwell 上、大小 > 8MB 的 全收集 使用 CE
          NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
        } else {
          NCCLCHECK(collTaskAppend(comm, info, opDev));
        }
      }
    }
  }

  return ncclSuccess;
}

// ncclAllReduce 等 API 的统一入口：ncclCollectivesEnqueue 在调完参数校验后会调用它。
// 它负责：① 把本次 集合 的相关信息放入 线程-本地 的 组 队列；
// ② 若当前不在 组 内则隐式开启 组，调用 taskAppend 把任务暂存进 planner，
// ③ 组 深度为 1 时 ncclGroupEndInternal 会真正触发内核生成与启动。
// 其后的 collTaskAppend / p2pTaskAppend 才是具体生成 全规约/P2P 任务的地方。
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // 对无效或已被撤销的通信域提前返回
  ncclResult_t ret = CommCheck(info->comm, info->opName, "comm");
  if (ret != ncclSuccess) return ncclGroupErrCheck(ret);
  if (info->comm->revokedFlag) {
    WARN("%s: communicator was revoked", info->opName);
    return ncclGroupErrCheck(ncclInvalidUsage);
  }
  // 性能分析器——若 组 API 事件已开始，则更新 profilerGroupDepth，使
  // 隐式 ncclGroupStartInternal 与 ncclGroupEndInternal 调用的深度能正确更新
  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth++;
  }
  NCCLCHECK(ncclGroupStartInternal());
  ret = ncclSuccess;
  int devOld = -1;
  // 检查通信域是否已就绪、可以通信
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  if (info->comm->checkMode != ncclCheckModeDefault) {
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  // 若 信息->通信域->checkMode == ncclCheckModeDebugGlobal，ArgsCheck 会把 信息
  // 以及 sendrecv 的对端对入队，供后续做全局校验
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,
       "%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
       info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count, info->datatype, info->op,
       info->root, info->comm, info->comm->nRanks, info->stream);
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName,
             reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count,
             info->datatype, info->op, info->root, info->comm, info->stream);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret));
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void)ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
         ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
  NCCLCHECK(CommCheck(comm, "ncclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // 容量翻倍并扩容
    int cap = 2 * comm->userRedOpCapacity;
    if (cap < 4) cap = 4;
    ncclUserRedOp* ops = new ncclUserRedOp[cap];
    if (comm->userRedOpCapacity > 0)
      std::memcpy(ops, comm->userRedOps, comm->userRedOpCapacity * sizeof(ncclUserRedOp));
    for (int ix = comm->userRedOpCapacity; ix < cap; ix++) ops[ix].freeNext = ix + 1;
    delete[] comm->userRedOps;
    comm->userRedOps = ops;
    comm->userRedOpCapacity = cap;
  }
  // 从空闲链表弹出
  int ix = comm->userRedOpFreeHead;
  ncclUserRedOp* user = &comm->userRedOps[ix];
  comm->userRedOpFreeHead = user->freeNext;

  user->freeNext = -1; // allocated
  user->datatype = datatype;
  user->opFull.op = ncclDevPreMulSum;
  if (residence == ncclScalarHostImmediate) {
    int size = ncclTypeSize(datatype);
    if (size < 1) return ncclInternalError;
    user->opFull.scalarArgIsPtr = false;
    std::memcpy(&user->opFull.scalarArg, scalar, size);
  } else {
    user->opFull.scalarArgIsPtr = true;
    user->opFull.scalarArg = reinterpret_cast<uint64_t>(scalar);
  }
  *op = ncclRedOp_t(int(ncclNumOps) + ix);
  *op = ncclUserRedOpMangle(comm, *op);
  TRACE_CALL("ncclRedOpCreatePreMulSum(%d,%p,%d,%d,%p)", *op, scalar, datatype, residence, comm);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  if (0 <= int(op) && int(op) < int(ncclNumOps)) {
    WARN("ncclRedOpDestroy : operator is a NCCL builtin.");
    return ncclInvalidArgument;
  }
  // 整型(ncclMaxRedOp) < 整型(操作) 由于相关数据类型的大小，恒为假，
  // 这本身是设计如此。但我们仍保留该检查，
  // 仅作提醒。
  // coverity[result_independent_of_operands]
  if (int(op) < 0 || int(ncclMaxRedOp) < int(op)) {
    WARN("ncclRedOpDestroy :  operator is garbage.");
    return ncclInvalidArgument;
  }
  if (comm == NULL) {
    WARN("ncclRedOpDestroy : invalid communicator passed.");
    return ncclInvalidArgument;
  }

  int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
  if (comm->userRedOpCapacity <= ix || comm->userRedOps[ix].freeNext != -1) {
    WARN("ncclRedOpDestroy : operator unknown to this communicator.");
    return ncclInvalidArgument;
  }
  // 压入空闲链表
  comm->userRedOps[ix].freeNext = comm->userRedOpFreeHead;
  comm->userRedOpFreeHead = ix;
  TRACE_CALL("ncclRedOpDestroy(%d,%p)", op, comm);
  return ncclSuccess;
}
