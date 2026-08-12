/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/ce_coll.cc — CE(collective engine)聚合引擎实现
 * ----------------------------------------------------------------------------
 * 实现利用 GPU copy engine(CE) 完成集合通信的“聚合引擎”路径：把规约/广播等以 CE
 * 协作方式执行，作为 SIMPLE 协议之外的一种搬运实现，由 include/ce_coll.h 声明接口。
 */

#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "cudawrap.h"
#include "ce_coll.h"
#include "alloc.h"

// 用于图同步的静态常量
static const uint32_t GRAPH_SYNC_VALUE = 1;

// 用于批内(节点内-batch)同步的静态常量，旨在提升大规模下 CE 集合通信的性能
// 批内同步的频率
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// 批内同步的消息阈值
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512 * 1024 * 1024;

// 分层集合中单个子块(sub-块)的最大大小
static constexpr size_t HIER_COLL_MAX_CHUNK_SIZE = 64 * 1024 * 1024;

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  uint8_t* ceDevBase = nullptr;
  // 同步窗口有 lsaSize 个槽(每个 LSA 本地 rank 一个)：一个 就绪 数组 + 一个 完成 数组
  size_t ceDevBaseSize = alignUp(comm->devrState.lsaSize * sizeof(uint32_t), 16) * 2;
  ncclWindow_vidmem* ceWinDev = nullptr;
  ncclWindow_vidmem* ceWinDevHost = nullptr;

  // 确保对称内存运行时已初始化
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // 为对称内存分配并注册
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret,
                fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  NCCLCHECKGOTO(ncclCudaCalloc(&comm->ceColl.ceSeqNumDev, 2, comm->memManager), ret, fail);
  // 从 winHost 字段取出 ncclDevrWindow
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  comm->ceColl.baseUCSymReadyOffset = 0;
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->devrState.lsaSize * sizeof(uint32_t), 16);
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  comm->ceColl.ceSeqNum = 0;
  comm->ceColl.useCompletePtr = false;
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
  NCCLCHECKGOTO(ncclCudaMemcpy(comm->ceColl.ceSeqNumDev + 1, (uint32_t*)&GRAPH_SYNC_VALUE, 1), ret, fail);
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank,
       comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);

exit:
  return ret;
fail:
  ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager);
  // 清理部分初始化结果——两个函数都能安全处理 null
  ncclCommWindowDeregister(comm, ceWinDev);
  ncclMemFree(ceDevBase);
  goto exit;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // 清理 ceInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }

  // 清理 CE 资源——即便出错也继续清理，避免泄漏
  // 注意：两个函数都能安全处理 null
  NCCLCHECKIGNORE(ncclCommWindowDeregister(comm, comm->ceColl.ceSyncWin ? comm->ceColl.ceSyncWin->vidmem : nullptr),
                  ret);
  NCCLCHECKIGNORE(ncclMemFree(comm->ceColl.baseUCSymReadyPtr), ret);
  NCCLCHECKIGNORE(ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager), ret);

  comm->ceColl.ceSeqNumDev = nullptr;
  comm->ceColl.baseUCSymReadyPtr = nullptr;
  comm->ceColl.baseUCSymComplPtr = nullptr;
  comm->ceColl.ceSyncWin = nullptr;

  return ret;
}

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return false;

  // CE 在 CUDA 12.5 及以后版本中支持
  if (driverVersion >= 12050) {
    switch (coll) {
    case ncclFuncAllGather:
    case ncclFuncAlltoAll:
    case ncclFuncScatter:
    case ncclFuncGather:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not implemented");
    return false;
  }
  if (ncclTeamLsa(comm).nRanks < comm->nRanks) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not all ranks have NVLink connectivity");
    return false;
  }
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping CE collective: symmetric support is not enabled");
    return false;
  }
  if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) {
    TRACE(NCCL_TUNING, "Skipping CE collective: window registration type %d is not supported", winRegType);
    return false;
  }
  return true;
}

ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // 等待值要么是图同步的常量值，要么是序列号
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // 使用多播地址作为目标指针
  void* mcDstPtr;
  void* dstPtr = isComplete ? (void*)&completePtrs[myLsaRank] : (void*)&readyPtrs[myLsaRank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // 把更新后的序列号存入设备缓冲区。
  if (!capturing) {
    CUCHECKGOTO(cuStreamWriteValue32(stream, (CUdeviceptr)comm->ceColl.ceSeqNumDev, currentSeq,
                                     CU_STREAM_WRITE_VALUE_DEFAULT),
                ret, fail);
  }

  // 把自己的 就绪/完成 标志写入多播地址
  CUDACHECKGOTO(cudaMemcpyAsync(mcDstPtr, comm->ceColl.ceSeqNumDev + capturing, sizeof(uint32_t),
                                cudaMemcpyDeviceToDevice, stream),
                ret, fail);

  // 为所有其它 rank 添加本地等待操作
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = waitValue;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // 把更新后的序列号存入设备缓冲区。
  if (!capturing) {
    CUCHECKGOTO(cuStreamWriteValue32(stream, (CUdeviceptr)comm->ceColl.ceSeqNumDev, currentSeq,
                                     CU_STREAM_WRITE_VALUE_DEFAULT),
                ret, fail);
  }
  // 用 cudaMemcpyAsync 把自己的 就绪/完成 标志写给远端 rank
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
    void* peerDstPtr;
    void* dstPtr = isComplete ? (void*)&completePtrs[myLsaRank] : (void*)&readyPtrs[myLsaRank];
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    CUDACHECKGOTO(cudaMemcpyAsync(peerDstPtr, comm->ceColl.ceSeqNumDev + capturing, sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice, stream),
                  ret, fail);
  }

  // 为所有其它 rank 添加本地等待操作
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = capturing ? GRAPH_SYNC_VALUE : currentSeq;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

// 通过内存操作完成的 LSA 内部 rank 间同步。
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* profilerArgs) {
  ncclResult_t ret = ncclSuccess;
  void* ceSyncHandle = NULL;
  int lsaSize = comm->devrState.lsaSize;

  // 获取 就绪 与 完成 同步数组的指针
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // 为所有可能的操作分配足够的槽位
  // 对跨 clique 场景，NVLS 多播在 clique 间不可用——改用单播(unicast)同步
  bool useMCSync = comm->nvlsSupport && !comm->p2pCrossClique;
  size_t batchSize = (useMCSync ? NCCL_CE_SYNC_OPS_PER_RANK_MC : NCCL_CE_SYNC_OPS_PER_RANK_UC) * lsaSize;
  size_t opIdx = 0;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  // 启动 CE 同步性能分析(profilerArgs 为 nullptr 时为空操作)
  NCCLCHECKGOTO(ncclProfilerStartCeSyncEvent(comm, profilerArgs, stream, &ceSyncHandle), ret, fail);

  // 为同步准备批处理内存操作
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  if (useMCSync) {
    NCCLCHECKGOTO(ncclPrepMCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  }

  // 对 CUDA 图 捕获，添加重置操作
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    for (int i = 0; i < lsaSize; i++) {
      batchParams[opIdx] = {};
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address =
        (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[opIdx].writeValue.value = 0;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;
    }
  }

  // 把全部内存操作在一个批次中执行
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);

  // 翻转标志位，供下次调用使用
  comm->ceColl.useCompletePtr = !comm->ceColl.useCompletePtr;

exit:
  // 停止 CE 同步性能分析——即使出错也总是尝试停止
  ncclProfilerStopCeSyncEvent(comm, ceSyncHandle, stream);
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity) {
  ncclResult_t ret = ncclSuccess;

  void** srcs = nullptr;
  void** dsts = nullptr;
  size_t* sizes = nullptr;
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes* attrs = nullptr;
  size_t* attrIdxs = nullptr;
#endif

  NCCLCHECKGOTO(ncclCalloc(&srcs, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&dsts, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sizes, capacity), ret, fail);
#if CUDART_VERSION >= 12080
  NCCLCHECKGOTO(ncclCalloc(&attrs, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&attrIdxs, capacity), ret, fail);
#endif

exit:
  params->srcs = srcs;
  params->dsts = dsts;
  params->sizes = sizes;
  params->numOps = 0;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  params->attrs = attrs;
  params->attrIdxs = attrIdxs;
  params->numAttrs = 0;
#endif
  return ret;
fail:
  if (srcs) free(srcs);
  srcs = nullptr;
  if (dsts) free(dsts);
  dsts = nullptr;
  if (sizes) free(sizes);
  sizes = nullptr;
#if CUDART_VERSION >= 12080
  if (attrs) free(attrs);
  attrs = nullptr;
  if (attrIdxs) free(attrIdxs);
  attrIdxs = nullptr;
#endif
  goto exit;
}

void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params->srcs) free(params->srcs);
  params->srcs = nullptr;
  if (params->dsts) free(params->dsts);
  params->dsts = nullptr;
  if (params->sizes) free(params->sizes);
  params->sizes = nullptr;
  params->numOps = 0;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  if (params->attrs) free(params->attrs);
  params->attrs = nullptr;
  if (params->attrIdxs) free(params->attrIdxs);
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif
}

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs) {
  ncclResult_t ret = ncclSuccess;
  bool capturing;
  int driverVersion;
  void* ceBatchHandle = NULL;

  // cudaMemcpyBatchAsync 不接受传统的 null 流(如 PyTorch 的 null 流)。
  // 当 流 为 NULL 时，回退为每个操作单独调用 cudaMemcpyAsync。
  bool isLegacyStream;
  NCCLCHECKGOTO(ncclCudaStreamIsLegacyNull(stream, &isLegacyStream), ret, fail);

  // 启动 CE 批处理性能分析(profilerArgs 为 nullptr 时为空操作)
  NCCLCHECKGOTO(ncclProfilerStartCeBatchEvent(comm, profilerArgs, params, stream, &ceBatchHandle), ret, fail);

  // 检查是否有需要执行的操作
  if (params->numOps == 0) goto exit;

  // 检查是否处于 CUDA 图 捕获中
  capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, fail);

  // --------------CUDA 图 捕获 / 传统流--------------
  // cudaMemcpyBatchAsync 在 CUDA 图 捕获期间、或使用传统流时不支持
  if (capturing || isLegacyStream) {
    for (int i = 0; i < params->numOps; i++) {
      CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                    cudaMemcpyDeviceToDevice, stream),
                    ret, fail);

      if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i + 1) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
      }
    }
    // 变通方案：这是一种规避手段，用于保证批内同步操作的数量始终为偶数，
    // 
    if (params->intraBatchSync &&
        ((params->numOps + comm->ceColl.intraBatchSyncFreq - 1) / comm->ceColl.intraBatchSyncFreq) % 2 == 0) {
      NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
    }
  }
  // --------------无 图 捕获 / 非传统流--------------
  else {
    if (CUDART_VERSION >= 12080 && driverVersion >= 12080) {
#if CUDART_VERSION >= 12080
    // 对 CUDA 12.8 及以上，使用批处理内存拷贝以获得更好性能
      params->attrs[0] = {};
      params->attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      params->attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
      params->attrIdxs[0] = 0;
      params->numAttrs = 1;

      if (params->intraBatchSync) {
      // 找到最大传输尺寸，以决定需要分几轮
        size_t maxSize = 0;
        size_t totalSize = 0;
        for (int i = 0; i < params->numOps; i++) {
          if (params->sizes[i] > maxSize) {
            maxSize = params->sizes[i];
          }
          totalSize += params->sizes[i];
        }

        size_t chunkSize = comm->ceColl.intraBatchSyncMsgThreshold / params->numOps;
        int numRounds = (maxSize + chunkSize - 1) / chunkSize;

        size_t numTmpOps = params->numOps * numRounds;

      // 为所有分块操作分配临时数组
      // 使用 ncclUniqueArrayPtr，使任意退出路径都能自动清理
        ncclUniqueArrayPtr<void*> tmpDsts{nullptr};
        ncclUniqueArrayPtr<void*> tmpSrcs{nullptr};
        ncclUniqueArrayPtr<size_t> tmpSizes{nullptr};

        NCCLCHECKGOTO(ncclCalloc(tmpDsts, numTmpOps), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(tmpSrcs, numTmpOps), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(tmpSizes, numTmpOps), ret, fail);

        int opIdx = 0;
        for (int round = 0; round < numRounds; round++) {
          size_t offset = round * chunkSize;
        // 为这一轮准备各分块传输
          for (int i = 0; i < params->numOps; i++) {
            int index = (i + round) % params->numOps;
            if (offset < params->sizes[index]) {
              size_t remainingSize = params->sizes[index] - offset;
              size_t currentChunkSize = (remainingSize > chunkSize) ? chunkSize : remainingSize;

              tmpDsts[opIdx] = (void*)((uint8_t*)params->dsts[index] + offset);
              tmpSrcs[opIdx] = (void*)((uint8_t*)params->srcs[index] + offset);
              tmpSizes[opIdx] = currentChunkSize;
              opIdx++;
            }
          }
        }

      // 为所有分块启动单个批次
        if (opIdx > 0) {
#if CUDART_VERSION >= 13000
          CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), opIdx, params->attrs,
                                             params->attrIdxs, params->numAttrs, stream),
                        ret, fail);
#else
          CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), opIdx, params->attrs,
                                             params->attrIdxs, params->numAttrs, nullptr, stream),
                        ret, fail);
#endif
        }
      } else {
      // 对所有操作使用单个批次
#if CUDART_VERSION >= 13000
        CUDACHECKGOTO(cudaMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs,
                                           params->attrIdxs, params->numAttrs, stream),
                      ret, fail);
#else
        CUDACHECKGOTO(cudaMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs,
                                           params->attrIdxs, params->numAttrs, nullptr, stream),
                      ret, fail);
#endif
      }
#endif
    } else {
      // 对较旧的 CUDA 版本，回退为逐次单独传输
      for (int i = 0; i < params->numOps; i++) {
        CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                      cudaMemcpyDeviceToDevice, stream),
                      ret, fail);

        if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i + 1) < params->numOps)) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
        }
      }
    }
  }

exit:
  // 停止 CE 批处理性能分析——即使出错也总是尝试停止
  ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream);
  return ret;
fail:
  goto exit;
}

// 在 LSA 团队内做 全收集(仅节点内)。
ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // 在开始传输前，确保所有 rank 都已就绪
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // 若是非原地(出-of-place)操作，把自身数据拷入接收缓冲区
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // 把数据拷给其它 rank
  for (int r = 1; r < lsaSize; r++) {
    int targetRank = (myLsaRank + r) % lsaSize;
    offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // 检查是否需要做批内同步
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                   chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // 启动批处理操作
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // 确保所有 rank 之间的传输都已全部完成
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// 在 LSA 团队内做 AllToAll(仅节点内)。
ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // 计算每个 rank 发给其它每个 rank 的数据量
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // 在开始传输前，确保所有 rank 都已就绪
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // 把数据拷给其它 rank：针对每个目标 rank 发送相应的数据块
  for (int r = 0; r < lsaSize; r++) {
    int dstRank = (myLsaRank + r) % lsaSize;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;

    if (dstRank == myLsaRank) {
      // 对自己数据做本地拷贝
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    } else {
      // 对其它 rank 的远程拷贝：发往 dstRank 的接收缓冲区中、本 通信域->rank 对应的位置
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // 检查是否需要做批内同步
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                   chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // 启动批处理操作
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // 确保所有 rank 之间的传输都已全部完成
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// 在 LSA 团队内做 散播(仅节点内)。
ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // 计算每个 rank 发给其它每个 rank 的数据量
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerDstPtr;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // 在开始传输前，确保所有 rank 都已就绪
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // 检查这是否为原地(入-place)散播 操作
    bool isInPlace = (myRecvBuff == mySendBuff + myLsaRank * chunkBytes);

    // 若非原地，先拷 根 自己的数据
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + myLsaRank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // 根 rank 把数据分发给其它 rank
    for (int r = 1; r < lsaSize; r++) {
      int dstRank = (myLsaRank + r) % lsaSize;
      uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
      uint8_t* dstPtr = isInPlace ? myRecvBuff + dstRank * chunkBytes : myRecvBuff;

      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerDstPtr), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerDstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }
  // 非 根 的 rank 无需执行任何拷贝操作

  // 启动批处理操作
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // 确保所有 rank 之间的传输都已全部完成
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// 在 LSA 团队内做 收集(仅节点内)。
ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  // 计算每个 rank 发给其它每个 rank 的数据量
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // 在开始传输前，确保所有 rank 都已就绪
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // 根 rank 把自身数据拷到接收缓冲区中的正确位置
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // 非 根 的 rank 把各自数据发往 根 的接收缓冲区
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootLsaRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // 启动批处理操作
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // 确保所有 rank 之间的传输都已全部完成
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not implemented");
    return false;
  }
  if (coll != ncclFuncAllGather && coll != ncclFuncAlltoAll) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: only AllGather and AlltoAll are supported");
    return false;
  }

  // 必须是多节点(单节点走常规 CE 路径)
  if (comm->nNodes <= 1) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not multi-node");
    return false;
  }
  // 若 LSA 已覆盖整个 通信域，则改用 CE 路径
  if (ncclDevrIsOneLsaTeam(comm)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA spans the comm; use CE path instead");
    return false;
  }
  // 节点内 CE 散播 通过 LSA 指针写入
  if (ncclTeamLsa(comm).nRanks < comm->localRanks) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA team does not cover all local ranks");
    return false;
  }
  // 需要对称内存支持
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: symmetric support is not enabled");
    return false;
  }
  // 跨节点的 放置 需要 RMA 代理
  if (!comm->hostRmaSupport || comm->config.numRmaCtx == 0) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: RMA proxy not available");
    return false;
  }
  // 发送与接收缓冲区都需要已注册的窗口
  if (winRegType != ncclSymSendRegRecvReg) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: window registration type %d not supported", winRegType);
    return false;
  }
  return true;
}

// 扁平形式的(对端, 块)分块计划。对端 p 的各块
// 覆盖区间 [chunkStart[p], chunkStart[p+1])；总块数 = chunkStart[nPeers]。
struct ncclHierChunkPlan {
  int nPeers;
  int* chunkStart;   // [nPeers + 1]  -- prefix sums
  size_t* chunkBytes;   // [chunkStart[nPeers]]  -- per-chunk byte size
  size_t* chunkOff;     // [chunkStart[nPeers]]  -- per-chunk offset within
                         //                          对等端's perRankBytes slice
};

// 构建统一的分块计划
// 每个对端拿到相同的块列表，每个对端的最后一块吸收余数。
static ncclResult_t ncclHierCollBuildChunk(size_t perRankBytes, int nPeers, size_t maxChunk,
                                           struct ncclHierChunkPlan* outPlan) {
  ncclResult_t ret = ncclSuccess;
  const size_t align = 8 * 1024;

  outPlan->nPeers = nPeers;
  outPlan->chunkStart = nullptr;
  outPlan->chunkBytes = nullptr;
  outPlan->chunkOff = nullptr;

  int numChunks;
  size_t uniformSize, lastChunk;
  if (perRankBytes == 0 || maxChunk == 0 || perRankBytes <= maxChunk) {
    numChunks = 1;
    uniformSize = perRankBytes;
    lastChunk = perRankBytes;
  } else {
    numChunks = (int)((perRankBytes + maxChunk - 1) / maxChunk);
    uniformSize = (perRankBytes / numChunks / align) * align;
    if (uniformSize < align) uniformSize = align;
    lastChunk = perRankBytes - uniformSize * (numChunks - 1);
  }

  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkStart, nPeers + 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkBytes, nPeers * numChunks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkOff, nPeers * numChunks), ret, fail);

  for (int p = 0; p <= nPeers; p++) {
    outPlan->chunkStart[p] = p * numChunks;
  }
  for (int p = 0; p < nPeers; p++) {
    size_t off = 0;
    for (int c = 0; c < numChunks; c++) {
      int idx = p * numChunks + c;
      size_t sz = (c == numChunks - 1) ? lastChunk : uniformSize;
      outPlan->chunkBytes[idx] = sz;
      outPlan->chunkOff[idx] = off;
      off += sz;
    }
  }
exit:
  return ret;
fail:
  free(outPlan->chunkStart);
  outPlan->chunkStart = nullptr;
  free(outPlan->chunkBytes);
  outPlan->chunkBytes = nullptr;
  free(outPlan->chunkOff);
  outPlan->chunkOff = nullptr;
  goto exit;
}

static void ncclHierCollFreeChunkPlan(struct ncclHierChunkPlan* plan) {
  if (plan == nullptr) return;
  free(plan->chunkStart);
  free(plan->chunkBytes);
  free(plan->chunkOff);
  plan->chunkStart = nullptr;
  plan->chunkBytes = nullptr;
  plan->chunkOff = nullptr;
  plan->nPeers = 0;
}

// 分层 CE 集合通信的跨节点 rail-同步 入口屏障。
static ncclResult_t ncclRailSync(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                 struct ncclKernelPlan* plan, int ctx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  bool persistent = plan->persistent;

  // 没有远端节点 -> 无需跨节点屏障；走快速路径空操作。
  if (nRemoteNodes <= 0) return ncclSuccess;

  int* railPeers = nullptr;
  int* railSigOnes = nullptr;
  // 每个 rail 对端一个仅发信号的 放置 操作，打包进单个组描述符。
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* putBatch = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&railPeers, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&railSigOnes, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, nRemoteNodes), ret, fail);

  // 为每个 rail 对端构建一个仅发信号的 放置 操作
  {
    int idx = 0;
    for (int n = 0; n < nNodes; n++) {
      if (n == comm->node) continue;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      railPeers[idx] = railPeer;
      railSigOnes[idx] = 1;

      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent,
                                           /*srcWin=*/nullptr, /*srcOff=*/0,
                                           /*peerWin=*/nullptr, /*peerOff=*/0,
                                           /*size=*/0, railPeer, NCCL_SIGNAL, &groupOps[idx]),
                    ret, fail);
      idx++;
    }
  }

  // 构建组 放置 描述符
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &groupOps, ctx, groupDesc), ret,
                fail);

  // 构建一个等待描述符，覆盖所有 nRemoteNodes 个入站信号。
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &railPeers, &railSigOnes, waitDesc),
                ret, fail);

  // ------------------------------------------------------------------
  // 阶段 1：把组 放置(开始 + 完成)作为一个批次下发。
  // ------------------------------------------------------------------
  {
    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    int putBatchOps = startOps + doneOps;

    NCCLCHECKGOTO(ncclCalloc(&putBatch, putBatchOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, &putBatch[0]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, &putBatch[startOps]), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, putBatchOps, putBatch), ret, fail);
  }

  // ------------------------------------------------------------------
  // 阶段 2：把入站信号等待作为单独的批次下发。
  // ------------------------------------------------------------------
  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(putBatch);
  free(waitBatch);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(groupOps);
  free(railPeers);
  free(railSigOnes);
  return ret;
fail:
  goto exit;
}

// 等待单个对端信号的辅助函数。
static ncclResult_t ncclProxyWaitOnePeer(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                         struct ncclKernelPlan* plan, int ctx, cudaStream_t stream, int peer,
                                         int nsignals) {
  ncclResult_t ret = ncclSuccess;

  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, 1), ret, fail);
  waitPeers[0] = peer;
  waitSigCounts[0] = nsignals;

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, 1, &waitPeers, &waitSigCounts, waitDesc), ret, fail);

  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(waitBatch);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  return ret;
fail:
  goto exit;
}

// 分层 全收集：跨节点 rail 所有-to-所有 + 节点内 CE 散播。
// 每个 rank 的切片被分成块。单个 PutGroup 描述符
// 打包了全部 nRemoteNodes * nChunks 次 放置。
//
// 用户流上的有向无环图(DAG)：
//   RailSync                    // 跨-节点 entry 屏障 (网络 + 等待)
//   PutGroupSubmit              // one memop fires 所有 网络 puts 入 并行的
//   IntraNodeBarrier            // gates LSA 对等端' recvbuf writes; runs 当 代理 is 进行中
//   SelfBcast                   // CE 散播 of 自身的 slice to LSA 对等端
//   for (对等端, 块) 入 shift order:
//     等待 块's 信号; CE-散播 it to 本地 对等端 via LSA
//   PutGroupDone                // one memop 线程块 直到 所有 网络 puts 完成
//   IntraNodeBarrier            // gates 用户 代码 reading recvbuf

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perRankBytes = args->nElts * args->eltSize;

  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  // (对端, 块)粒度的计划。
  struct ncclHierChunkPlan chunkPlan = {};
  // 跨节点 放置-信号 组描述符。
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  // 节点内广播用的批操作临时区。
  struct ncclCeBatchOpsParams ceBcastOps = {};
  // 每块的节点内 CE 散播 用的批操作临时区。
  struct ncclCeBatchOpsParams ceScatterOps = {};

  // ====================================================================
  // 阶段 1：Rail 同步(跨节点入口屏障)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, rmaProxyCtx, plan, ctx, stream), ret, fail);

  // ====================================================================
  // 阶段 2：启动所有跨节点 放置(单个组描述符，已分块)
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perRankBytes, nRemoteNodes, HIER_COLL_MAX_CHUNK_SIZE, &chunkPlan), ret, fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    // 窗口相对偏移
    size_t srcWinOffset = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
    size_t peerWinOffset = ((const uint8_t*)recvbuff + myRank * perRankBytes) - (const uint8_t*)recvWin->userPtr;

    // 分配描述符 + 操作数组
    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node + s) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, sendWin, srcWinOffset + off, recvWin,
                                             peerWinOffset + off, subBytes, railPeer, NCCL_SIGNAL, &groupOps[c]),
                      ret, fail);
      }
    }

    // 构建组描述符
    NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, totalOps, &groupOps, ctx, groupDesc), ret,
                  fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);

    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  // ====================================================================
  // 阶段 3：初始的节点内屏障
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // ====================================================================
  // 阶段 4：自发广播(节点内 CE 对自己块的 广播)
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    // 非原地：把自己的数据拷到自己的 recvbuf 槽位
    if (myRecvSlot != (const uint8_t*)sendbuff) {
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    // 广播给所有其它 LSA 对端
    for (int r = 1; r < lsaSize; r++) {
      int targetLsaRank = (myLsaRank + r) % lsaSize;
      void* peerBuf;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, offset, targetLsaRank, &peerBuf), ret, fail);
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = peerBuf;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // ====================================================================
  // 阶段 5：等待每个(对端, 块) + 节点内 CE 散播(流水线化)
  // ====================================================================
  {
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      size_t peerSliceOffset = railPeer * perRankBytes;

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        uint8_t* chunkSlot = (uint8_t*)recvbuff + peerSliceOffset + off;
        size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

        // ----- 等待 railPeer 发来该子块的信号 -----
        NCCLCHECKGOTO(ncclProxyWaitOnePeer(comm, rmaProxyCtx, plan, ctx, stream, railPeer, /*nsignals=*/1), ret, fail);

        // ----- 把该子块 CE 散播 给所有其它 LSA 对端 -----
        NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, lsaSize), ret, fail);
        for (int r = 1; r < lsaSize; r++) {
          int targetLsaRank = (myLsaRank + r) % lsaSize;
          void* peerBuf;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, winOffset, targetLsaRank, &peerBuf), ret, fail);
          ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
          ceScatterOps.dsts[ceScatterOps.numOps] = peerBuf;
          ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
          ceScatterOps.numOps++;
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
        ncclCeFreeBatchOpsParams(&ceScatterOps);
      }
    }
  }

  // ====================================================================
  // 阶段 6：等待所有出站数据 放置 完成
  // ====================================================================
  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  // ====================================================================
  // 阶段 7：最终的节点内屏障
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  }
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

// 分层 AllToAll：跨节点 alltoall + 节点内 CE alltoall。
// 用户流上的有向无环图(DAG)：
//   RailSync                    // rail-仅 entry 屏障
//   IntraNodeBarrier #1         // 所有 ranks 入 同步
//   PutGroupSubmit              // one memop fires 所有 放置 操作
//   IntraNodeAlltoAll           // 批处理的 CE alltoall
//   AggregateWait               // 单个 multi-对等端 等待 descriptor covering 所有 远端 对等端
//   PutGroupDone                // one memop 线程块 直到 outbound puts 已完成
//   IntraNodeBarrier #2         // 所有 ranks 入 同步

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int ctx = 0;
  int myRank = comm->rank;
  int myNode = comm->node;
  int nNodes = comm->nNodes;
  int localRanks = comm->localRanks;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  int numRemotePeers = (nNodes - 1) * localRanks;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perPeerBytes = args->nElts * args->eltSize;
  bool inPlace = (sendbuff == recvbuff);

  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  // 跨节点 放置-信号 组的块计划。
  struct ncclHierChunkPlan chunkPlan = {};
  // 跨节点 放置-信号 组描述符。
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  CUstreamBatchMemOpParams* groupStartParam = nullptr;
  CUstreamBatchMemOpParams* groupDoneParam = nullptr;
  // 聚合入站等待描述符(覆盖所有远端对端)。
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;
  // 节点内 alltoall 临时区。
  struct ncclCeBatchOpsParams ceLocalA2A = {};

  // ====================================================================
  // 阶段 1：Rail 同步(仅 rail 的跨节点入口屏障)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, rmaProxyCtx, plan, ctx, stream), ret, fail);

  // ====================================================================
  // 阶段 2：节点内屏障
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // ====================================================================
  // 阶段 3：构建并提交 放置-信号 组(起始 memop)。
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perPeerBytes, numRemotePeers, HIER_COLL_MAX_CHUNK_SIZE, &chunkPlan), ret,
                  fail);
    int totalOps = chunkPlan.chunkStart[chunkPlan.nPeers];

    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCalloc(&groupStartParam, startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParam, doneOps), ret, fail);

    NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupOps, totalOps), ret, fail);

    int p = 0;  // chunk plan slot index
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode + s) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        int peer = comm->nodeRanks[n].localRankToRank[lr];
        size_t srcWinOffset =
          ((const uint8_t*)sendbuff + (size_t)peer * perPeerBytes) - (const uint8_t*)sendWin->userPtr;
        size_t peerWinOffset =
          ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

        for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
          size_t subBytes = chunkPlan.chunkBytes[c];
          size_t off = chunkPlan.chunkOff[c];

          NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, sendWin, srcWinOffset + off, recvWin,
                                               peerWinOffset + off, subBytes, peer, NCCL_SIGNAL, &groupOps[c]),
                        ret, fail);
        }
        p++;
      }
    }

    NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, totalOps, &groupOps, ctx, groupDesc), ret,
                  fail);

    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, groupStartParam), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, groupDoneParam), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, startOps, groupStartParam), ret, fail);
  }

  // ====================================================================
  // 阶段 4：节点内 alltoall(基于 LSA 的批处理 CE 拷贝)。
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceLocalA2A, lsaSize), ret, fail);
  {
    size_t myRecvOffset = ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

    for (int k = 0; k < lsaSize; k++) {
      int targetLsa = (myLsaRank + k) % lsaSize;
      int targetWorldRank = comm->nodeRanks[myNode].localRankToRank[targetLsa];

      if (inPlace && targetLsa == myLsaRank) continue;

      void* peerRecvSlot;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, myRecvOffset, targetLsa, &peerRecvSlot), ret, fail);

      ceLocalA2A.srcs[ceLocalA2A.numOps] = (void*)((const uint8_t*)sendbuff + (size_t)targetWorldRank * perPeerBytes);
      ceLocalA2A.dsts[ceLocalA2A.numOps] = peerRecvSlot;
      ceLocalA2A.sizes[ceLocalA2A.numOps] = perPeerBytes;
      ceLocalA2A.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceLocalA2A, stream, args), ret, fail);
  }

  // ====================================================================
  // 阶段 5：对所有远端对端的聚合等待。
  // ====================================================================
  {
    NCCLCHECKGOTO(ncclCalloc(&waitPeers, numRemotePeers), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, numRemotePeers), ret, fail);

    int p = 0;
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode - s + nNodes) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        waitPeers[p] = comm->nodeRanks[n].localRankToRank[lr];
        waitSigCounts[p] = chunkPlan.chunkStart[p + 1] - chunkPlan.chunkStart[p];
        p++;
      }
    }

    NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, numRemotePeers, &waitPeers, &waitSigCounts,
                                            waitDesc),
                  ret, fail);

    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

  // ====================================================================
  // 阶段 6：PutGroupDone memop(出站 放置 已在链路上完成)。
  // ====================================================================
  {
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, doneOps, groupDoneParam), ret, fail);
  }

  // ====================================================================
  // 阶段 7：节点内屏障
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceLocalA2A);
  free(groupStartParam);
  free(groupDoneParam);
  free(groupOps);
  if (groupDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  }
  free(waitBatch);
  if (waitDesc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  }
  free(waitPeers);
  free(waitSigCounts);
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  // 启动 CE 集合通信性能分析
  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream), ret, fail);

  // 分层路径：跨节点 RMA + 节点内 CE
  // 用 ncclDevrIsOneLsaTeam 而非 通信域->nNodes，因为多 clique 单 NVLD 场景也应走 CE 路径
  if (!ncclDevrIsOneLsaTeam(comm)) {
    switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclHierCeAllGather(comm, plan, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclHierCeAlltoAll(comm, plan, stream), ret, fail);
      break;
    default:
      WARN("Hierarchical CE collective not supported for %s", ncclFuncToString(args->func));
      ret = ncclInvalidUsage;
    }
  }
  // LSA 本地的 CE 路径
  else {
    switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
      break;
    case ncclFuncScatter:
      NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
      break;
    case ncclFuncGather:
      NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
    }
  }

exit:
  // 停止 CE 集合通信性能分析——即使出错也总是尝试停止
  ncclProfilerStopCeCollEvent(comm, args, stream);
  return ret;
fail:
  goto exit;
}

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);

  plan->isCeColl = true;
  plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
  plan->ceCollArgs->rootRank = task->root;
  plan->ceCollArgs->datatype = task->datatype;
  plan->ceCollArgs->nElts = task->count;
  plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);
  plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;
  plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;
  plan->ceCollArgs->func = task->func;
  plan->ceCollArgs->sendWin = task->sendWin;
  plan->ceCollArgs->recvWin = task->recvWin;
  plan->ceCollArgs->collApiEventHandle = task->collApiEventHandle;

  if (comm->rank == 0) {
    if (!ncclDevrIsOneLsaTeam(comm)) {
      INFO(NCCL_TUNING, "%s [Hierarchical CE]: %ld Bytes -> RMA proxy + CE", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype));
    } else {
      const char* nvlsSync = comm->nvlsSupport ? "; CE synchronization with NVLS" : "";
      INFO(NCCL_TUNING, "%s [Copy Engine]: %ld Bytes -> cudaMemcpy%s", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype), nvlsSync);
    }
  }

  ncclIntruQueueEnqueue(&planner->planQueue, plan);
  ncclIntruQueueDequeue(&planner->collCeTaskQueue);
  ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);

  return ncclSuccess;
}
