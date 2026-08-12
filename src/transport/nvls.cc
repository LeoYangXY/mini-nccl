/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/transport/nvls.cc — NVLS(NVLink SHARP)传输实现
 * ----------------------------------------------------------------------------
 * 实现经 NVSwitch 的多播/聚合传输：通过 NVLS 把数据在交换机层直接规约/广播，减少
 * GPU 间往返。注意：mini-nccl 当前多为单节点 2 卡 P2P，本文件可能整体被 #if 0 屏蔽。
 */

// NVLink SHARP(NVLS)传输层的实现

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"
#include "enqueue.h"
#include "register.h"
#include "transport.h"
#include "register_inline.h"

#if 0 /* mini-nccl: NVLS support removed (single-node 2-GPU P2P only) */

struct graphRegData {
  uintptr_t offset;
  size_t size;
};

struct localRegData {
  struct ncclReg reg;
  intptr_t offset;
  int handleTypes;
};

ncclResult_t nvlsCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1,
                            struct ncclPeerInfo* info2) {
  // 该传输层不能用于 P2P
  *ret = 0;
  return ncclSuccess;
}

ncclResult_t nvlsSendFree(struct ncclComm* comm, struct ncclConnector* send) {
  return ncclSuccess;
}

ncclResult_t nvlsRecvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  return ncclSuccess;
}

struct ncclTransport nvlsTransport = {"NVLS",
                                      nvlsCanConnect,
                                      {NULL, NULL, nvlsSendFree, NULL, NULL, NULL, NULL, NULL},
                                      {NULL, NULL, nvlsRecvFree, NULL, NULL, NULL, NULL, NULL}};

ncclResult_t ncclNvlsGroupCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
                                 CUmemGenericAllocationHandle* mcHandle, char* shareableHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  size_t size = prop->size;

  // 创建一个多播(Multicast)组

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zu on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(mcHandle, prop));

  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
    // 获取一个句柄，传给其它 rank
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, *mcHandle, ncclCuMemHandleType, 0));
  } else {
    memcpy(shareableHandle, mcHandle, sizeof(CUmemGenericAllocationHandle));
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zu on rank %d", *mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t ncclNvlsGroupConnect(struct ncclComm* comm, char* shareableHandle, int rank,
                                  CUmemGenericAllocationHandle* mcHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  int fd = -1;
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // 导入远端内存描述符，并把它映射到本地 GPU 的地址空间
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem 的 UDS(Unix 域套接字)支持
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank,
          *(uint64_t*)shareableHandle, rank);
    NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, rank, shareableHandle, &fd), ret, fail);
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void*)(uintptr_t)fd, type), ret, fail);
    SYSCHECK(close(fd), "close");
  } else {
    if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void*)shareableHandle, type), ret, fail);
    } else {
      memcpy(mcHandle, shareableHandle, sizeof(CUmemGenericAllocationHandle));
    }
  }
exit:
  return ret;
fail:
  if (fd != -1) close(fd);
  goto exit;
}

ncclResult_t nvlsGroupUnbind(struct ncclComm* comm, size_t size, CUmemGenericAllocationHandle* mcHandle) {
  int dev = comm->cudaDev;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zu dev %d", *mcHandle, size, dev);

  // 把给定设备的物理内存从组中解绑
  if (size) CUCHECK(cuMulticastUnbind(*mcHandle, dev, 0 /*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle* mcHandler, CUdeviceptr ptr,
                                 int dev, size_t ucsize, size_t mcsize) {
  // 若该缓冲区已被用户释放，unbind 可能触发 RM 错误
  // 不过忽略该错误是安全的，unbind 最终仍会成功
  CUCALL(cuMulticastUnbind(*mcHandler, dev, 0 /*mcOffset*/, ucsize));
  CUCHECK(cuMemUnmap(ptr, mcsize));
  CUCHECK(cuMemAddressFree(ptr, mcsize));
  CUCHECK(cuMemRelease(*mcHandler));
  INFO(NCCL_NVLS, "rank %d - NVLS deregistered buffer %p on device %d ucsize %ld mcsize %ld", comm->rank, (void*)ptr,
       dev, ucsize, mcsize);
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnmapMem(struct ncclComm* comm, size_t ucsize, void* ucptr,
                               CUmemGenericAllocationHandle* ucHandle, size_t mcsize, void* mcptr,
                               CUmemGenericAllocationHandle* mcHandle) {
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) ucsize %zu MC handle 0x%llx(%p) mcsize %zd", *ucHandle, ucptr,
       ucsize, *mcHandle, mcptr, mcsize);

  // 释放 UC(单播)内存及其映射
  if (ucptr) {
    CUCHECK(cuMemUnmap((CUdeviceptr)ucptr, ucsize));
    CUCHECK(cuMemAddressFree((CUdeviceptr)ucptr, ucsize));
    CUCHECK(cuMemRelease(*ucHandle));
  }

  // 释放 MC(多播)内存及其映射
  if (mcptr) {
    CUCHECK(cuMemUnmap((CUdeviceptr)mcptr, mcsize));
    CUCHECK(cuMemAddressFree((CUdeviceptr)mcptr, mcsize));
    CUCHECK(cuMemRelease(*mcHandle));
  }

  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define NVLS_MEM_ALIGN_SIZE (1 << 21)
#define NVLS_NCHANNELS_SM90 16
#define NVLS_NCHANNELS_SM100 32
#define NVLS_NCHANNELS_SM100_NVL 24

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM(NvlsChunkSize, "NVLS_CHUNKSIZE", 128 * 1024);
NCCL_PARAM(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

// 返回 SM100 多节点配置下的最优 NVLSTree 调优参数。
static ncclResult_t ncclNvlsTreeSm100Tuning(struct ncclComm* comm, int* nChannels, int* chunkSize,
                                            int* treeMaxChunkSize) {
  int nNodes = comm->nNodes;
  int ppn = comm->minLocalRanks;
  float nicBw = comm->minNetBw;
  int gpuToNicPathType = comm->graphs[NCCL_ALGO_NVLS].typeInter;

  *chunkSize = 128 * 1024;

  if (nNodes == 2 && nicBw >= 48.0f) {
    *nChannels = 32;
    *treeMaxChunkSize = 128 * 1024;
    if (ppn <= 4) {
      *chunkSize = 256 * 1024;
      *treeMaxChunkSize = 256 * 1024;
    } else if (ppn >= 16 || (ppn <= 8 && gpuToNicPathType <= PATH_PXB && nicBw < 96.0f)) {
      *treeMaxChunkSize = 64 * 1024;
    }
  } else if (nicBw >= 96.0f) {
    if (ppn <= 8) {
      *nChannels = 24;
      *chunkSize = 256 * 1024;
      *treeMaxChunkSize = 256 * 1024;
    } else {
      *nChannels = 32;
      *treeMaxChunkSize = (ppn < 32) ? 128 * 1024 : 64 * 1024;
    }
  } else if (nicBw >= 48.0f) {
    *nChannels = 24;
    *treeMaxChunkSize = 128 * 1024;
    if (gpuToNicPathType <= PATH_PXB) {
      *treeMaxChunkSize = 64 * 1024;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  int nChannels;
  int chunkSize = 0;
  int treeMaxChunkSize = 0;
  const char* chunkSizeEnv = ncclGetEnv("NCCL_NVLS_CHUNKSIZE");
  bool userSetChunkSize = (chunkSizeEnv != NULL && strlen(chunkSizeEnv) > 0);

  // 根据 SM 架构设置默认的 通道 数量
  if (comm->compCap >= 100) {
    nChannels = (comm->nNodes > 1) ? NVLS_NCHANNELS_SM100 : NVLS_NCHANNELS_SM100_NVL;
  } else {
    nChannels = NVLS_NCHANNELS_SM90;
  }

  // SM100 多节点 NVLSTree 调优(可能调整全部三个值)
  if (comm->minCompCap >= 100 && comm->nNodes > 1) {
    NCCLCHECK(ncclNvlsTreeSm100Tuning(comm, &nChannels, &chunkSize, &treeMaxChunkSize));
  }

  // 用户的覆盖设置优先于自动调优
  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT) nChannels = comm->config.nvlsCTAs;
  // 若用户已设置 块 大小、或 chunkSize 未设置，则使用 ncclParamNvlsChunkSize() 确定的值
  if (userSetChunkSize || chunkSize == 0) chunkSize = ncclParamNvlsChunkSize();

  // 确定最终的 treeMaxChunkSize：环境变量 > 自动调优 > 兜底值
  int envTreeMaxChunkSize = (int)ncclParamNvlsTreeMaxChunkSize();
  if (envTreeMaxChunkSize == -2 && treeMaxChunkSize == 0) {
    treeMaxChunkSize = (comm->nNodes >= 4) ? 65536 : chunkSize;
  } else if (envTreeMaxChunkSize != -2) {
    treeMaxChunkSize = envTreeMaxChunkSize;
  }

  // 把 nvlsChannels 钳制到 [minCTAs, maxCTAs] 区间内
  nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, nChannels));

  // 应用最终取值
  comm->nvlsChannels = nChannels;
  comm->nvlsChunkSize = chunkSize;
  comm->nvlsTreeMaxChunkSize = treeMaxChunkSize;

  INFO(NCCL_INIT, "NVLS tuning: nChannels %d chunkSize %d treeMaxChunkSize %d", comm->nvlsChannels, comm->nvlsChunkSize,
       comm->nvlsTreeMaxChunkSize);

  return ncclSuccess;
}

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsSupport = 0;
  comm->nvlsChannels = 0;

  if (comm->hasMultiRankNvml) {
    if (ncclParamNvlsEnable() == 1) {
      WARN("NCCL_NVLS_ENABLE has been set to \"1\" and communicator has multiple ranks using the same NVML device. "
           "This is not compatible with NCCL_NVLS_ENABLE=1.");
      return ncclInvalidUsage;
    }
    return ncclSuccess;
  }
  int gpuCount;
  NCCLCHECK(ncclTopoGetGpuCount(comm->topo, &gpuCount));
  if (!ncclParamNvlsEnable() || gpuCount < 2) return ncclSuccess;

  CUdevice dev;
  int driverVersion;

  if (CUPFN(cuDeviceGet) == NULL) return ncclSuccess;
  CUCHECK(cuCtxGetDevice(&dev));
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (ncclParamNvlsEnable() == 2) {
    // NVLS 多播支持需要 CUDA12.1 的用户态驱动(UMD)与内核态驱动(KMD)
    if (CUPFN(cuMulticastCreate) != NULL /*&& driverVersion >= 12010 */) {
      CUCHECK(cuDeviceGetAttribute(&comm->nvlsSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    }
  } else {
    comm->nvlsSupport = 1;
  }

  INFO(NCCL_INIT, "NVLS multicast support is %savailable on dev %d", comm->nvlsSupport ? "" : "not ", dev);
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nvlsSupport && comm->nNodes > 1) {
    for (int c = 0; c < comm->nvlsChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1,
                                            &channel->nvls.treeUp, 0),
                    ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY,
                                            channel->nvls.treeDown, 0),
                    ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_NVLS], 0), ret, fail);
    INFO(NCCL_INIT, "Connected NVLS tree");
  }
exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t nvlsAllocateMem(struct ncclComm* comm, const CUmemAccessDesc* desc, size_t size,
                                    CUmemGenericAllocationHandle* ucHandle, CUmemGenericAllocationHandle* mcHandle,
                                    void** ucptr, void** mcptr, size_t* ucsizePtr, size_t* mcsizePtr) {
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  CUresult err;
  ncclResult_t ret = ncclSuccess;
  size_t mcsize;
  size_t ucsize;
  size_t ucgran, mcgran;
  int allocMcHandle = 0;

  mcsize = ucsize = size;
  *ucptr = *mcptr = NULL;
  memset(shareableHandle, '\0', sizeof(shareableHandle));
  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = size;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, mcHandle, shareableHandle), ret,
                  fail);
    allocMcHandle = 1;
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], mcHandle), ret, fail);
    allocMcHandle = 1;
  }

  CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ret, fail);

  memset(&ucprop, 0, sizeof(CUmemAllocationProp));
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ucprop.location.id = comm->cudaDev;
  ucprop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(ucsize, ucgran);
  // 为 UC 内存映射一个满足 MC 对齐与大小要求的虚拟地址(VA)
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)ucptr, ucsize, ucgran, 0U, 0), ret, fail);

  // 为本 NVLS 组分配本地物理内存
  CUCHECKGOTO(cuMemCreate(ucHandle, ucsize, &ucprop, 0), ret, fail1);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ucptr, ucsize, 0, *ucHandle, 0), ret, fail2);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ucptr, ucsize, desc, 1), ret, fail3);
  CUDACHECKGOTO(cudaMemset(*ucptr, 0, ucsize), ret, fail3);
  // 把 NVLS 缓冲区登记为持久内存
  NCCLCHECKGOTO(ncclMemTrack(comm->memManager, *ucptr, ucsize, *ucHandle, ncclCuMemHandleType, ncclMemPersist), ret,
                fail3);

  // 节点内屏障，用于缓解 中止 时 cuMulticastBindMem 可能卡死的问题
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail3);
  // 把物理内存绑定到多播组
  // 注意：它会阻塞，直到所有 rank 都被加入该组
  // 如果系统的 NVLS/多播支持有问题，通常就是在这里暴露出来
  err = CUPFN(cuMulticastBindMem(*mcHandle, 0 /*mcOffset*/, *ucHandle, 0 /*memOffset*/, ucsize, 0 /*flags*/));
  if (err != CUDA_SUCCESS) {
    const char* errStr;
    (void)pfn_cuGetErrorString(err, &errStr);    // Fail the job as NVLS support is not functional
    WARN("Failed to bind NVLink SHARP (NVLS) Multicast memory of size %ld : CUDA error %d '%s'.\nThis is usually "
         "caused by a system or configuration error in the Fabric Manager or NVSwitches.\nDisable NVLS "
         "(NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.",
         ucsize, err, errStr);
    ret = ncclUnhandledCudaError;
    goto fail3;
  }

  // 映射多播(mc)虚拟地址
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)mcptr, mcsize, mcgran, 0U, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*mcptr, mcsize, 0, *mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*mcptr, mcsize, desc, 1), ret, fail);
  *ucsizePtr = ucsize;
  *mcsizePtr = mcsize;

  INFO(
    NCCL_NVLS,
    "NVLS rank %d (dev %d) alloc done, ucptr %p ucgran %ld mcptr %p mcgran %ld ucsize %ld mcsize %ld (inputsize %ld)",
    comm->rank, comm->cudaDev, *ucptr, ucgran, *mcptr, mcgran, ucsize, mcsize, size);

exit:
  return ret;
fail3:
  CUCHECK(cuMemUnmap((CUdeviceptr)*ucptr, ucsize));
fail2:
  CUCHECK(cuMemRelease(*ucHandle));
fail1:
  CUCHECK(cuMemAddressFree((CUdeviceptr)*ucptr, ucsize));
fail:
  if (allocMcHandle && *mcptr == NULL && *ucptr == NULL) CUCHECK(cuMemRelease(*mcHandle));
  goto exit;
}

ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  int nHeads = -1;
  int headRank = -1;
  ncclResult_t res = ncclSuccess;
  int nvlsStepSize = -1;
  size_t buffSize = 0;
  size_t nvlsPerRankSize = 0;
  size_t nvlsTotalSize = 0;
  struct ncclNvlsSharedRes* resources = NULL;
  int nChannels = -1;
  cudaStream_t deviceStream, hostStream;

  if (comm->nvlsSupport == 0 || comm->nvlsResources->inited) return ncclSuccess;
  // 在检查过 通信域->nvlsSupport 之后再初始化
  nHeads = comm->channels[0].nvls.nHeads;
  headRank = comm->channels[0].nvls.headRank;
  resources = comm->nvlsResources;
  nChannels = comm->nvlsChannels;
  nvlsStepSize = comm->nvlsChunkSize;
  buffSize = nvlsStepSize * NCCL_STEPS;
  nvlsPerRankSize = nChannels * 2 * buffSize;
  nvlsTotalSize = nvlsPerRankSize * nHeads;

  INFO(NCCL_INIT | NCCL_NVLS,
       "NVLS comm %p headRank %d nHeads %d nvlsRanks %d buffSize %zu nvlsPerRankSize %zu nvlsTotalSize %zu", comm,
       headRank, nHeads, comm->localRanks, buffSize, nvlsPerRankSize, nvlsTotalSize);

  NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, nvlsTotalSize, &resources->ucBuffHandle,
                                &resources->mcBuffHandle, (void**)&resources->ucBuff, (void**)&resources->mcBuff,
                                &resources->buffUCSize, &resources->buffMCSize),
                res, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                        /*concurrent=*/false, &hostStream),
                res, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false, &deviceStream),
                res, fail);
  for (int h = 0; h < nHeads; h++) {
    int nvlsPeer = comm->nRanks + 1 + h;
    for (int c = 0; c < nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

      // 把 UC 数据规约到 MC(单播规约写入多播)
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + (h * 2 * nChannels + c) * buffSize;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + (h * 2 * nChannels + c) * buffSize;

      // 把 MC 数据广播到 UC(多播读出到单播)
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + ((h * 2 + 1) * nChannels + c) * buffSize;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + ((h * 2 + 1) * nChannels + c) * buffSize;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
    }
  }

  NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false),
                res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                        /*concurrent=*/false),
                res, fail);
  // 目前这个屏障是必须的：它保证在访问对端缓冲区之前，所有缓冲区都已完成 mc 映射
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                res, fail);
  comm->nvlsResources->inited = true;

exit:
  return res;
fail:
  comm->nvlsResources->inited = false;
  goto exit;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t res = ncclSuccess;
  size_t typeSize;
  char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];
  uintptr_t* nvlsShmem = NULL;
  bool nvlsShare = parent && parent->nvlsSupport && parent->shareResources && parent->localRanks == comm->localRanks;

  if (comm->nvlsSupport == 0 || comm->nvlsChannels == 0) return ncclSuccess;

  if (nvlsShare) {
    /* reuse NVLS resources */
    comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    /* Inherit chunk sizes from the shared resource since we're reusing the parent's
     * NVLS buffers, which were allocated and laid out based on these values. */
    comm->nvlsChunkSize = parent->nvlsResources->chunkSize;
    comm->nvlsTreeMaxChunkSize = parent->nvlsResources->treeMaxChunkSize;
    for (int c = 0; c < comm->nvlsChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, parent, true), res, fail);
    }

    comm->nvlsResources = parent->nvlsResources;
    ncclAtomicRefCountIncrement(&parent->nvlsResources->refCount);
  } else {
    struct ncclNvlsSharedRes* resources = NULL;
    int nHeads = comm->channels[0].nvls.nHeads;
    size_t memSize = 64;
    cudaStream_t hostStream, deviceStream;

    if (parent != nullptr && parent->nvlsSupport && parent->shareResources) {
      /* ranks on other nodes might share the NVLS resources, we need to cap nvlsChannels
       * and match NVLS chunk sizes to make sure they agree for each rank. */
      comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
      comm->nvlsChunkSize = parent->nvlsResources->chunkSize;
      comm->nvlsTreeMaxChunkSize = parent->nvlsResources->treeMaxChunkSize;
    }

    int nChannels = comm->nvlsChannels;
    size_t creditSize = nChannels * 2 * memSize * nHeads;
    int nvlsStepSize = comm->nvlsChunkSize;

    NCCLCHECKGOTO(ncclCalloc(&comm->nvlsResources, 1), res, fail);
    comm->nvlsResources->inited = false;
    comm->nvlsResources->refCount = 1;
    comm->nvlsResources->nChannels = nChannels;
    comm->nvlsResources->nHeads = nHeads;
    comm->nvlsResources->chunkSize = comm->nvlsChunkSize;
    comm->nvlsResources->treeMaxChunkSize = comm->nvlsTreeMaxChunkSize;
    resources = comm->nvlsResources;

    for (int c = 0; c < nChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, NULL, false), res, fail);
    }

    memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
    resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    resources->accessDesc.location.id = comm->cudaDev;
    resources->dev = comm->cudaDev;

    NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, creditSize, &resources->ucCreditHandle,
                                  &resources->mcCreditHandle, (void**)&resources->ucCredit,
                                  (void**)&resources->mcCredit, &resources->creditUCSize, &resources->creditMCSize),
                  res, fail);

    // 目前只设置 头 与 尾
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                          /*concurrent=*/false, &hostStream),
                  res, fail);
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
                                          &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream),
                  res, fail);
    for (int h = 0; h < nHeads; h++) {
      int nvlsPeer = comm->nRanks + 1 + h;
      for (int c = 0; c < nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        char* mem = NULL;
        struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

        // 把 UC 数据规约到 MC(单播规约写入多播)
        mem = resources->ucCredit + (h * 2 * nChannels + c) * memSize;
        peer->send[1].transportComm = &nvlsTransport.send;
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[1].conn.head = (uint64_t*)mem;
        peer->send[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[1].conn.stepSize = nvlsStepSize;
        mem = resources->mcCredit + (h * 2 * nChannels + c) * memSize;
        peer->recv[0].transportComm = &nvlsTransport.recv;
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[0].conn.head = (uint64_t*)mem;
        peer->recv[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[0].conn.stepSize = nvlsStepSize;
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        // 把 MC 数据广播到 UC(多播读出到单播)
        mem = resources->ucCredit + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->recv[1].transportComm = &nvlsTransport.recv;
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[1].conn.head = (uint64_t*)mem;
        peer->recv[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[1].conn.stepSize = nvlsStepSize;
        mem = resources->mcCredit + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->send[0].transportComm = &nvlsTransport.send;
        peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[0].conn.head = (uint64_t*)mem;
        peer->send[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[0].conn.stepSize = nvlsStepSize;
        peer->send[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
      }
    }
    NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                          /*concurrent=*/false),
                  res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
                                          &comm->sharedRes->deviceStream, /*concurrent=*/false),
                  res, fail);
  }

  // MNNVL 不支持 NVLS 缓冲区注册
  if (!comm->MNNVL && comm->nvlsResources->nvlsShmemHandle == NULL) {
    /* create shared memory for fast NVLS buffer registration */
    typeSize = DIVUP(sizeof(struct localRegData) << 1, CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

    if (comm->localRank == 0) {
      shmPath[0] = '\0';
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath),
                                (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2,
                                (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsResources->nvlsShmemHandle),
                    res, fail);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank,
                                                comm->localRanks, 0, shmPath, sizeof(shmPath)),
                    res, fail);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank,
                                                comm->localRanks, 0, shmPath, sizeof(shmPath)),
                    res, fail);
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath),
                                (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2,
                                (void**)&nvlsShmem, NULL, -1, &comm->nvlsResources->nvlsShmemHandle),
                    res, fail);
    }
    /* need 2 pools and a shared counter for shmem-based collectives */
    comm->nvlsResources->nvlsShmem.cnt[0] = (size_t*)nvlsShmem;
    comm->nvlsResources->nvlsShmem.ptr[0] =
      (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[0] + CACHE_LINE_SIZE * comm->localRanks);
    comm->nvlsResources->nvlsShmem.cnt[1] =
      (size_t*)((char*)comm->nvlsResources->nvlsShmem.ptr[0] + typeSize * comm->localRanks);
    comm->nvlsResources->nvlsShmem.ptr[1] =
      (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[1] + CACHE_LINE_SIZE * comm->localRanks);
    comm->nvlsResources->nvlsShmem.round = 0;
    comm->nvlsResources->nvlsShmem.maxTypeSize = typeSize;
  }

exit:
  return res;
fail:
  comm->nvlsSupport = 0;
  goto exit;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;
  if (resources == NULL) return ncclSuccess;

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {
    if (!comm->MNNVL && resources->nvlsShmemHandle) NCCLCHECK(ncclShmClose(resources->nvlsShmemHandle));

    if (resources->ucCredit || resources->mcCredit) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->creditUCSize, &resources->mcCreditHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->creditUCSize, resources->ucCredit, &resources->ucCreditHandle,
                                  resources->creditMCSize, resources->mcCredit, &resources->mcCreditHandle));
    }

    if (comm->nvlsResources->inited) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->buffUCSize, &resources->mcBuffHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->buffUCSize, resources->ucBuff, &resources->ucBuffHandle,
                                  resources->buffMCSize, resources->mcBuff, &resources->mcBuffHandle));
    }
    free(resources);
    comm->nvlsResources = NULL;
  }
  return ncclSuccess;
}

ncclResult_t tryRegisterBuffer(struct ncclComm* comm, uintptr_t userBuff, size_t buffSize, CUdeviceptr* regAddr,
                               int* regUsed) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg* regRecord = NULL;
  CUdeviceptr regPtr = 0;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle mcHandle = 0;
  size_t minSize = SIZE_MAX;
  struct localRegData* regData = NULL;
  cudaPointerAttributes attr;
  size_t ucgran, mcgran, ucsize = 0, mcsize = 0;
  bool bindComplete = false, mapComplete = false;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);

  if (userBuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, (void*)userBuff, buffSize, &regRecord), ret, fail);
    if (regRecord) {
      CUDACHECKGOTO(cudaPointerGetAttributes(&attr, (void*)regRecord->begAddr), ret, fail);
      if (attr.type == cudaMemoryTypeDevice) {
        size_t regSize = regRecord->endAddr - regRecord->begAddr;
        memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
        mcprop.numDevices = comm->localRanks;
        mcprop.handleTypes = ncclCuMemHandleType;
        mcprop.flags = 0;
        mcprop.size = regSize;
        CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);

        memset(&ucprop, 0, sizeof(CUmemAllocationProp));
        ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ucprop.location.id = comm->cudaDev;
        ucprop.requestedHandleTypes = ncclCuMemHandleType;
        CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);

        if (regRecord->begAddr % ucgran == 0) {
          if (regSize % ucgran != 0) {
            regRecord->regUCSize = ALIGN_SIZE(regSize, ucgran);
          } else {
            regRecord->regUCSize = regSize;
          }
          regRecord->state |= NVLS_REG_POSSIBLE;
          memcpy(&regData[comm->localRank].reg, regRecord, sizeof(struct ncclReg));
          regData[comm->localRank].offset = userBuff - regRecord->begAddr;
        }
      }

      if ((regRecord->state & NVLS_REG_POSSIBLE) == 0) {
        regRecord->state |= NVLS_REG_NO_SUPPORT;
      }
    }
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank, regData,
                                   sizeof(struct localRegData)),
                ret, fail);

  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i].reg.state & NVLS_REG_POSSIBLE) == 0) {
      goto fail;
    }
    // 我们需要检查各 rank 之间的偏移是否一致。
    if (i > 0 && regData[i].offset != regData[i - 1].offset) {
      goto fail;
    }
    /* get minimal reg size of nvls buffers */
    if (minSize > regData[i].reg.regUCSize) minSize = regData[i].reg.regUCSize;
  }

  /* start registration */
  mcsize = ucsize = minSize;
  mcprop.size = minSize;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle),
                  ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);
  // 节点内屏障，用于缓解 中止 时 cuMulticastBindAddr 可能卡死的问题
  // 它同时保证：若 cuMulticastBindAddr 失败，清理代码不会与 UDS 代理 发生竞态
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail);
  // Coverity 抱怨 regRecord 可能为 NULL。实践中不会如此，因为我们此前已检查过
  // 所有本地 rank 的 (regData[i].reg.状态 & NVLS_REG_POSSIBLE)，那样会提前捕获并退出。
  // coverity[var_deref_op]
  CUresult err;
  err = CUPFN(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)regRecord->begAddr, ucsize, 0));
  if (err != CUDA_SUCCESS) {
    // 对于与 MC 不兼容的缓冲区，不要打印错误。
    if (err != CUDA_ERROR_INVALID_VALUE) {
      const char* errStr;
      CUCALL(cuGetErrorString(err, &errStr));
      INFO(NCCL_REG, "Failed to multicast-bind user buffer: CUDA error %d '%s'", err, errStr);
    }
    goto fail;
  }
  bindComplete = true;

  // 为 NVLS 创建一个虚拟地址(VA)
  CUCHECKGOTO(cuMemAddressReserve(&regPtr, mcsize, mcgran, 0U, 0), ret, fail);
  // 在本地映射该 VA
  CUCHECKGOTO(cuMemMap(regPtr, mcsize, 0, mcHandle, 0), ret, fail);
  mapComplete = true;
  CUCHECKGOTO(cuMemSetAccess(regPtr, mcsize, &comm->nvlsResources->accessDesc, 1), ret, fail);

  /* get all buffer addresses */
  regRecord->caddrs[comm->localRank] = regRecord->begAddr;
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regRecord->caddrs + comm->localRank,
                                   regRecord->caddrs, sizeof(uintptr_t)),
                ret, fail);

  regRecord->regAddr = regPtr;
  regRecord->regUCSize = ucsize;
  regRecord->regMCSize = mcsize;
  regRecord->dev = comm->nvlsResources->dev;
  regRecord->mcHandle = mcHandle;
  regRecord->state |= NVLS_REG_COMPLETE;

  *regAddr = (uintptr_t)regPtr + regData[comm->localRank].offset;
  *regUsed = 1;
exit:
  free(regData);
  return ret;
fail:
  if (regPtr) {
    if (mapComplete) CUCALL(cuMemUnmap(regPtr, mcsize));
    CUCALL(cuMemAddressFree(regPtr, mcsize));
  }
  if (mcHandle) {
    if (bindComplete) CUCALL(cuMulticastUnbind(mcHandle, comm->nvlsResources->dev, 0 /*mcOffset*/, ucsize));
    CUCALL(cuMemRelease(mcHandle));
  }
  *regUsed = 0;
  goto exit;
}

static ncclResult_t nvlsRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize,
                                       size_t recvbuffSize, struct ncclReg* sendRegRecord,
                                       struct ncclReg* recvRegRecord, int* outRegBufUsed, void** outRegBufSend,
                                       void** outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  int regBufUsed = 0;
  struct localRegData* regData = NULL;
  bool sendNeedReg = false, recvNeedReg = false;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks * 2), ret, fail);

  if (sendRegRecord) {
    memcpy(&regData[comm->localRank * 2].reg, sendRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2].offset = (uintptr_t)sendbuff - sendRegRecord->begAddr;
  }
  if (sendbuff) {
    CUCHECKGOTO(cuPointerGetAttribute((void*)&regData[comm->localRank * 2].handleTypes,
                                      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)sendbuff),
                ret, fail);
  }

  if (recvRegRecord) {
    memcpy(&regData[comm->localRank * 2 + 1].reg, recvRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2 + 1].offset = (uintptr_t)recvbuff - recvRegRecord->begAddr;
  }
  if (recvbuff) {
    CUCHECKGOTO(cuPointerGetAttribute((void*)&regData[comm->localRank * 2 + 1].handleTypes,
                                      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)recvbuff),
                ret, fail);
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank * 2, regData,
                                   sizeof(struct localRegData) * 2),
                ret, fail);

  /* first check whether all local ranks find their registered buffer */
  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i * 2].reg.state & NVLS_REG_COMPLETE) == 0 ||
        regData[comm->localRank * 2].reg.caddrs[i] != regData[i * 2].reg.begAddr) {
      sendNeedReg = true;
    }

    if ((regData[i * 2 + 1].reg.state & NVLS_REG_COMPLETE) == 0 ||
        regData[comm->localRank * 2 + 1].reg.caddrs[i] != regData[i * 2 + 1].reg.begAddr) {
      recvNeedReg = true;
    }

    if ((regData[i * 2].reg.state & NVLS_REG_NO_SUPPORT) || (regData[i * 2 + 1].reg.state & NVLS_REG_NO_SUPPORT)) {
      goto fail;
    }

    if ((sendbuff && (regData[i * 2].handleTypes & ncclCuMemHandleType) == 0) ||
        (recvbuff && (regData[i * 2 + 1].handleTypes & ncclCuMemHandleType) == 0)) {
      goto fail;
    }
  }

  if (sendNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i * 2].offset != regData[(i + 1) * 2].offset) {
        /* offset are different, we cannot apply user buffer registration */
        goto fail;
      }
    }

    /* reuse previous registered buffer if possible */
    if (!sendNeedReg)
      regSendPtr = (CUdeviceptr)((uintptr_t)sendRegRecord->regAddr + regData[comm->localRank * 2].offset);
  }

  if (recvNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i * 2 + 1].offset != regData[(i + 1) * 2 + 1].offset) {
        goto fail;
      }
    }

    if (!recvNeedReg)
      regRecvPtr = (CUdeviceptr)((uintptr_t)recvRegRecord->regAddr + regData[comm->localRank * 2 + 1].offset);
  }

  if ((!sendNeedReg || sendbuff == NULL) && (!recvNeedReg || recvbuff == NULL)) {
    regBufUsed = 1;
    INFO(NCCL_REG,
         "rank %d reuse registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff "
         "%p, reg recvbuff %p",
         comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);
    goto exit;
  }

  /* Start Registration. Not found registered buffers, then check whether both send and recv buffer locate
   * in register request cache. */
  if (sendNeedReg && sendbuff && sendbuffSize > 0) {
    tryRegisterBuffer(comm, (uintptr_t)sendbuff, sendbuffSize, &regSendPtr, &regBufUsed);
    if (regBufUsed == 0) goto fail;
  }

  if (recvNeedReg && recvbuff && recvbuffSize > 0) {
    tryRegisterBuffer(comm, (uintptr_t)recvbuff, recvbuffSize, &regRecvPtr, &regBufUsed);
    if (regBufUsed == 0) goto fail;
  }

  INFO(NCCL_REG,
       "rank %d successfully registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg "
       "sendbuff %p, reg recvbuff %p",
       comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);

exit:
  *outRegBufSend = (void*)regSendPtr;
  *outRegBufRecv = (void*)regRecvPtr;
  *outRegBufUsed = regBufUsed;
  free(regData);
  return ncclSuccess;
fail:
  regBufUsed = 0;
  INFO(NCCL_REG, "rank %d failed to NVLS register sendbuff %p sendbuffSize %ld recvbuff %p recvbuffSize %ld",
       comm->rank, sendbuff, sendbuffSize, recvbuff, recvbuffSize);
  goto exit;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;
  bool sendIsValid = false;
  bool recvIsValid = false;
  void* baseSend = NULL;
  void* baseRecv = NULL;
  size_t baseSendSize = 0;
  size_t baseRecvSize = 0;

  *outRegBufUsed = 0;
  if (sendbuff) {
    NCCLCHECK(ncclRegFind(comm, sendbuff, sendbuffSize, &sendRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(sendRegRecord, &sendIsValid));
    if (sendIsValid) {
      int numSegments = 0;
      NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)sendbuff, sendbuffSize, (CUdeviceptr*)&baseSend, &baseSendSize,
                                         &numSegments));
      if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    }
  } else {
    sendIsValid = true;
  }

  if (recvbuff) {
    NCCLCHECK(ncclRegFind(comm, recvbuff, recvbuffSize, &recvRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(recvRegRecord, &recvIsValid));
    if (recvIsValid) {
      int numSegments = 0;
      NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)recvbuff, recvbuffSize, (CUdeviceptr*)&baseRecv, &baseRecvSize,
                                         &numSegments));
      if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    }
  } else {
    recvIsValid = true;
  }

  if (sendIsValid && recvIsValid)
    NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                                 outRegBufUsed, outRegBufSend, outRegBufRecv));

exit:
  return ncclSuccess;
}

struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;
  struct ncclReg* reg;
  struct ncclComm* comm;
};

static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  struct ncclNvlsCleanupCallback* sendRecord = NULL;
  struct ncclNvlsCleanupCallback* recvRecord = NULL;
  void* baseSend = NULL;
  void* baseRecv = NULL;
  size_t baseSendSize = 0;
  size_t baseRecvSize = 0;
  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;

  *outRegBufUsed = 0;
  if (sendbuff) {
    int numSegments = 0;
    NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)sendbuff, sendbuffSize, (CUdeviceptr*)&baseSend, &baseSendSize,
                                       &numSegments));
    if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    NCCLCHECK(ncclCommGraphRegister(comm, baseSend, baseSendSize, (void**)&sendRegRecord));
  }

  if (recvbuff) {
    int numSegments = 0;
    NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)recvbuff, recvbuffSize, (CUdeviceptr*)&baseRecv, &baseRecvSize,
                                       &numSegments));
    if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    NCCLCHECK(ncclCommGraphRegister(comm, baseRecv, baseRecvSize, (void**)&recvRegRecord));
  }

  NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                               outRegBufUsed, outRegBufSend, outRegBufRecv));

  if (*outRegBufUsed) {
    if (sendRegRecord) {
      sendRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      sendRecord->base.fn = cleanupNvls;
      sendRecord->reg = sendRegRecord;
      sendRecord->comm = comm;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)sendRecord);
      *nCleanupQueueEltsAdded += 1;
    }

    if (recvRegRecord) {
      recvRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      recvRecord->base.fn = cleanupNvls;
      recvRecord->reg = recvRegRecord;
      recvRecord->comm = comm;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)recvRecord);
      *nCleanupQueueEltsAdded += 1;
    }
  } else {
    if (sendbuff) NCCLCHECK(ncclCommGraphDeregister(comm, sendRegRecord));
    if (recvbuff) NCCLCHECK(ncclCommGraphDeregister(comm, recvRegRecord));
  }

exit:
  return ncclSuccess;
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  int factor;
  ncclResult_t ret = ncclSuccess;
  if (comm->nNodes == 1) {
    if (info->func == ncclFuncReduceScatter) {
      factor = (comm->compCap >= 100 ? 6 : 5) * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllGather) {
      factor = 4 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllReduce) {
      if (comm->compCap >= 100) {
        factor = 8 * 8;
      } else {
        factor = 4 * 8;
      }
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else {
      goto fail;
    }
  } else {
    // 针对 Blackwell + NVLS 注册缓冲区的进一步微调
    if (info->func == ncclFuncReduceScatter) {
      factor = (comm->bandwidths[ncclFuncReduceScatter][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] > 400 ? 7 : 6) * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllGather) {
      factor = 6 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllReduce) {
      if (comm->compCap >= 100 && comm->minNetBw >= 96.0f) {
        factor = 10 * 8;
      } else if (comm->compCap >= 100) {
        factor = 7 * 8;
      } else {
        factor = 6 * 8;
      }
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else {
      goto fail;
    }
  }

exit:
  return ret;
fail:
  ret = ncclInvalidArgument;
  goto exit;
}

#else

/*
 * Pre CUDA 12.1 stubs
 */

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle* mcHandler, CUdeviceptr ptr,
                                 int dev, size_t ucsize, size_t mcsize) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricInit(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricMap(struct ncclComm* comm, size_t offset, size_t ucsize, void* ucaddr) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricFree(struct ncclComm* comm, size_t ucsize, void* ucaddr) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricFinalize(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  *recChannels = 0;
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
