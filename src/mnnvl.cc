/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mnnvl.h"
#include "transport.h"
#include <cuda.h>
#include "cudawrap.h"

// Determine 若 MNNVL 支持 可用
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  // MNNVL requires cuMem to be 启用
  if (!ncclCuMemEnable()) return ncclSuccess;

  // MNNVL 也 requires FABRIC 句柄 支持
  int cudaDev;
  int flag = 0;
  CUdevice currentDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  // Ignore 错误 若 CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is 不 受支持的
  (void)CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (!flag) return ncclSuccess;
  // 检查 那个 所有 ranks have 已初始化 the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) return ncclSuccess;
  }

  // Determine our MNNVL 域/clique 并且 NVL 域 大小
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  comm->nvlDomainSize = 0;
  for (int i = 0; i < comm->nRanks; i++) {
    nvmlGpuFabricInfoV_t* fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    nvmlGpuFabricInfoV_t* fabricInfo2 = &comm->peerInfo[i].fabricInfo;
    // 检查 若 cluster UUID 并且 cliqueId match
    // A zero UUID means we don't have MNNVL fabric 信息 - disable MNNVL
    unsigned long uuid0 = 0;
    unsigned long uuid1 = 0;
    memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
    memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
    if ((uuid0 | uuid1) == 0) return ncclSuccess;
    // 检查 若 相同 NVL 域 (clusterUuid match)
    if (memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) {
      comm->nvlDomainSize++;
      // 也 检查 若 相同 clique (cliqueId match)
      if (fabricInfo1->cliqueId == fabricInfo2->cliqueId) {
        if (i == comm->rank) {
          comm->cliqueRank = comm->clique.size;
        }
        comm->clique.ranks[comm->clique.size++] = i;
      }
    }
  }

  // ncclCommSplit: clique.大小 可能为 1 当 nvlDomainSize > 1; 仍 enable MNNVL.
  if (comm->clique.size <= 1 && comm->nvlDomainSize <= 1) return ncclSuccess;

  // 检查 那个 FABRIC 句柄 可以 exported & imported by IMEX
  {
    void* ptr = NULL;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc cuDesc;
    CUresult err;

    // 分配 FABRIC 句柄 compatible 内存
    ncclResult_t ret =
      ncclCuMemAlloc(&ptr, &handle, CU_MEM_HANDLE_TYPE_FABRIC, CUDA_IPC_MIN, comm->memManager, ncclMemOffload);
    if (ret != ncclSuccess) {
      // 返回 an 错误 若 这是 a MNNVL capable 系统 但 FABRIC 句柄 are 不 受支持的
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX channel configuration "
           "(/dev/nvidia-caps-imex-channels). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    err = CUPFN(cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (err != CUDA_SUCCESS ||
        (err = CUPFN(cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC))) != CUDA_SUCCESS) {
      const char* errStr;
      (void)pfn_cuGetErrorString(err, &errStr);
      NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));
      // 返回 an 错误 若 这是 a MNNVL capable 系统 但 it's 不 working
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX configuration "
           "(nvidia-imex-ctl -N). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));

    // Force the CUMEM 句柄 类型 to be FABRIC for MNNVL
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    comm->MNNVL = 1;
    INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d nvlDomainSize %d", comm->MNNVL, comm->clique.id,
         comm->clique.size, comm->cliqueRank, comm->nvlDomainSize);
  }
  return ncclSuccess;
}
