/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/nccl_device/core.cc — 设备端核心 API 实现
 * ----------------------------------------------------------------------------
 * 实现设备端(dBV2 API)核心功能：ncclTeamWorld 等 team/世界组接口，以及面向 kernel
 * 的设备资源管理。通过 nccl_device/impl/core__funcs.h 暴露给 kernel 调用。
 */

#include "core.h"
#include "comm.h"
#include "nccl_device/impl/core__funcs.h"

NCCL_API(ncclTeam_t, ncclTeamWorld, ncclComm_t comm);
ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
  ncclTeam_t ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API(ncclTeam_t, ncclTeamLsa, ncclComm_t comm);
ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
  // Ignoring 错误 自 若 it 失败 ncclDevrInitOnce will 尝试 again.
  // The 已返回 team 将会 junk 并且 下一个 "interesting" API 调用 那个
  // needs ncclDevrInitOnce will 报告 the 错误.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t ans;
  ans.nRanks = comm->devrState.lsaSize;
  ans.rank = comm->devrState.lsaSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API(ncclTeam_t, ncclTeamRail, ncclComm_t comm);
ncclTeam_t ncclTeamRail(ncclComm_t comm) {
  // Ignoring 错误 as 上方.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t ans;
  ans.nRanks = comm->nRanks / comm->devrState.lsaSize;
  ans.rank = comm->rank / comm->devrState.lsaSize;
  ans.stride = comm->devrState.lsaSize;
  return ans;
}

NCCL_API(int, ncclTeamRankToWorld, ncclComm_t comm, ncclTeam_t team, int rank);
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}

NCCL_API(int, ncclTeamRankToLsa, ncclComm_t comm, ncclTeam_t team, int rank);
int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) {
  // Ignoring 错误 as 上方.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return -1;

  return comm->devrState.lsaSelf + (rank - team.rank) * team.stride;
}
