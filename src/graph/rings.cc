/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * rings.cc —— 生成 AllReduce 的“环形(ring)”通信拓扑
 * ----------------------------------------------------------------------------
 * 在 mini-nccl 链路中的位置：
 *   bootstrap(收集信息) -> topo.cc(构建拓扑图) -> search.cc(为每个通道选定
 *   ring 的 prev/next 邻居) -> 【本文件：把 prev/next 展开成线性 ring 顺序】
 *   -> enqueue.cc(启动 kernel)。
 *
 * ring 算法让数据沿环单向流动，nRanks-1 步即可让每个 rank 拿到完整规约结果，
 * 是 AllReduce 在节点内(NVLink/PCIe)最常用、可扩展到多节点的拓扑之一。
 * ============================================================================
 */

#include "core.h"

// 调试辅助：把一串整型值(如某个通道的 prev/next 或 ring 顺序)拼成一个字符串打印。
// 超出缓冲区时以 "..." 截断，避免溢出。
void dumpLine(int* values, int nranks, const char* prefix) {
  constexpr int line_length = 128;
  char line[line_length];
  int num_width = snprintf(nullptr, 0, "%d", nranks - 1);  // safe as per "man snprintf"
  int n = snprintf(line, line_length, "%s", prefix);
  for (int i = 0; i < nranks && n < line_length - 1; i++) {
    n += snprintf(line + n, line_length - n, " %*d", num_width, values[i]);
    // At this point n may be more than line_length-1, so don't use it
    // for indexing into "line".
  }
  if (n >= line_length) {
    // Sprintf wanted to write more than would fit in the buffer. Assume
    // line_length is at least 4 and replace the end with "..." to
    // indicate that it was truncated.
    snprintf(line + line_length - 4, 4, "...");
  }
  INFO(NCCL_INIT, "%s", line);
}

// 根据 prev/next 数组，为每个 ring 通道生成线性 rank 序列 rings[]。
//   nrings : 通道数
//   rings  : 输出，rings[r*nranks + i] 表示通道 r 上第 i 个位置的 rank
//   prev/next : 每个通道每个 rank 的前驱/后继 rank（由搜索阶段给出）
//   rank   : 本进程 rank（仅用于日志与合法性校验）
// 关键流程：
//   1) 从本 rank 出发，沿 next 走 nranks 步，把经过的 rank 依次写入 rings；
//   2) 校验能否回到起点（环必须闭合）；
//   3) 用 64-bit 位图(每 64 个 rank 一组)快速校验所有 rank 都被包含。
ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  ncclResult_t ret = ncclSuccess;
  uint64_t* rankFound;
  int rankFoundSize = DIVUP(nranks, 64);
  NCCLCHECK(ncclCalloc(&rankFound, rankFoundSize));

  for (int r = 0; r < nrings; r++) {
    char prefix[40];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i = 0; i < nranks; i++) {
      rankFound[current / 64] |= (1ULL << (current % 64));
      rings[r * nranks + i] = current;
      current = next[r * nranks + current];
    }
    snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
    if (rank == 0) dumpLine(rings + r * nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      ret = ncclInternalError;
      goto end;
    }
    // Check that all ranks are there
    for (int i = 0; i < nranks; i++) {
      uint64_t bits = rankFound[i / 64], mask = 1ULL << (i % 64);
      // Fast check 64 ranks at a time
      if (mask == 1 && bits == 0xffffffffffffffff) {
        i += 63;
        continue;
      }
      if ((bits & mask) == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        ret = ncclInternalError;
        goto end;
      }
    }
    memset(rankFound, 0, rankFoundSize * sizeof(uint64_t));
  }
end:
  free(rankFound);
  return ret;
}
