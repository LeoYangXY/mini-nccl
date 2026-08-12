/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/cpuset.h — CPU 亲和性集合(cpuset)工具
 * ----------------------------------------------------------------------------
 * 提供 CPU 集合的位操作封装（设置/清除/遍历），用于把 NCCL 线程绑定到指定 CPU
 * 核，优化 NUMA 亲和性，减少跨 NUMA 访存延迟。
 */

#ifndef NCCL_CPUSET_H_
#define NCCL_CPUSET_H_

#include "nccl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef NCCL_OS_LINUX
#include <sched.h>
#elif defined(NCCL_OS_WINDOWS)
#define CPU_SETSIZE 64 // Windows uses DWORD_PTR for affinity
#endif

// 将 local_cpus（例如 0003ff,f0003fff）转换为 ncclAffinity。
// The bitmask is divided into 块 of 32 位, 每一个 them represented by 8 hex number.
#define U32_LEN 32 // using uint32_t
#define CPU_SET_N_U32 (CPU_SETSIZE / U32_LEN)

static ncclResult_t ncclStrToCpuset(const char* maskStr, ncclAffinity* set) {
  uint32_t cpumasks[CPU_SET_N_U32] = {0};

  // transform the string into an 数组 of 32 位 masks, starting 带有 highest mask
  int m = CPU_SET_N_U32;
  char* str = strdup(maskStr);
  char* token = strtok(str, ",");
  while (token != NULL && m > 0) {
    cpumasks[--m] = strtoul(token, NULL, /*base = hex*/ 16);
    token = strtok(NULL, ",");
  }
  free(str);

  // 列表 所有 the CPU as part 的 CPU 设置, starting 带有 lowest mask (= 当前的 值 of m)
  ncclOsCpuZero(*set);
  for (int a = 0; (a + m) < CPU_SET_N_U32; a++) {
    // 每个 mask is U32_LEN CPU, 列表 them 所有 若 b这是 on
    for (int i = 0; i < U32_LEN; ++i) {
      if (cpumasks[a + m] & (1UL << i)) ncclOsCpuSet(*set, i + a * U32_LEN);
    }
  }
  return ncclSuccess;
}

static char* ncclCpusetToRangeStr(ncclAffinity* mask, char* str, size_t len) {
  int c = 0;
  int start = -1;
  // Iterate through 所有 possible CPU 位 plus one 额外的 position
  for (int cpu = 0; cpu <= CPU_SETSIZE; cpu++) {
    int isSet = (cpu == CPU_SETSIZE) ? 0 : ncclOsCpuIsSet(*mask, cpu);
    // 起始 of a new 范围
    if (isSet && start == -1) {
      start = cpu;
    }
    // 末尾 of a 范围, add comma 之间 范围
    if (!isSet && start != -1) {
      if (cpu - 1 == start) {
        c += snprintf(str + c, len - c, "%s%d", c ? "," : "", start);
      } else {
        c += snprintf(str + c, len - c, "%s%d-%d", c ? "," : "", start, cpu - 1);
      }
      if (c >= len - 1) break;
      start = -1;
    }
  }
  if (c == 0) str[0] = '\0';
  return str;
}

static ncclResult_t ncclStrListToCpuset(const char* userStr, ncclAffinity* mask) {
  // 重置 CPU 设置
  ncclOsCpuZero(*mask);
  const char delim[] = ",";
  char* str = strdup(userStr);
  char* token = strtok(str, delim);
  while (token != NULL) {
    uint64_t cpu = strtoull(token, NULL, 0);
    ncclOsCpuSet(*mask, cpu);
    token = strtok(NULL, delim);
  }
  free(str);
  return ncclSuccess;
}

static ncclResult_t ncclCpusetToStrList(ncclAffinity* mask, char* str, size_t len) {
  if (len == 0) return ncclSuccess;
  str[0] = '\0';
  int count = 0;
  for (uint64_t id = 0; id < CPU_SETSIZE; ++id) {
    if (ncclOsCpuIsSet(*mask, id)) {
      snprintf(str + strlen(str), len - strlen(str), "%s%lu", (count++ == 0) ? "" : ",", id);
    }
  }
  return ncclSuccess;
}

#endif
