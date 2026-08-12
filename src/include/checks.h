/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/checks.h — 断言与错误检查宏
 * ----------------------------------------------------------------------------
 * 定义 INFO/WARN/ERROR 日志宏，以及 ASSERT/CHECK 系列断言（运行时校验，失败抛
 * ncclInternalError）。NCCL 代码大量使用这些宏做参数与状态校验，是排查问题的
 * 主要日志来源。
 */

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

// 检查 CUDA RT 调用
#define CUDACHECK(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
      return ncclUnhandledCudaError; \
    } \
  } while (false)

#define CUDACHECKGOTO(cmd, RES, label) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
      RES = ncclUnhandledCudaError; \
      goto label; \
    } \
  } while (false)

// 报告 失败 但 clear 错误 并且 continue
#define CUDACHECKIGNORE(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      INFO_LOC(NCCL_ALL, "Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
    } \
  } while (false)

// 使用 内联 函数 to clear CUDA 错误 inside expressions
static inline cudaError_t cuda_clear(cudaError_t err) {
  if (err != cudaSuccess) (void)cudaGetLastError();
  return err;
}

// 检查 若 cudaSuccess & clear CUDA 错误
#define CUDASUCCESS(cmd) cuda_clear(cmd) == cudaSuccess
// Clear CUDA 错误, 返回 CUDA 返回 代码
#define CUDACLEARERROR(cmd) cuda_clear(cmd)

#include <errno.h>
// 检查 系统 调用
#define SYSCHECK(statement, name) \
  do { \
    int retval; \
    SYSCHECKSYNC((statement), name, retval); \
    if (retval == -1) { \
      WARN("Call to " name " failed: %s", strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (false)

#define SYSCHECKSYNC(statement, name, retval) \
  do { \
    retval = (statement); \
    if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
      INFO_LOC(NCCL_ALL, "Call to " name " returned %s, retrying", strerror(errno)); \
    } else { \
      break; \
    } \
  } while (true)

#define SYSCHECKGOTO(statement, name, RES, label) \
  do { \
    int retval; \
    SYSCHECKSYNC((statement), name, retval); \
    if (retval == -1) { \
      WARN("Call to " name " failed: %s", strerror(errno)); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

// Pthread 调用 don't 设置 errno 并且 never 返回 EINTR.
#define PTHREADCHECK(statement, name) \
  do { \
    int retval = (statement); \
    if (retval != 0) { \
      WARN("Call to " name " failed: %s", strerror(retval)); \
      return ncclSystemError; \
    } \
  } while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) \
  do { \
    int retval = (statement); \
    if (retval != 0) { \
      WARN("Call to " name " failed: %s", strerror(retval)); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#define NEQCHECK(statement, value) \
  do { \
    if ((statement) != value) { \
      /* Print the back trace*/ \
      INFO_LOC(NCCL_ALL, "-> %d (%s)", ncclSystemError, strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (0)

#define NEQCHECKGOTO(statement, value, RES, label) \
  do { \
    if ((statement) != value) { \
      /* Print the back trace*/ \
      RES = ncclSystemError; \
      INFO_LOC(NCCL_ALL, "-> %d (%s)", RES, strerror(errno)); \
      goto label; \
    } \
  } while (0)

#define EQCHECK(statement, value) \
  do { \
    if ((statement) == value) { \
      /* Print the back trace*/ \
      INFO_LOC(NCCL_ALL, "-> %d (%s)", ncclSystemError, strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (0)

#define EQCHECKGOTO(statement, value, RES, label) \
  do { \
    if ((statement) == value) { \
      /* Print the back trace*/ \
      RES = ncclSystemError; \
      INFO_LOC(NCCL_ALL, "-> %d (%s)", RES, strerror(errno)); \
      goto label; \
    } \
  } while (0)

// Propagate 错误 up
#define NCCLCHECK(call) \
  do { \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      /* Print the back trace*/ \
      if (ncclDebugNoWarn == 0) INFO_LOC(NCCL_ALL, "-> %d", RES); \
      return RES; \
    } \
  } while (0)

#define NCCLCHECKGOTO(call, RES, label) \
  do { \
    RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      /* Print the back trace*/ \
      if (ncclDebugNoWarn == 0) INFO_LOC(NCCL_ALL, "-> %d", RES); \
      goto label; \
    } \
  } while (0)

// 报告 失败 但 continue - useful for cleanup 路径 w这里 希望
// 尝试 所有 cleanup 步骤. Preserves 第一个 错误 入 RES.
#define NCCLCHECKIGNORE(call, RES) \
  do { \
    ncclResult_t TMPRES = call; \
    if (TMPRES != ncclSuccess && TMPRES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) INFO_LOC(NCCL_ALL, "-> %d", TMPRES); \
      if (RES == ncclSuccess) RES = TMPRES; \
    } \
  } while (0)

#define NCCLCHECKNOWARN(call, FLAGS) \
  do { \
    ncclResult_t RES; \
    NOWARN(RES = call, FLAGS); \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      return RES; \
    } \
  } while (0)

#define NCCLCHECKGOTONOWARN(call, RES, label, FLAGS) \
  do { \
    NOWARN(RES = call, FLAGS); \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      goto label; \
    } \
  } while (0)

#define NCCLWAIT(call, cond, abortFlagPtr) \
  do { \
    uint32_t* tmpAbortFlag = (abortFlagPtr); \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) INFO_LOC(NCCL_ALL, "-> %d", RES); \
      return ncclInternalError; \
    } \
    if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECK(*tmpAbortFlag, 0); \
  } while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) \
  do { \
    uint32_t* tmpAbortFlag = (abortFlagPtr); \
    RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) INFO_LOC(NCCL_ALL, "-> %d", RES); \
      goto label; \
    } \
    if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
  } while (!(cond))

#define NCCLCHECKTHREAD(a, args) \
  do { \
    if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
      INFO_LOC(NCCL_INIT, "-> %d [Async thread]", (args)->ret); \
      return args; \
    } \
  } while (0)

#define CUDACHECKTHREAD(a) \
  do { \
    cudaError_t err = (a); \
    if (err != cudaSuccess) { \
      INFO_LOC(NCCL_INIT, "-> %d [Async thread]", (int)(err)); \
      args->ret = ncclUnhandledCudaError; \
      return args; \
    } \
  } while (0)

// 通用 线程 creation 实现 with 错误 handling
#define STDTHREADCREATE_IMPL(var, func, error_action, ...) \
  do { \
    try { \
      (var) = std::thread(func, __VA_ARGS__); \
    } catch (const std::exception& e) { \
      WARN("Thread creation failed: %s", e.what()); \
      error_action; \
    } \
  } while (0)

#define STDTHREADCREATE(var, func, ...) STDTHREADCREATE_IMPL(var, func, return ncclSystemError, __VA_ARGS__)

#define STDTHREADCREATE_GOTO(var, func, RES, label, ...) \
  STDTHREADCREATE_IMPL( \
    var, func, \
    do { \
      RES = ncclSystemError; \
      goto label; \
    } while (0), \
    __VA_ARGS__)

#define NEW_NOTHROW(var, x) \
  do { \
    (var) = new (std::nothrow) x{}; \
    if (!(var)) { \
      WARN("Allocation failed"); \
      return ncclSystemError; \
    } \
  } while (0)

#define NEW_NOTHROW_GOTO(var, x, RES, label) \
  do { \
    (var) = new (std::nothrow) x{}; \
    if (!(var)) { \
      WARN("Allocation failed"); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#endif
