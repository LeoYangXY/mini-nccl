/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/debug.h — 调试/日志级别与 INFO/WARN 声明
 * ----------------------------------------------------------------------------
 * 定义调试宏与日志级别：INFO/WARN/ERROR 输出、NCCL_DEBUG 等级控制、函数名/行号
 * 打印辅助。几乎所有源文件都包含本文件以使用日志。
 */

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <thread>
#include "compiler.h"

// Conform to pthread 并且 NVTX 标准
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern FILE* ncclDebugFile;

#ifdef NCCL_OS_LINUX
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...)
  __attribute__((format(printf, 5, 6)));
#elif defined(NCCL_OS_WINDOWS)
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...);
#else
/* Fallback so headers (e.g. alloc.h via checks.h) compile when OS is not set (e.g. unit tests with MPI). */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...);
#endif

#ifdef NCCL_OS_LINUX
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...) __attribute__((format(printf, 6, 7)));
#elif defined(NCCL_OS_WINDOWS)
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...);
#else
/* Fallback so headers (e.g. alloc.h via checks.h) compile when OS is not set (e.g. unit tests with MPI). */
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...);
#endif

// Let 代码 temporarily downgrade WARN into 信息
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define VERSION(...) ncclDebugLogInternal(NCCL_LOG_VERSION, NCCL_ALL, nullptr, nullptr, 0, __VA_ARGS__)
#define WARN(...) ncclDebugLogInternal(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)

#define NOWARN(EXPR, FLAGS) \
  do { \
    int oldNoWarn = ncclDebugNoWarn; \
    ncclDebugNoWarn = FLAGS; \
    (EXPR); \
    ncclDebugNoWarn = oldNoWarn; \
  } while (0)

#define INFO(FLAGS, ...) \
  do { \
    int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
    if ((level >= NCCL_LOG_INFO && ((unsigned long)(FLAGS) & ncclDebugMask)) || (level < 0)) \
      ncclDebugLogInternal(NCCL_LOG_INFO, (unsigned long)(FLAGS), nullptr, nullptr, 0, __VA_ARGS__); \
  } while (0)

#define INFO_LOC_FN(FLAGS, file, line, fn, fmt, ...) \
  INFO((FLAGS), "%s:%d (%s) " fmt, (file), (line), (fn), ##__VA_ARGS__)
#define INFO_LOC(FLAGS, fmt, ...) INFO_LOC_FN((FLAGS), __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define TRACE_CALL(...) \
  do { \
    int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
    if ((level >= NCCL_LOG_TRACE && (NCCL_CALL & ncclDebugMask)) || (level < 0)) { \
      ncclDebugLogInternal(NCCL_LOG_TRACE, NCCL_CALL, nullptr, __func__, __LINE__, __VA_ARGS__); \
    } \
  } while (0)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) \
  do { \
    int level = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire); \
    if ((level >= NCCL_LOG_TRACE && ((unsigned long)(FLAGS) & ncclDebugMask)) || (level < 0)) { \
      ncclDebugLogInternal(NCCL_LOG_TRACE, (unsigned long)(FLAGS), nullptr, __func__, __LINE__, __VA_ARGS__); \
    } \
  } while (0)
#define TRACE_LOC_FN(FLAGS, file, line, fn, fmt, ...) \
  TRACE((FLAGS), "%s:%d (%s) " fmt, (file), (line), (fn), ##__VA_ARGS__)
#define TRACE_LOC(FLAGS, fmt, ...) TRACE_LOC_FN((FLAGS), __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define TRACE(...)
#define TRACE_LOC_FN(FLAGS, file, line, fn, fmt, ...)
#define TRACE_LOC(FLAGS, fmt, ...)
#endif

void ncclSetThreadName(std::thread& thread, const char* fmt, ...);
#ifdef __cplusplus
extern "C" {
#endif
#ifdef ncclResetDebugInit
#undef ncclResetDebugInit
#endif
void ncclResetDebugInit();
#ifdef __cplusplus
}
#endif

#endif
