/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/strongstream.h — CUDA stream 封装(strong stream)
 * ----------------------------------------------------------------------------
 * 定义“强 stream”：在普通 CUDA stream 之上增加捕获(capture)/事件(event)管理能力，
 * 支撑 CUDA graph 的录制与回放，被 kernel 启动与 graph 处理复用。
 */

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include "checks.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <mutex>

// ncclCudaContext: wraps a CUDA 上下文 with 每个-上下文 状态.
struct ncclCudaContext;

// 获取 a ncclCudaContext to track 当前ly 活跃的 CUDA 上下文.
ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out);
// Drop 参考.
void ncclCudaContextDrop(struct ncclCudaContext* cxt);

/* ncclCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaStream_t origin;
  cudaGraph_t graph;
  unsigned long long graphId;
  int graphUsageMode;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNone(int graphUsageMode) {
  struct ncclCudaGraph tmp;
#if CUDART_VERSION >= 11030
  tmp.origin = nullptr;
  tmp.graph = nullptr;
  tmp.graphId = ULLONG_MAX;
  tmp.graphUsageMode = graphUsageMode;
#endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
#if CUDART_VERSION >= 11030
  return graph.graphId != ULLONG_MAX;
#else
  return false;
#endif
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
#if CUDART_VERSION >= 11030
  return a.graphId == b.graphId;
#else
  return true;
#endif
}

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, cudaStream_t stream, int graphUsageMode);
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg);

/* ncclStrongStream: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * All updates to a strong stream must be enclosed by a Acquire/Release pair.
 *
 * Acquire retrieves a "work" stream (cudaStream_t) which may be used to add
 * work.
 *
 * Release publishes the work streams work into the strong stream. The Release
 * must be issued by the same thread that did the Acquire.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// 获取 the strong 流. Upon 返回 `*workStream` 将会 usable to add work.
// `并发的` indicates 若 其他 线程 可能为 使用 the strong 流.
ncclResult_t ncclStrongStreamAcquire(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent,
                                     cudaStream_t* workStream);

// 获取 workStream for an 已经 acquired strong 流.
// `并发的` indicates 若 其他 线程 可能为 使用 the strong 流.
ncclResult_t ncclStrongStreamAcquiredWorkStream(struct ncclCudaGraph graph, struct ncclStrongStream* ss,
                                                bool concurrent, cudaStream_t* workStream);

// 释放 的 strong 流.
// `并发的` indicates 若 其他 线程 可能为 使用 the strong 流.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent);
ncclResult_t ncclCudaGraphRecordEvent(struct ncclCudaGraph graph, cudaEvent_t event, cudaStream_t stream);

ncclResult_t ncclStreamWaitStream(cudaStream_t a, cudaStream_t b, cudaEvent_t scratchEvent);

// Like cudaStreamWaitEvent except `e` 必须为 strictly ahead of everything 入 `s`.
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph g, cudaStream_t s, cudaEvent_t e);

// Synchrnoization 执行 不 需要 the strong 流 to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStreamCapture; // internal to ncclStrongStream

struct ncclStrongStream {
  // The 流 to 使用 for non-captured work.
  cudaStream_t liveStream;
  void* liveAcquiredBy;
#if CUDART_VERSION >= 11030
  // 此 流 ever appeared 入 a 图 capture.
  bool everCaptured;
  std::mutex mutex;
  struct ncclStrongStreamCapture* captureHead;
  // The 事件 用于 establish order 之间 图 并且 流. 期间 获取
  // 此 事件 is waited on, 期间 释放 这是 recorded to.
  cudaEvent_t serialEvent;
#endif
};

struct ncclCudaContext {
  struct ncclCudaContext* next;
  CUcontext hcontext;
  int refCount;
  struct ncclStrongStream launchOrder;
};

#endif
