/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/gin.h — [GIN 相关] GIN session 设备 API
 * ----------------------------------------------------------------------------
 * 声明 GIN(第三方 GPU 内部接口库)的 session 设备接口（ncclGinSession 等），用于
 * outbox 收发会话。GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_GIN_SESSION_H_
#include "core.h"
#include "gin/gin_device_common.h"

#if NCCL_CHECK_CUDACC
struct ncclGinCtx; // Definition in nccl_device/gin/gin_device_host_common.h
template <unsigned>
struct ncclGinCtx_M; // ...

struct ncclGinDescriptorSmem; // A type user allocates in __shared__ memory

// 用作 ncclGinSession::放置 的完成动作
struct ncclGin_None {};

// 强 VA 信号：一旦可见，意味着之前所有 放置 都已落定。
struct ncclGin_StrongVASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};
// 弱 VA 信号：仅保证本次捆绑的 放置 已落定。
struct ncclGin_WeakVASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};
// 已废弃：请改用 ncclGin_StrongVASignalInc 或 ncclGin_WeakVASignalInc。
struct ncclGin_VASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};

// 强 VA 加信号：一旦可见，意味着之前所有 放置 都已落定。
struct ncclGin_StrongVASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};
// 弱 VA 加信号：仅保证本次捆绑的 放置 已落定。
struct ncclGin_WeakVASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};
// 已废弃：请改用 ncclGin_StrongVASignalAdd 或 ncclGin_WeakVASignalAdd。
struct ncclGin_VASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};

// 强加信号：一旦可见，意味着之前所有 放置 都已落定。
struct ncclGin_StrongSignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};
// 弱加信号：仅保证本次捆绑的 放置 已落定。
struct ncclGin_WeakSignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};
// 已废弃：请改用 ncclGin_StrongSignalAdd 或 ncclGin_WeakSignalAdd。
struct ncclGin_SignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};

// 强信号：一旦可见，意味着之前所有 放置 都已落定。
// Inc 不能与其它信号算子混用，除非中间插入一次 reset()。
struct ncclGin_StrongSignalInc {
  ncclGinSignal_t signal;
};

// 弱信号：仅保证本次捆绑的 放置 已落定。
// Inc may 不 be mixed with 其他 信号 operators 在没有 ... 的情况下 an
// 中间的 reset()。
struct ncclGin_WeakSignalInc {
  ncclGinSignal_t signal;
};

// 已废弃：请显式使用 ncclGin_StrongSignalInc 或 ncclGin_WeakSignalInc。
struct ncclGin_SignalInc {
  ncclGinSignal_t signal;
};

// 支持延迟(Deferred)模式：
// 结构体 ncclGin_SignalSet { ncclGinSignal_t 信号; uint64_t 值; };

// 已废弃：请改用 ncclGin_WeakCounterInc。
struct ncclGin_CounterInc {
  ncclGinCounter_t counter;
};

// 弱计数器自增：仅保证本次捆绑的 放置 在本地完成。
struct ncclGin_WeakCounterInc {
  ncclGinCounter_t counter;
};

struct ncclGin_DescriptorSmem {
  ncclGinDescriptorSmem* descriptor;
};

// 段类型标签描述一个缓冲区的物理 cuMem 段组成。
struct ncclGin_SegmentDevice {};       // all segments are device-backed
struct ncclGin_SegmentMixed {}; // mix of HOST_NUMA and device-backed segments
struct ncclGin_SegmentHostNuma {};     // all segments are HOST_NUMA (CPU-backed)

template <unsigned backendMask>
struct ncclGin_BackendMask;

template <ncclNetDeviceType backend>
using ncclGin_BackendOne = ncclGin_BackendMask<(1u << (int)backend)>;

using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;

#endif

#if NCCL_CHECK_CUDACC
struct ncclGin_C {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  ncclGinResourceSharingMode resourceSharingMode;

  //////////////////////////////////////////////////////////////////////////////
  // 内部(内部实现)：
  void* _ginHandle;
  uint64_t* _signalShadows;
  unsigned backendMask;

  NCCL_DEVICE_INLINE ncclGin_C(ncclDevComm const& comm_, unsigned backendMask_, int contextIndex,
                               ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);
};

// 用 placement new 包装的辅助初始化函数
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_init(ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm,
                                                        int contextIndex);

// 带显式资源共享模式的辅助初始化函数
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_initWithResourceSharingMode(
  ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex,
  ncclGinResourceSharingMode resourceSharingMode);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal(
  ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
  uint64_t signalOpArg, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut_v2(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal_v2(
  ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
  uint64_t signalOpArg, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinFlush(ncclGin_C* net, ncclCoopAny coop, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(ncclGin_C* net, ncclGinCounter_t counter, int bits,
                                                                cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitCounter(ncclGin_C* net, ncclCoopAny coop, ncclGinCounter_t counter,
                                                            uint64_t least, int bits, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(ncclGin_C* net, ncclGinSignal_t signal, int bits,
                                                               cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitSignal(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                                           uint64_t least, int bits, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetCounter(ncclGin_C* net, ncclGinCounter_t counter);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetSignal(ncclGin_C* net, ncclGinSignal_t signal);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
  cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue_v2(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
  cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(ncclGin_C* net, ncclGinSignal_t signal);

template <unsigned backendMask>
struct ncclGin_BackendMask {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  // 本上下文在运行时选定的资源共享模式。
  ncclGinResourceSharingMode resourceSharingMode;

  // 把 GIN 上下文载入寄存器。每个上下文对每个对端有一对 QP(队列对)。
  NCCL_DEVICE_INLINE ncclGin_BackendMask(
    ncclDevComm const&, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                     Coop coop = ncclCoopThread{}, uint32_t optFlags = ncclGinOptFlagsDefault,
                                     DescriptorSmem descriptor = ncclGin_None{}) const;

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void wait(ncclGinRequest_t& outRequest, Coop coop = ncclCoopThread{},
                               DescriptorSmem descriptor = ncclGin_None{},
                               cuda::memory_order ord = cuda::memory_order_acquire) const;

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None,
            typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void get(ncclTeam, int peer, ncclWindow_t remoteWnd, size_t remoteOffset, ncclWindow_t localWnd,
                              size_t localOffset, size_t bytes, Coop coop = ncclCoopThread{},
                              DescriptorSmem descriptor = ncclGin_None{}, uint32_t optFlags = ncclGinOptFlagsDefault,
                              SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <
    // 放置 完成时在对端采取的动作。
    // 对强信号：保证本 放置 以及
    // 本上下文发往同一对端的所有先前 放置 都已落定。
    // 对弱信号：仅保证本次捆绑的 放置 已落定。
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|StrongVASignal[Inc|Add]|WeakVASignal[Inc|Add],
                                          // StrongSignal[Inc|Add]|WeakSignal[Inc|Add]}
    // 源数据被消费后本地采取的动作。
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|WeakCounterInc}
    // 参与本次 放置 的线程集合。必须是 Coop 的子集。
    typename Coop = ncclCoopThread,
    // 可选的共享内存描述符空间。取 ncclGin_{无|DescriptorSmem} 之一
    typename DescriptorSmem = ncclGin_None,
    // 当虚拟地址中包含
    // CPU 后端的段时，使用非 设备 的标签
    typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault,
    SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <
    typename T,
    // 放置 完成时在对端采取的动作。
    // For strong 信号: guarantees 此 放置 并且 所有 preceding puts on 此 上下文 to 相同 对等端 are settled.
    // 对弱信号：仅保证本次捆绑的 放置 已落定。
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|StrongVASignal[Inc|Add]|WeakVASignal[Inc|Add],
                                          // StrongSignal[Inc|Add]|WeakSignal[Inc|Add]}
    // 源数据被消费后本地采取的动作。
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|ncclGin_WeakCounterInc}
    // 参与本次 放置 的线程集合。必须是 Coop 的子集。
    typename Coop = ncclCoopThread,
    // 可选的共享内存描述符空间。取 ncclGin_{无|DescriptorSmem} 之一
    typename DescriptorSmem = ncclGin_None,
    // 取 ncclGin_{SegmentDevice|SegmentMixed|SegmentHostNuma} 之一；当 VA 中含
    // CPU 后端的段时，使用非 设备 的标签
    typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void put(ncclTeam, int peer, ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
                              RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{},
                              Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{},
                              cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                              cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                              uint32_t optFlags = ncclGinOptFlagsDefault,
                              SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <typename T, // requires sizeof(T) <= 8
    // 所有模板参数见 放置()。
            typename RemoteAction = ncclGin_None, typename Coop = ncclCoopThread,
            typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void putValue(ncclTeam, int peer, ncclWindow_t dstWnd, size_t dstOffset, T value,
                                   RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
                                   DescriptorSmem descriptor = ncclGin_None{},
                                   cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                   cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                   uint32_t optFlags = ncclGinOptFlagsDefault) const;

  template <typename T, // requires sizeof(T) <= 8
    // 所有模板参数见 放置()。
            typename RemoteAction = ncclGin_None, typename Coop = ncclCoopThread,
            typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void putValue(ncclTeam, int peer, ncclSymPtr<T> dst, T value,
                                   RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
                                   DescriptorSmem descriptor = ncclGin_None{},
                                   cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                   cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                   uint32_t optFlags = ncclGinOptFlagsDefault) const;

  template <typename RemoteAction, typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void signal(ncclTeam, int peer, RemoteAction remoteAction, Coop coop = ncclCoopThread(),
                                 DescriptorSmem descriptor = ncclGin_None{},
                                 cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                 cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                 uint32_t optFlags = ncclGinOptFlagsDefault) const;

  // 本协作组内任意线程的 放置 所使用的源缓冲区都将可安全复用。
  // 刷写 不保证数据已落定在远端内存中。
  template <typename Coop, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire,
                                DescriptorSmem descriptor = ncclGin_None{}) const;

  // 计数器与信号等待使用指定位宽的“滚动(rolling)”比较逻辑，
  // 使得无符号溢出也不会破坏“x < x+1”这一性质。
  //
  // bool rolling_less_equal(uint64_t a, uint64_t b, 整型 位) {
  //   uint64_t m = uint64_t(-1)>>(64-位);
  //   返回 ((b-a) & m) <= (m>>1);
  // }
  //
  // 所等待的条件是：给定值与内部值的滚动比较满足 rolling_less_equal，
  // 即内部值不小于给定值。
  //
  // 计数器最多只能用 56 位，尽管这少于 uint64_t 能承载的量，
  // 

  NCCL_DEVICE_INLINE uint64_t readCounter(ncclGinCounter_t counter, int bits = 56,
                                          cuda::memory_order ord = cuda::memory_order_acquire) const;

  template <typename Coop>
  NCCL_DEVICE_INLINE void waitCounter(Coop, ncclGinCounter_t counter, uint64_t least, int bits = 56,
                                      cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 每个信号都有一个专用的“影子(shadow)值”，用户可自由操纵它——
  // 唯一会改动 shadow 的调用是 increaseSignalShadow 与 resetSignal。
  // 
  NCCL_DEVICE_INLINE uint64_t* getSignalShadowPtr(ncclGinSignal_t signal) const;
  NCCL_DEVICE_INLINE void increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const;

  // 返回信号当前值，除低几位外全部清零。
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclGinSignal_t signal, int bits = 64,
                                         cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 返回指定窗口与偏移处 VA 信号的当前值，除低几位外全部清零。
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits = 64,
                                         cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 等待信号达到或超过给定值。
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclGinSignal_t signal, uint64_t least, int bits = 64,
                                     cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 等待指定窗口与偏移处的 VA 信号达到或超过给定值。
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least,
                                     int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 等待信号达到或超过 shadow 值。
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignalMeetShadow(Coop, ncclGinSignal_t signal, int bits = 64,
                                               cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 等待信号超过 shadow 至少 leastDelta(通常为 1)，并更新 shadow
  // 为最新值，返回时 之前 等于先前的 shadow 值，
  // delta 等于差值。
  template <typename Coop, typename Uint>
  NCCL_DEVICE_INLINE void waitSignalFollowShadow(Coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before,
                                                 Uint* delta, int bits = 64,
                                                 cuda::memory_order ord = cuda::memory_order_acquire) const;

  // 清零。不得与对计数器的并发修改产生竞态。
  NCCL_DEVICE_INLINE void resetCounter(ncclGinCounter_t counter) const;
  // 把信号与 shadow 清零。不得与对信号的并发修改产生竞态。
  NCCL_DEVICE_INLINE void resetSignal(ncclGinSignal_t signal) const;
  // 重置指定窗口与偏移处的 VA 信号。
  NCCL_DEVICE_INLINE void resetSignal(ncclWindow_t signalWindow, size_t signalOffset) const;

  //////////////////////////////////////////////////////////////////////////////
  // 内部(内部实现)：

  void* _ginHandle;
  uint64_t* _signalShadows;

  NCCL_DEVICE_INLINE ncclGinCtx_M<backendMask> _makeCtx() const;
};
#endif

#endif // _NCCL_DEVICE_GIN_SESSION_H_
