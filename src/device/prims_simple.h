/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "network/unpack/unpack.h"
#include <cassert>

/* ============================================================================
 * device/prims_simple.h —— Simple 协议的 GPU 传输原语（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 在 AllReduce 全链路中的定位：all_reduce.h 里的 runRing/runTree 通过“Primitives”
 * 这个模板类实际完成“把本地数据写到对端 GPU 的 buffer / 从对端读回 / 做原地 reduce”。
 * 本文件是 Protocol=Simple 时的 Primitives 特化实现（最常用、最直接的数据搬运方式）。
 *
 * 关键概念：
 *   - Fan : 收/发的邻居集合（ring 时 1 收 1 发；tree 时多收 1 发等）。
 *   - 角色标志(RoleInput/Output/WaitRecv/PostSend/...) : 描述本线程在线程块里扮演的
 *     角色（谁负责读输入、谁负责发、谁负责等收）。
 *   - step/stepSize : 把数据切成若干 step，按 step 与对端做带步号同步的收发。
 *   - directBuff / conn 等 : 通过 P2P(IPC) 拿到的对端显存指针，load/store 直接跨卡访问。
 *   - Direct=1 时走“直写”：数据直接写到对端 buffer，省一次中转。
 * 一句话：Simple 原语 = “用 load/store 在相邻 GPU 间直接搬数据 + 原地 reduce”。
 * ============================================================================
 */

enum primsMode {
  primsModeDefault = 0,
  primsModePatRs = 1,
  primsModePatAg = 2
};

// Simple 协议的 Primitives 特化类：封装 全规约 在 GPU 上“收-规约-发”的底层动作。
// T=数据类型, RedOp=规约算子, Fan=邻居集合, ProtoSimple=Simple 协议参数。
// 它的 directSend/directRecv/directCopy 等方法就是跨卡 加载/存储 的具体实现。
template <typename T, typename RedOp, typename Fan, int Direct, int SlicePerChunk, int StepPerSlice, int Unroll,
          int P2p, int MultimemSrcs, int MultimemDsts, bool isNetOffload>
class Primitives<T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>,
                 P2p, isNetOffload> {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;  // 最大接收源数/发送目标数
  static constexpr int Input = 0, Output = 1;                           // 输入/输出缓冲区的数组下标
  /* 角色(Role)标志位：这是理解 NCCL kernel 线程模型的关键。
   *
   * 一个线程块内的线程并非都干同样的活，而是被划分成不同角色分工协作：
   *   - RoleInput/RoleOutput : 负责持有输入/输出缓冲区指针
   *   - RoleWaitRecv         : 等待对端把数据放进 FIFO(消费者侧的“等数据到”)
   *   - RoleWaitSend         : 等待对端腾出 FIFO 空间(生产者侧的“等有位置写”)
   *   - RolePostRecv         : 数据消费完后，通知对端“这块空间可以复用了”(归还信用)
   *   - RolePostSend         : 数据写完后，通知对端“数据已就绪，可以来取”
   * 其余标志位是状态/模式标记：
   *   - Aborted              : 通信被中止(如出错或超时)
   *   - NetRegMode           : 网络注册内存模式
   *   - ConnFifoEnabled      : 启用了连接 FIFO
   *   - DirectWrite/DirectRead : 直连写/直连读(P2P 可直接访问对端显存，省去中转)
   *   - PatMode              : PAT 算法模式
   *   - NvlsMinPolling       : NVLS 场景下用 multimem 指令做最小值轮询
   *   - NetDeviceUnpack      : 需要设备端解包(网络设备卸载场景)
   * 少量线程做同步(Wait/Post)，大量线程做搬运，这样同步开销被摊薄，是高带宽的关键。
   */
  static constexpr int RoleInput = 0x01, RoleOutput = 0x02, RoleWaitRecv = 0x04, RoleWaitSend = 0x08,
                       RolePostSend = 0x10, RolePostRecv = 0x20, Aborted = 0x40, NetRegMode = 0x80,
                       ConnFifoEnabled = 0x100, DirectWrite = 0x200, DirectRead = 0x400, PatMode = 0x800,
                       NvlsMinPolling = 0x1000, NetDeviceUnpack = 0x2000, AnyNetDeviceUnpack = 0x4000;
  const int tid, tidInBlock;   // tid：本原语组内的线程号；tidInBlock：在整个线程块中的线程号
  const int nthreads;          // 本原语组的线程总数(含同步线程)
  int nworkers;                // 其中真正参与数据搬运的“工作线程”数量
  const int stepSize;          // 一个 step 对应的元素个数(FIFO 单个槽位的容量)
  Fan fan;                     // 扇形拓扑描述：记录有几个接收源、几个发送目标
  int index;                   // 本线程负责的对端序号(每个 Wait/Post 线程盯一个对端)
  int flags;                   // 本线程的角色与状态标志位(上面那些 Role* 的按位组合)
  int group;                   // 同步组编号：不同组使用不同的 barrier，互不干扰
  uint64_t step;               // 本连接当前推进到的步数(单调递增，用作 FIFO 的逻辑时钟)
  struct ncclConnInfo* conn = NULL;         // 连接信息(指针、FIFO 地址等)
  struct ncclConnFifo* connFifo = NULL;     // 连接的 FIFO 元数据数组
  T* connEltsFifo;                          // FIFO 的实际数据缓冲区
  T* directBuff = NULL;                     // 直连缓冲区：可直接读写的对端显存地址
  uint64_t* connStepPtr;                    // 指向对端步数计数器的指针(跨卡可见，用于同步)
  uint64_t connStepCache;                   // 缓存上次读到的 (*connStepPtr) 值，减少昂贵的远程读取
  int connStepSize;                         // 连接的步长(每步搬运多少元素)
  void* netDeviceHandle;                    // 网络设备句柄(设备端网络卸载用)
  uint64_t accSize;                         // 已累计处理的数据量

  // 组内同步屏障。注意不要使用 0 号 屏障：它被保留给 内核 结束时的最终同步。
  // 这里用 15-组 来给每个组分配独立的 屏障 编号，避免不同组互相干扰。
  __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15 - group;
      barrier_sync(bar, nthreads);
    }
  }
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = 15 - group - (nworkers != nthreads ? 1 : 0);
      barrier_sync(bar, nworkers);
    }
  }

  // PAT 算法在所有分组之间共用同一个屏障
  __device__ void patBarrier() {
    barrier_sync(15, NCCL_PAT_NWORKERS);
  }

  __device__ bool barrierAny(int vote) {
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15 - group;
      return barrier_red_or(vote, name, nthreads);
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15 - group - (nworkers != nthreads ? 1 : 0);
      return barrier_red_or(vote, name, nworkers);
    }
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    if (flags & NvlsMinPolling) {
      uint64_t ans;
      asm volatile("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];"
                   : "=l"(ans)
                   : "l"(cvta_to_global(ptr))
                   : "memory");
      return ans;
    }
#endif
    // 这里刻意使用 易变的 而非 获取 语义：易变的 更快，但内存序保证较弱。
    // 之所以可以这样做，是因为我们保证了 reduceCopy 也用 易变的 方式读取数据，
    // 从而绕过 L1 缓存、不会读到陈旧数据。这是用“弱内存序 + 全链路一致的访问方式”
    // 换取性能的典型手法：只要读写双方都不走 L1，就不会出现可见性问题。
    return ld_volatile_global(ptr);
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
    // coverity[dead_error_line]
    if ((flags & (Recv * RoleWaitRecv)) || (flags & (Send * RoleWaitSend))) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
        // 若 (spins == 0) {
        //   printf("r=%d b=%d t=%d SPUN 出 已获取=%d 想要=%d\n", ncclShmem.通信域.rank, blockIdx.x, threadIdx.x,
        //          整型(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), 整型(步骤+StepPerSlice));
        // }
      }
    }

    if (flags & (Recv * RoleWaitRecv | Send * RoleWaitSend)) {
      if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
        connFifo[step % NCCL_STEPS].size = nelts * sizeof(T);

      void** ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst) : (ncclShmem.groups[group].srcs + Src);
      if ((flags & NetRegMode) && ((!isSendNotRecv && DirectRecv) || (isSendNotRecv && DirectSend))) {
        if (P2p) {
          ptrs[index] = NULL;
        } else {
          if (isSendNotRecv) {
            if (!Recv) ptrs[index] = NULL;
            else ptrs[index] = (T*)ncclShmem.groups[group].userOutput + dstIx + offset;
          } else {
            ptrs[index] = (T*)ncclShmem.groups[group].userOutput + srcIx + offset;
          }
        }
      } else if ((flags & ConnFifoEnabled) && connFifo[step % NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
        ptrs[index] = connEltsFifo + loadInt(&connFifo[step % NCCL_STEPS].offset) / sizeof(T);
      } else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;
        } else if (flags & DirectRead) {
          // 空发送(无数据可发)
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
        }
      } else if (!isSendNotRecv && DirectRecv) {
        if (flags & DirectRead) {
          ptrs[index] = directBuff + srcIx + offset;
        } else if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
        } else {
          ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
        }
      } else {
        // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
        // coverity[dead_error_line]
        ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
      }
      if (flags & NetDeviceUnpack) {
        ncclNetDeviceIncrementHead(group, index);
      }
      step += StepPerSlice;
    }
  }

  template <int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv * RolePostRecv | Send * RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored || (flags & ConnFifoEnabled))) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize * StepPerSlice;
    sliceSize = max(divUp(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem && !isNetOffload) {
      /* 这个循环专供“工作线程 + 非空 slice”使用。非工作线程以及空 slice
       * 由紧随本 if 块之后的那个循环来处理。
       *
       * 为什么要把一个循环拆成两个？为了把两个分支判断移出关键路径。
       * 以“动态执行到的分支指令条数(无论是否跳转)”作为性能度量：
       *   原实现：perf_orig = 2 * numslices  (每个 slice 都要判断 2 次)
       *   新实现：perf_new  = 2 + numslices  (2 次判断只在循环外做一次)
       * 因此 numslices=2 时两者持平，numslices>2 时新写法更优。
       * 而 numslices=1 时循环可被平凡展开(只有一次迭代)，不会产生尾部分支，
       * 新写法依然是 perf_new=2，不吃亏。
       *
       * 原始代码形态如下(供对照理解)：
       *   unrolled for(slices) {
       *     if(worker) {                // 这个分支被提到了循环外
       *       wait();
       *       subBarrier();
       *       if(slice not empty)       // 这个分支也被消除了
       *         ReduceCopyMulti();
       *     }
       *     barrier();
       *     post();
       *   }                             // 由于不再展开，这里新增了一个循环分支
       */
#if __CUDA_ARCH__ < 700
      // 在 Volta 之前的老硬件上，上述分支优化收益不明显(分支预测与调度机制不同)，
      // 因此这里仍然让编译器完全展开循环，用代码膨胀换取更少的循环开销。
      NVCC_PRAGMA_UNROLL(SlicePerChunk)
#else
      // Volta 及以后：禁止展开，配合上面的循环拆分获得更少的动态分支数
      NVCC_PRAGMA_UNROLL_DISABLED
#endif
      do {
        sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf == Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf == Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (flags & AnyNetDeviceUnpack) {
          ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group,
                                    ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src,
                                    workSize);
          // 在此同步，确保所有工作线程读取到的都是更新后的 srcs 指针
          subBarrier();
        }

        if (DirectRecv &&
            ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            /* NVLS can have srcs[0] == dsts[0], but we cannot enter this "if branch",
             * so we need to check whether MultimemSrcs and MultimemDsts are 0. */
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // 直连接收最多只能有一个。由于 srcs[0] == dstPtr+偏移(源和目标是同一块地址)，可以省掉一次拷贝
          if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/ 0>(
              tid, nworkers, /*redArg*/ 0, /*postOp*/ false, 1, ncclShmem.groups[group].srcs, fan.nsend(),
              ncclShmem.groups[group].dsts + 1, workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // 用于 CollNet 广播场景下执行空发送
          reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/ 0>(tid, nworkers,
                                                                          ncclShmem.groups[group].redOpArgs, postOp,
                                                                          Recv, ncclShmem.groups[group].srcs, Dst,
                                                                          ncclShmem.groups[group].dsts, workSize);
        } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 : 1;
          if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
            // 这种情况只应出现在：使用已注册缓冲区的 directCopySend()，且发送目标是网络对端
            reduceCopy<Unroll, RedOp, T, 0, Recv + Src, Recv * MaxRecv + Src, 0, 1, 1, PreOpSrcs>(
              tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp, Recv * fan.nrecv() + Src,
              ncclShmem.groups[group].srcs, 1, ncclShmem.groups[group].dsts, workSize);
          } else {
            reduceCopy<Unroll, RedOp, T, MultimemSrcs, Recv + Src, Recv * MaxRecv + Src, MultimemDsts, Send + Dst,
                       Send * MaxSend + Dst, PreOpSrcs>(tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp,
                                                        Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                                                        Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
                                                        workSize);
          }
        } else {
          // 当对网络对端调用 prims.directSend 时会走到这里，
          // 此时 ncclShmem.组[组].dsts[0] 为 NULL，因此我们
          // 跳过数据刷新。
          workSize = 0;
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < workSize);
        offset += sliceSize;
        slice += 1;
        // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
        // coverity[dead_error_line]
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // 非工作线程会直接执行到这里。工作线程也会来，但仅当剩余的
    // slice 全部为空时才会到达。由于空 slice 属于少见情况，而且
    // 性能瓶颈在工作线程一侧，因此从性能角度看这个循环基本等同于不会进入，
    // 因此只需一条分支指令。
    NVCC_PRAGMA_UNROLL_DISABLED
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // 因为我们已经从上面的循环退出了。
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, sliceSize);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      int workSize = ncclShmem.aborted ? 0 : sliceSize;
      postPeer<Recv, Send>(0 < workSize);
      offset += sliceSize;
      slice += 1;
    }
  }

public:
  static inline __device__ void sendPeerNotify(int peer, int connIndex, int steps) {
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->send[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->send[connIndex].tail, peerPtr->send[connIndex].step);
  }

  static inline __device__ void recvPeerNotify(int peer, int connIndex, int steps) {
    int spins = 0;
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->recv[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->recv[connIndex].head, peerPtr->recv[connIndex].step);
    while (ld_volatile_global(peerPtr->recv[connIndex].tail) < peerPtr->recv[connIndex].step) {
      int abort = 0;
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  template <int Recv, int Send, typename Fn>
  __device__ __forceinline__ void process(Fn&& fn, uint32_t sendDirectFlag = 0, uint32_t recvDirectFlag = 0) {
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int slice = 0; slice < SlicePerChunk; slice++) {
      if (tid < nworkers) {
        int nsend, nrecv;
        if (flags & (Recv * RoleWaitRecv | Send * RoleWaitSend)) {
          const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
          int spins = 0;
          while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
            connStepCache = loadStepValue(connStepPtr);
            if (checkAbort(flags, Aborted, spins)) break;
          }
          void** ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts : ncclShmem.groups[group].srcs;
          if ((flags & ConnFifoEnabled) && connFifo[step % NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
            int offset = loadInt(&connFifo[step % NCCL_STEPS].offset);
            ptrs[index] = connEltsFifo + offset / sizeof(T);
          } else if (Direct && fn.work->regUsed) {
            if (isSendNotRecv) {
              if (flags & DirectWrite) {
                ptrs[index] = directBuff;
              } else if (flags & DirectRead) {
                // 空发送(无数据可发)
                ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
              }
            } else {
              if (flags & DirectRead) {
                ptrs[index] = directBuff;
              } else if (flags & DirectWrite) {
                if (Send) ptrs[index] = directBuff;  // send to next from my output buffer
                else ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
              }
            }
          } else {
            ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
          }
        }
        subBarrier();
        if (Recv == 0 || ncclShmem.groups[group].srcs[0] == nullptr) {
          nrecv = 0;
        } else {
          nrecv = fan.nrecv();
        }

        if (Send == 0 || ncclShmem.groups[group].dsts[0] == nullptr) {
          nsend = 0;
        } else {
          nsend = fan.nsend();
        }
        fn.template operator()<SlicePerChunk, 0, Recv * MaxRecv, 0, Send * MaxSend, MultimemSrcs, MultimemDsts>(
          tid, nworkers, slice, stepSize * StepPerSlice, nrecv, ncclShmem.groups[group].srcs, nsend,
          ncclShmem.groups[group].dsts, ncclShmem.groups[group].dstSizes, sendDirectFlag, recvDirectFlag);
      }
      barrier();
      int32_t dstSize = 0;
      if (flags & Send * RolePostSend) {
        // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
        // coverity[dead_error_begin]
        dstSize = ncclShmem.groups[group].dstSizes[index];
        ncclShmem.groups[group].dstSizes[index] = 0;
        if (flags & ConnFifoEnabled) connFifo[step % NCCL_STEPS].size = dstSize * sizeof(T);
      }
      barrier();
      if (flags & (Recv * (RoleWaitRecv | RolePostRecv) | Send * (RoleWaitSend | RolePostSend))) {
        step += StepPerSlice;
      }
      if (flags & (Recv * RolePostRecv | Send * RolePostSend)) {
        if (Send && (!Recv || (flags & RolePostSend)) && (dstSize != 0 || (flags & ConnFifoEnabled))) {
          fence_acq_rel_sys();
        }
        st_relaxed_sys_global(connStepPtr, step);
      }
    }
  }

private:
  // 散播/收集 通用操作
  // skip：本 rank 在缓冲区分块中的次序
  // shift：对端偏移量，用于避免所有 rank 同时向同一个对端收发(打散热点)
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void ScatterGatherOp(intptr_t inpIx, intptr_t outIx, ssize_t totalElem, int peerElem,
                                                  ssize_t peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize * StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16 * SlicePerChunk) * 16, sliceSize / 32);  // per-peer slice size

    NVCC_PRAGMA_UNROLL_AUTO
    for (int slice = 0; slice < SlicePerChunk; ++slice) {
      ssize_t realSize = max(0, min(dataSize, peerElem - offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // 仅在非直连(non-Direct)场景下，散播 才会对输入缓冲区的数据做预缩放
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (tid == 0) ncclShmem.groups[group].srcs[0] = (T*)ncclShmem.groups[group].userInput + inpIx + offset;
          // 此处的 realSize 并不精确；但节点内通信不依赖 sizes FIFO，因此无妨
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          NVCC_PRAGMA_UNROLL_AUTO
          // 遍历所有对端
          for (int j = 0; j < fan.nsend(); j++) {
            int i = (j + shift) % fan.nsend();
            ssize_t pOffset = i * peerOffset;
            // 跳过由我自己负责规约的那部分数据
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem - pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, PreOpSrcs>(tid, nworkers,
                                                                        ncclShmem.groups[group].redOpArgs, false, 1,
                                                                        &src0, 1, ncclShmem.groups[group].dsts + i,
                                                                        realPeerSize);
              // 打标记：在结尾处需要执行 threadfence(内存栅栏)
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (tid == 0) ncclShmem.groups[group].dsts[0] = (T*)ncclShmem.groups[group].userOutput + outIx + offset;
          ssize_t pOffset = index * peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerOffset;
          // 用对端偏移量修正远程下标，以应对“直接从对端输出缓冲区拉取数据”的情形
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx + pOffset, outIx + pOffset, offset, realSize);
          subBarrier();
          NVCC_PRAGMA_UNROLL_AUTO
          for (int j = 0; j < fan.nrecv(); j++) {
            int i = (j + shift) % fan.nrecv();
            pOffset = i * peerOffset;
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem - pOffset);
            if (DirectRecv && ncclShmem.groups[group].srcs[i] == dst0) realPeerSize = 0;
            if (realPeerSize > 0)
              reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>(tid, nworkers,
                                                                              ncclShmem.groups[group].redOpArgs, postOp,
                                                                              1, ncclShmem.groups[group].srcs + i, 1,
                                                                              &dst0, realPeerSize);
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer* peer, int connIndex, uint32_t direct, int ipcRegFlag,
                                               int netRegFlag) {
    conn = &peer->recv[connIndex];
    if (conn->netDeviceHandle.netDeviceType == NCCL_NET_DEVICE_UNPACK) {
      // 句柄 必须是设备指针
      netDeviceHandle = conn->netDeviceHandle.handle;
      // 缓存该句柄
      ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
      flags |= NetDeviceUnpack;
    }
    step = conn->step;
    step = roundUp(step, SlicePerChunk * StepPerSlice);
    if (flags & RolePostRecv) {
      connStepPtr = conn->head;
      *connStepPtr = step; // Return credits in case we rounded up.
    }
    if (flags & RoleWaitRecv) {
      // 由 WaitRecv 角色的线程保存，因为在 setDataPtrs() 中正是它需要用到
      if ((flags & PatMode) == 0) ncclShmem.groups[group].recvConns[index] = conn;
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->tail;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize / sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (conn->connFifo != nullptr) {
        flags |= ConnFifoEnabled;
        connFifo = conn->connFifo;
      }
      if (Direct) {
        if (ipcRegFlag) {
          // 用户缓冲区已经注册
          if (conn->flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectRead;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
            connFifo[step % NCCL_STEPS].size = 0;
          }
        }
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer* peer, int connIndex, uint32_t direct, int ipcRegFlag,
                                               int netRegFlag) {
    conn = &peer->send[connIndex];
    step = conn->step;
    step = roundUp(step, SlicePerChunk * StepPerSlice);

    connFifo = conn->connFifo;
    if (connFifo != nullptr) flags |= ConnFifoEnabled;

    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    }
    if (flags & RoleWaitSend) {
      // 由 WaitSend 角色的线程保存，因为在 setDataPtrs() 中正是它需要用到
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn;
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize / sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (Direct) {
        if (ipcRegFlag) {
          // 用户缓冲区已经注册
          if (conn->flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectWrite;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
          }
        }
      }
    }
  }

public:
  __device__ Primitives(int tid, int nthreads, int const* recvPeers, int const* sendPeers, void const* inputBuf,
                        void* outputBuf, uint64_t redOpArg, uint8_t group = 0, uint8_t connIndexRecv = 0,
                        uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
                        struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, int mode = primsModeDefault)
    : tid(tid), nthreads(nthreads), tidInBlock(threadIdx.x), group(group),
      stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T) : stepSize_) {
    int peer = -1;
    flags = 0;
    index = -1;
    if (mode == primsModeDefault) {
      // 与 sendPeers/recvPeers 中的各个 rank 建立连接
      // 对于发送操作，需要额外一个 线程束，以便让 threadfence 与数据拷贝相互重叠
      this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

      int nrecv = 0, nsend = 0;
      // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
      // coverity[dead_error_line]
      while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
      // coverity[dead_error_line]
      while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        // NVLS 的扇出可能超过 8。这种情况下需要增大分组的规模
        MaxSend >= 16 || MaxRecv >= 16 ?
          32 :
        MaxSend >= 8 || MaxRecv >= 8 ?
          16 :
          8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
      static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

      assert(2 * (nrecv + nsend) <= nthreads); // Ensure no thread is assigned more than one role.
      // Coverity 会根据下面这行代码推断 索引 等于 tid，但它没有考虑到
      // 标志 的设置逻辑。这导致它在此处以及 all_reduce.h 中报出多处越界误报。
      // 遗憾的是，我们未能用一条指令统一屏蔽这些误报，因此
      // 这项工作改由调用方完成。
      // coverity[assignment:假]
      if (tid < nrecv) {
        flags |= RoleWaitRecv;
        index = tid;
      }
      // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
      // coverity[dead_error_begin]
      else if (tid < nrecv + nsend) {
        flags |= RoleWaitSend;
        index = tid - nrecv;
      } else if (nthreads - nsend <= tid) {
        flags |= RolePostSend;
        index = tid - (nthreads - nsend);
      } else if (nthreads - nrecv - nsend <= tid) {
        flags |= RolePostRecv;
        index = tid - (nthreads - nrecv - nsend);
      }

      if (flags & (RoleWaitRecv | RolePostRecv)) peer = recvPeers[index];
      if (flags & (RoleWaitSend | RolePostSend)) peer = sendPeers[index];

      // Coverity thinks 那个 索引 可能为 -1 here 但 那个's 不 actually the 情形.
      // coverity[negative_returns:假]
      int sendIpcReg;
      int recvIpcReg;
      int sendNetReg;
      int recvNetReg;
      if (P2p) {
        sendIpcReg = p2pWork ? p2pWork->sendIpcReg : 0;
        recvIpcReg = p2pWork ? p2pWork->recvIpcReg : 0;
        sendNetReg = p2pWork ? p2pWork->sendNetReg : 0;
        recvNetReg = p2pWork ? p2pWork->recvNetReg : 0;
      } else {
        recvIpcReg = sendIpcReg = collWork ? collWork->regUsed : 0;
        recvNetReg = sendNetReg = collWork ? collWork->netRegUsed : 0;
      }

      // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
      if (flags & (RoleWaitRecv | RolePostRecv))
        loadRecvConn(ncclShmem.channel.peers[peer], connIndexRecv, collWork ? collWork->direct : 0, recvIpcReg,
                     recvNetReg);
      // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
      if (flags & (RoleWaitSend | RolePostSend))
        loadSendConn(ncclShmem.channel.peers[peer], connIndexSend, collWork ? collWork->direct : 0, sendIpcReg,
                     sendNetReg);

      // coverity[negative_returns:假] => coverity thinks 那个 索引 可能为 -1 但 那个's 不 actually the 情形
      // coverity[var_deref_model] => coverity thinks work can dereferenced 若 NULL 但 这是 不 the 情形
      setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclDevWorkCollReg*)collWork, sendIpcReg || recvIpcReg, peer);
      // coverity[uninit_member] => coverity thinks fan.n is 不 已初始化

      if (barrierAny(flags & NetDeviceUnpack)) {
        flags |= AnyNetDeviceUnpack;
        // RoleWaitRecv 从 tid=0 开始，因此这里构建出“哪些接收对端”的位掩码，
        // 它们启用了 NetDeviceUnpack。
        uint32_t mask = __ballot_sync(~0u, ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
        if (tid == 0) {
          ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
        }
      }
    } else if (mode == primsModePatRs || mode == primsModePatAg) {
      // 连接到所有距离为 ±2^n 的 rank(指数阶梯，分散链路)
      flags |= PatMode;
      const int roles[5] = {RoleWaitRecv, RolePostRecv, RoleWaitSend, RolePostSend, RoleInput | RoleOutput};
      if (tid < 5) flags |= roles[tid];

      int nranks = ncclShmem.comm.nRanks;
      if (tid < 32 && ((1UL << tid) < nranks)) {
        int rank = ncclShmem.comm.rank;
        uint32_t delta = 1 << tid;
        // 加载接收对端信息
        int recvPeer = mode == primsModePatRs ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        struct ncclPatPeer* peer = ((struct ncclPatPeer*)recvPeers) + tid;
        struct ncclConnInfo* conn = peer->conn = ncclShmem.channel.peers[recvPeer]->recv + connIndexRecv;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->stepCache = loadStepValue(peer->tailPtr = conn->tail);
        peer->headPtr = conn->head;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSize / sizeof(T);
        // 加载发送对端信息
        int sendPeer = mode == primsModePatAg ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        peer = ((struct ncclPatPeer*)sendPeers) + tid;
        conn = peer->conn = ncclShmem.channel.peers[sendPeer]->send + connIndexSend;
        peer->step = conn->step;
        peer->connFifo = conn->connFifo;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->stepCache = loadStepValue(peer->headPtr = conn->head);
        peer->tailPtr = conn->tail;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSize / sizeof(T);
      }
      if (tid == 0) {
        ncclShmem.groups[group].userInput = (void*)inputBuf;
        ncclShmem.groups[group].userOutput = (void*)outputBuf;
        ncclShmem.groups[group].redOpArgs = redOpArg; // scaler for local input
      }
      patBarrier();
    }
  }

  __device__ ~Primitives() {
    if (flags & PatMode) return;
    // 为下一次操作保存 步骤 进度
    if (flags & (RolePostSend | RolePostRecv)) conn->step = step;
    if ((flags & NetRegMode) && (flags & RoleWaitSend)) {
      // 确保在返回之前，代理 已经把数据发送出去。
      // 我们不希望下一个 CUDA 内核 覆盖发送缓冲区——
      // 该缓冲区是被(对端)直接访问的。
      uint64_t prevStep = step - StepPerSlice;
      volatile ssize_t* ptr = &(connFifo[prevStep % NCCL_STEPS].size);
      int spins = 0;
      while (*ptr != -1) {
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }

    if (flags & NetDeviceUnpack) {
      ncclNetDeviceSaveHead(netDeviceHandle, group, index);
    }

    // 确保所有线程都已完成对 conn->步骤 的回写，并且不再使用
    // ncclShmem.组[组]。
    barrier();

    if ((flags & DirectRead) && (flags & RoleWaitSend) && P2p) {
      // 对于 sendrecv 的 DirectRead，发送方需要等待接收方从 源 读完数据。
      // 这必须在 屏障() 之后进行，因为 后 线程可能与
      // 本次检查存在竞争。
      int spins = 0;
      volatile uint64_t* tail = conn->tail;
      volatile uint64_t* head = conn->head;
      while (*tail > *head) {
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }
  }

  __device__ void setDataPtrs(void const* inputBuf, void* outputBuf, uint64_t redOpArg, struct ncclDevWorkCollReg* work,
                              uint8_t ipcReg, int peer) {
    if (tid == 0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.groups[group].redOpArgs = redOpArg; // scaler for local input
    }

    if (Direct && ipcReg) {
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite);
      // 发送方提供直连缓冲区(供对端拉取)
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead);
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead); // receiver accepts direct buffer
      if (recvProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        // 在覆盖旧值之前，等待消费者先消费完它。
        if (slot) {
          T* exchgPtr;
          directBuff = (T*)outputBuf;
          while (*slot != nullptr && !checkAbort(flags, Aborted, spins));
          if (P2p) {
            exchgPtr = (T*)outputBuf;
          } else {
            // For 跨-clique P2P, 使用 对等端 rank directly to 避免 localRank conflicts 之间 cliques
            int localPeer = ncclShmem.comm.p2pCrossClique ? peer : ncclShmem.comm.rankToLocalRank[peer];
            // coverity[deref_parm:假] => work cannot be NULL 若 ipcReg != NULL
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
      if (sendAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // coverity[var_deref_op]
          directBuff = (T*)work->dnOutputs[index];
        }
      }
      if (sendProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        // 在覆盖旧值之前，等待消费者先消费完它。
        if (slot) {
          T* exchgPtr;
          while ((*slot != nullptr) && !checkAbort(flags, Aborted, spins));
          // 若无接收方，则是直接从输入缓冲区拉取(例如 directScatter)，
          // 否则是从输出缓冲区拉取(例如 recvCopyDirectSend)
          directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          if (P2p) {
            exchgPtr = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          } else {
            // For 跨-clique P2P, 使用 对等端 rank directly to 避免 localRank conflicts 之间 cliques
            int localPeer = ncclShmem.comm.p2pCrossClique ? peer : ncclShmem.comm.rankToLocalRank[peer];
            if (MaxRecv == 0)
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.sendbuffOffset + work->coll.sendbuffRmtAddrs[localPeer]);
            else
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }

          // 交换预缩放因子，供直连拉取时使用
          *slot = reinterpret_cast<T*>(exchgPtr);
        }
      }
      if (recvAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // Coverity complains about work being possibly NULL 下方.  然而, slot
          // being NULL means 那个 the NVLS 缓冲区 is 已注册 (regUsed == 1)
          // 所以 work can't be NULL 入 此 代码 路径.
          // coverity[var_deref_op]
          directBuff = (T*)work->dnInputs[index];
        }
      }
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (tid == 0) {
      ncclShmem.groups[group].userInput = (T*)ncclShmem.groups[group].userInput + delta;
      ncclShmem.groups[group].userOutput = (T*)ncclShmem.groups[group].userOutput + delta;
    }
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopy(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN,
                                                           bool postOp = false) {
    genericOp<1, 1, 1, 1, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<1, 1, 1, 1, -1, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvDirectSend(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 1, 1, 1, -1, -1>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvSend(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<1, 0, 1, 1, -1, -1>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, int eltN, bool postOp = false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
                                                             bool postOp = false) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN,
                                                           bool postOp = false) {
    // 直连仅适用于发送部分
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
                                                                 bool postOp = false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void scatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip,
                                          int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void directScatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset,
                                                int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void gather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip,
                                         int shift, bool postOp = false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void directGather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset,
                                               int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void patReduce(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) {
      patBarrier();
      patBarrier();
      return;
    } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv | RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend | RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims + ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims + ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + (step % NCCL_STEPS) * peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] =
        ((T*)peer->buff) + ((step + ps->stepOffset) % NCCL_STEPS) * peer->connStepSize + ps->sendOffset;
      if (peer->accSize < ps->sendOffset + nelem + (step + ps->stepOffset) * peer->connStepSize) {
        // 是新数据，把自己的数据加进去(规约)。
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
      } else {
        // 里面已有数据，应当累加而不是覆盖写入。
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 && (flags & RoleOutput)) {
      // 目标是本地的本地缓冲区
      ncclShmem.groups[group].dsts[0] = userOutput + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // 是新数据，把自己的数据加进去(规约)。
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // 里面已有数据，应当累加而不是覆盖写入。
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    patBarrier();
    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;
    if (ps->recvDim < 0) {
      srcs++;
      nSrcs--;
    } // No peer to receive from, remove one source

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, /*PreOpSrcs*/ 0>(tid, nthreads, ncclShmem.groups[group].redOpArgs,
                                                                    /*postOp=*/false, nSrcs, srcs, 1,
                                                                    ncclShmem.groups[group].dsts, workSize);

    // 在两个屏障之间保存 conn 步骤，确保下次重载能看到更新。
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step % NCCL_STEPS].size = (ps->sendOffset + nelem) * sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // 更新已累计处理的数据量 accSize
    if (ps->sendDim < 0 && (flags & RoleOutput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & RoleWaitSend))
      atomicMax(&peer->accSize, ps->sendOffset + nelem + (step + ps->stepOffset) * peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

  __device__ __forceinline__ void patCopy(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) {
      patBarrier();
      patBarrier();
      return;
    } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv | RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend | RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims + ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims + ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] =
        ((T*)peer->buff) + ((step + ps->stepOffset) % NCCL_STEPS) * peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      if (peer->accSize < ps->recvOffset + nelem + (step + ps->stepOffset) * peer->connStepSize) {
        // 是新数据，拷贝到输出缓冲区。
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + (step % NCCL_STEPS) * peer->connStepSize + ps->sendOffset;
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->recvDim < 0 && (flags & RoleInput)) {
      // 源是本地的本地缓冲区
      ncclShmem.groups[group].srcs[0] = userInput + ps->inpIx;
      if (localAccSize < ps->inpIx + nelem) {
        // 是新数据，拷贝到输出缓冲区。
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
        localAccSize = ps->inpIx + nelem;
      } else {
        // 已完成，跳过
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0];
      }
    }
    patBarrier();
    int nDsts = 2;
    void** dsts = ncclShmem.groups[group].dsts;
    if (ps->sendDim < 0) {
      dsts++;
      nDsts--;
    } // No peer to send to, remove one dest
    if (ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[1]) nDsts--; // In-place or already done.

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 2, /*PreOpSrcs*/ 0>(tid, nthreads, ncclShmem.groups[group].redOpArgs,
                                                                    /*postOp=*/false, 1, ncclShmem.groups[group].srcs,
                                                                    nDsts, dsts, workSize);

    // 在两个屏障之间保存 conn 步骤，确保下次重载能看到更新。
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step % NCCL_STEPS].size = (ps->sendOffset + nelem) * sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // 更新已累计处理的数据量 accSize
    if (ps->recvDim < 0 && (flags & RoleInput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->recvDim >= 0 && (flags & RoleWaitRecv))
      atomicMax(&peer->accSize, ps->recvOffset + nelem + (step + ps->stepOffset) * peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }
};
