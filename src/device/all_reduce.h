/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

/* ============================================================================
 * device/all_reduce.h —— AllReduce 的 GPU 端 kernel 模板（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 在 AllReduce 全链路中的定位：这是真正在 GPU 上跑的代码，由 enqueue.cc 启动。
 * 它根据算法(ring/tree)与协议(LL/LL128/Simple)实现“把各 rank 的数据规约到一起，
 * 再把结果广播回每个 rank”。
 *
 * 核心模板（本仓库保留的）：
 *   - runRing<T,RedOp,Proto>  : ring 算法——“边收边发、滚动规约”。每个 rank 沿环
 *     只跟 prev/next 交换，nRanks-1 步后每个 rank 都拿到完整规约结果。
 *   - runTree<T,RedOp,Proto>  : tree 算法——上行 reduce（子→父）做部分规约，下行
 *     broadcast（父→子）把结果发回去。
 *   - runColl  : 根据 work 里记录的算法/协议，分派到上面的具体 run* 模板。
 *
 * Proto 决定同步与搬运方式：LL(带 flag 同步位的低延迟)、LL128、Simple(直接 load/store)。
 * 本仓库测试用的是 Simple（见 tests/src/all_reduce.cu 的 case 0）。
 * ============================================================================
 */

namespace {
// 环 算法的 设备 函数：tid/nthreads 是当前线程块内线程标识，work 是本次要做的
// 全规约 工作描述（含 sendbuff/recvbuff/计数/redOp 等）。它构造 Primitives 原语，
// 沿 环 的 prev/下一个 邻居循环做“接收-规约-发送”，实现滚动全规约。
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
  // ncclShmem 是每个线程块共享内存里的“本 通道 上下文”，由 内核 入口预先加载好。
  // 环 里保存了本 通道 的环信息：索引(本 rank 在环上的位置)、prev/下一个(环上前后邻居)。
  ncclRing* ring = &ncclShmem.channel.ring;
  int ringIx = ring->index;                    // 本 rank 在这条环上的序号(0..nranks-1)
  const int nranks = ncclShmem.comm.nRanks;    // 参与本次集合通信的 rank 总数
  ssize_t gridOffset;                          // 本 channel 负责的数据区在整个 buffer 中的起始元素偏移
  ssize_t channelCount;                        // 本 channel 负责处理的元素总个数
  ssize_t chunkCount;                          // 每个 chunk(最小调度粒度)的元素个数
  // 按 通道 切分工作：把整个 计数 均分给各 通道，并算出 块 粒度。
  // 块 大小与协议(Simple/LL/LL128)相关，因为不同协议的 FIFO 槽位容量不同。
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount,
                  &chunkCount);
  // 一轮完整的环形流水要处理 nranks 个 块(每个 rank 各“负责”一个 块 的规约)
  const ssize_t loopCount = nranks * chunkCount;
  ssize_t offset;   // 当前要处理的数据在 buffer 中的元素偏移
  int nelem;        // 当前这一步实际搬运/规约的元素个数(尾部可能不足一个 chunk)
  int chunk;        // 当前处理的是第几个 chunk(即“数据块归属哪个 rank”)

  // Coverity 静态检查会认为被调用方把 &环->下一个 当数组用；但由于这里用的是
  // FanSymmetric<1>(收发扇入扇出都只有 1)，实际只会访问第一个元素，因此是安全的。
  // coverity[callee_ptr_arith:假]
  // 构造通信原语对象 prims：
  //   - FanSymmetric<1> : 一个接收源(prev) + 一个发送目标(下一个)，正是环形拓扑的形态
  //   - Direct=1        : 允许使用“直连指针”，即 P2P 可直接读写对端显存，省一次中转拷贝
  //   - prims 的构造过程会完成与邻居的连接建立、FIFO 指针/步进计数器的初始化
  Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff,
                                                           work->recvbuff, work->redOpArg, 0, 0, 0, work);

  // 外层循环：数据量可能远大于一轮流水能处理的量，因此按 loopCount 为步长分多轮跑
  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    ssize_t remCount = channelCount - elemOffset;   // 本轮还剩多少元素没处理
    ssize_t chunkOffset;                            // chunk 在本轮内部的元素偏移

    // 最后一轮数据不足一整轮时，缩小 chunkCount 让 nranks 个 块 刚好覆盖剩余数据；
    // alignUp 到 16 字节边界是为了保证向量化访存(128-位 加载/存储)仍然对齐。
    if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16 / sizeof(T));

    // 环形下标取模：把可能超过 nranks 的下标折回 [0, nranks) 区间。
    // 用减法代替 % 运算，是因为入参最大不超过 2*nranks-1，一次减法就够，且比取模快。
    auto modRanks = [&] __device__(int r) -> int { return r - (r >= nranks ? nranks : 0); };

    /* ------------------------------------------------------------------
     * Ring AllReduce = ReduceScatter(前 nranks-1 步) + AllGather(后 nranks-1 步)
     *
     * 核心思想：把数据切成 nranks 个 chunk，让第 i 个 chunk 的“规约任务”落在
     * 某个特定 rank 上。数据沿环单向流动，每经过一个 rank 就累加一次，
     * 转满 nranks-1 步后，每个 chunk 都在其“归属 rank”上得到了完整的规约结果；
     * 再沿环转 nranks-1 步把结果传一圈，所有 rank 就都拿到了全部结果。
     *
     * 这样做的好处：每个 rank 收发的总数据量只有 2*(nranks-1)/nranks * N，
     * 接近理论下界，且带宽利用与 nranks 无关(而不是朴素做法的 O(N*nranks))。
     * ------------------------------------------------------------------ */

    // 【第 0 步】把“上游第 1 个 块”原样发给下一个 GPU，为流水线注水。
    // 选 ringIx-1 号 块 是因为：它需要在环上走满 nranks-1 跳才能到达自己的归属 rank。
    // 这一步只发不收(此时还没有数据可收)，所以用 directSend 而不是 recvReduceSend。
    chunk = modRanks(ringIx + nranks - 1);
    chunkOffset = chunk * chunkCount;
    offset = gridOffset + elemOffset + chunkOffset;
    nelem = (int)min(chunkCount, remCount - chunkOffset);
    prims.directSend(offset, offset, nelem);

    // 【第 1 ~ k-2 步】ReduceScatter 主体：收 + 规约 + 转发。
    // 每一步从 prev 收到一个“已被上游累加过的部分和”，与本地对应 块 相加，
    // 再把新的部分和发给 下一个。数据在环上滚动，累加的 rank 数逐步增加。
    // 注意 j 从 2 开始递增、块 编号递减：保证每一步处理的都是“再走 nranks-j 跳就到家”的那个 块。
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }

    // 【第 k-1 步】ReduceScatter 收尾：轮到本 rank “归属”的那个 块(编号 = ringIx)。
    // 此时收到的部分和已经累加了其余 nranks-1 个 rank 的数据，再加上本地数据后
    // 就是该 块 的最终完整结果。因此这一步要同时做三件事：
    //   1) recvReduce : 收下并完成最后一次累加
    //   2) 拷贝       : 把最终结果写入本 rank 的 recvbuff(所以叫 ReduceCopy)
    //   3) DirectSend : 把最终结果发给 下一个，开启 全收集 阶段
    // postOp=真 表示在此处执行规约算子的“后处理”(如 Avg 需要除以 nranks，
    // 必须在累加完全部 rank 之后、且只做一次)。
    chunk = ringIx + 0;
    chunkOffset = chunk * chunkCount;
    offset = gridOffset + elemOffset + chunkOffset;
    nelem = (int)min(chunkCount, remCount - chunkOffset);
    prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

    // 【全收集 阶段，共 k-2 步】此时环上流动的已经是最终结果，不需要再规约。
    // 每步：从 prev 收到某个 块 的最终结果 -> 写入本地 recvbuff -> 原样转发给 下一个。
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvCopyDirectSend(offset, offset, nelem);
    }

    // 【最后一步】收下环上传回的最后一个 块，只写入本地即可。
    // 不再转发的原因：这个 块 的下一站正是它的“归属 rank”，那里早就有结果了，
    // 再发一次纯属浪费带宽，所以用 directRecv(只收不发)收尾。
    chunk = modRanks(ringIx + 1);
    chunkOffset = chunk * chunkCount;
    offset = gridOffset + elemOffset + chunkOffset;
    nelem = (int)min(chunkCount, remCount - chunkOffset);

    prims.directRecv(offset, nelem);
  }
}

/*
 * Tree AllReduce(上行/下行两趟串行版本)
 * ----------------------------------------------------------------------------
 * 与 ring 不同，tree 把所有 rank 组织成一棵(二叉)树，分两个阶段：
 *   1) Reduce(上行)   : 叶子 -> 父节点逐层规约，最终根节点(up == -1)拿到完整结果
 *   2) Broadcast(下行): 根 -> 子节点逐层广播，把最终结果发回所有 rank
 *
 * 相比 ring 的 O(nranks) 步，tree 只需 O(log nranks) 步，因此在**小数据量**、
 * **多节点**场景下延迟明显更低；但每步搬运的数据量更大，大数据量时带宽不如 ring。
 * 这个 UpDown 版本把两个阶段严格串行执行，实现简单但流水利用率不如 runTreeSplit。
 */
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
  // 树 中记录了本 rank 在树上的位置：up 是父节点，down[] 是子节点数组
  ncclTree* tree = &ncclShmem.channel.tree;
  size_t gridOffset;    // 本 channel 数据区起始偏移
  size_t channelCount;  // 本 channel 负责的元素个数
  size_t chunkCount;    // 单个 chunk 的元素个数
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount,
                  &chunkCount);
  size_t offset;
  int nelem;

  { // 阶段一 Reduce(上行)：最多从 3 个来源接收，最多向 1 个目标发送(二叉树两个子节点 + 本地)
    // FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>：扇入最多 ARITY 个子节点，扇出只有 1 个父节点
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims(
      tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
    if (tree->up == -1) {
      // 角色①：根节点。没有父节点可发，只需收齐所有子节点的部分和并与本地数据规约，
      // 结果直接落到 recvbuff。postOp=真 表示在此完成 Avg 之类的后处理(全局只做一次)。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    } else if (tree->down[0] == -1) {
      // 角色②：叶子节点。没有子节点可收，直接把本地 sendbuff 的数据发给父节点。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directSend(offset, offset, nelem);
      }
    } else {
      // 角色③：中间节点。收子节点的部分和 -> 与本地数据规约 -> 把新部分和转发给父节点。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }
    }
  }

  { // 阶段二 Broadcast(下行)：最多从 1 个来源接收，最多向 3 个目标发送(二叉树两个子节点 + 本地)
    // FanAsymmetric<1, NCCL_MAX_TREE_ARITY>：扇入只有 1 个父节点，扇出最多 ARITY 个子节点
    // 注意与上行阶段的模板参数正好相反，因为数据流方向反过来了。
    Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims(
      tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
    if (tree->up == -1) {
      // 角色①：根节点。上行阶段结束时结果已在 recvbuff 中，因此这里从“输出缓冲区”
      // 直接发给子节点(FromOutput 即读 recvbuff 而非 sendbuff)，无需再接收。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directSendFromOutput(offset, nelem);
      }
    } else if (tree->down[0] == -1) {
      // 角色②：叶子节点。没有子节点要转发，收下父节点发来的最终结果写入 recvbuff 即可。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecv(offset, nelem);
      }
    } else {
      // 角色③：中间节点。收父节点的结果 -> 存入本地 recvbuff -> 同时转发给自己的子节点。
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }
    }
  }
}

/*
 * Tree AllReduce(线程分组并行版本，性能优于 runTreeUpDown)
 * ----------------------------------------------------------------------------
 * 与 UpDown 版“先做完整个上行、再做整个下行”不同，这里把线程块里的线程**一分为二**：
 *   - 前 nthreadsSplit 个线程：专职负责 Reduce(上行)
 *   - 其余线程：           专职负责 Broadcast(下行)
 * 两组线程同时运行，形成流水：当上行组还在规约后面的 chunk 时，下行组已经在广播
 * 前面 chunk 的结果了，从而把上下行两个阶段的时间重叠起来，显著提升带宽利用率。
 *
 * 这也是实际运行中 tree 算法默认走的路径。
 */
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
  ncclTree* tree = &ncclShmem.channel.tree;
  size_t gridOffset;
  size_t channelCount;
  size_t chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount,
                  &chunkCount);
  size_t offset;
  int nelem;
  int nthreadsSplit;   // 分界线：tid < nthreadsSplit 的线程做 Reduce，其余做 Broadcast
  if (Proto::Id == NCCL_PROTO_SIMPLE) {
    // Simple 协议：上下行工作量相当，基本对半分。
    // 当线程数较多时给 规约 组多分 64 个线程(即两个 线程束)，因为规约端还要做加法运算。
    nthreadsSplit = nthreads / 2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;
  } else {
    // LL 与 LL128 协议：
    // “从最多 3 个来源接收并规约”比“向 3 个目标发送”计算量大得多
    // (前者要做数据比对、标志 校验和多次加法)，因此按 70% 给 规约、30% 给 bcast 分配。
    // 结果要向下取整到 WARP_SIZE 的整数倍，保证每组线程都是完整的 线程束，避免 线程束 分裂。
    nthreadsSplit = (nthreads * 7 / (10 * WARP_SIZE)) * WARP_SIZE;
  }

  if (tree->up == -1) {
    // 角色①：根节点。它既是上行的终点又是下行的起点，无需拆分线程，
    // 全部线程一起做“收齐子节点数据 -> 规约 -> 立刻发回子节点”。
    // FanSymmetric<ARITY_TOP>：收发对象都是同一批子节点，故用对称扇形。

    Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0> prims(
      tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
    }
  } else if (tid < nthreadsSplit) {
    /* 【上行组】负责 Reduce：最多从 3 个来源接收，最多向 1 个目标发送(二叉树两子节点 + 本地)。
     *
     * 为什么这里 Direct=1？
     * 答：即使这条路径实际上并不执行任何 direct(直连显存读写)操作，构造函数也必须
     *     按 Direct 模式来初始化。因为对端(远程 rank)的构造函数可能是 Direct 模式的，
     *     双方在建连握手时需要**交换直连指针**；如果这边不按 Direct 走，握手信息对不上，
     *     就会导致双方互相等待而**挂死(hang)**。
     *     更干净的做法是把能力拆分成 DirectRecv 和 DirectSend 两个独立开关：
     *     那样这里可以两个都设为 0，而上面根节点分支则设为 DirectRecv=0、DirectSend=1。
     */
    // Coverity 静态检查认为被调用方把 &树->up 当数组用；但由于使用了
    // FanAsymmetric<n, 1>(发送扇出为 1)，实际只会访问第一个元素，因此是安全的。
    // coverity[callee_ptr_arith:假]
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims(
      // 注意第 7 个参数 组=0*MaxGroupWidth：上行组使用同步组 0，
      // 与下面下行组的同步组 1 区分开，避免两组线程的 屏障 相互干扰。
      tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg,
      0 * Proto::MaxGroupWidth, 0, 0, work);
    if (tree->down[0] == -1) {
      // 叶子节点：无子节点可收，直接把本地数据发给父节点
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directSend(offset, offset, nelem);
      }
    } else {
      // 中间节点：收子节点部分和 -> 与本地规约 -> 转发给父节点
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }
    }
  } else {
    // 【下行组】负责 广播：最多从 1 个来源接收，最多向 3 个目标发送(二叉树两子节点 + 本地)
    // Coverity 静态检查认为被调用方把 &树->up 当数组用；但由于使用了
    // FanAsymmetric<1, n>(接收扇入为 1)，实际只会访问第一个元素，因此是安全的。
    // coverity[callee_ptr_arith:假]
    Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims(
      tid - nthreadsSplit, nthreads - nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
      // 下行组使用同步组 1(1*MaxGroupWidth)，并且把 tid 重新映射到 [0, nthreads-nthreadsSplit)，
      // 这样在原语内部看来它就是一个独立的、从 0 号线程开始的线程组。
      work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
    if (tree->down[0] == -1) {
      // 叶子节点：只需收下父节点广播来的最终结果
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecv(offset, nelem);
      }
    } else {
      // 中间节点：收父节点结果 -> 写本地 -> 继续向下广播给子节点
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }
    }
  }
}
} // namespace

/* ============================================================================
 * 以下是 RunWorkColl 模板特化：算法(ALGO) × 协议(PROTO) 的分派表。
 * kernel 入口会根据 work 中记录的 algorithm/protocol 实例化对应特化版本，
 * 从而在**编译期**就选定代码路径，运行时没有任何分支开销。
 * ============================================================================ */

// 特化：环 算法 + Simple 协议 —— 单机多卡 全规约 大数据量时的默认路径
template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // ProtoSimple<chunkSteps, sliceSteps>：把一个 块 再细分成若干 slice 做流水，
    // 使“数据搬运”与“同步等待”能够重叠，从而掩盖延迟、提高带宽。
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

// 特化：树 算法 + Simple 协议 —— 小数据量/跨节点时延迟更优
template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    // 这里的条件编译是为了规避特定 CUDA 版本的编译器缺陷：
    // 在 CUDA 11.2~11.3 且 Ampere(sm_80+) 架构上，runTreeSplit 的线程分组写法
    // 会触发编译器生成错误代码，因此退回到实现更简单的 UpDown 串行版本。
#if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
    runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
#else
    runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
#endif
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int /*nthreads*/, struct ncclDevWorkColl* work) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    const ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t size = work->collnet.count;
    const ssize_t loopSize = nChannels * direct->nHeads * chunkSize;

    const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS :
                                             hasUp            ? 3 * COLLNET_COPY_THREADS :
                                                                0);
    const int nThreadsGather = ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2 * COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS :
                                           hasUp            ? 0 :
                                                              2 * COLLNET_COPY_THREADS);
    const int nThreadsReduce = work->nWarps * WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;
    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // 散射(散播)：把数据分发到各个对端
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0> prims(
        tid - tidStartScatter, nThreadsScatter, NULL, direct->up, work->sendbuff, work->recvbuff, work->redOpArg,
        2 * Proto::MaxGroupWidth, 1, 1, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      // 对于 collnet UB(用户缓冲区)场景，需要以不同方式组织缓冲区以保证跨 通道 的连续访问
      // 该访问模式必须与 coll_net.cc 中的代码保持一致
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.scatter(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
      // Coverity 静态检查报告 prims 析构函数中可能存在越界，但这实际上是
      // 误报。
      // coverity[overrun-调用:假]
    } else if (tid >= tidStartReduce && direct->out != -1) {
      if (hasDn) {
        // 规约后发送到网络
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0> prims(
          tid - tidStartReduce, nThreadsReduce, direct->down, &direct->out, work->sendbuff, work->recvbuff,
          work->redOpArg, 3 * Proto::MaxGroupWidth, 1, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize :
                                              gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.recvReduceDirectSend(offset, offset, nelem);
        }
      } else {
        // 直接发送到网络
        if (work->netRegUsed) {
          if (tid == tidStartReduce) {
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0> prims(tid - tidStartReduce, nThreadsReduce,
                                                                                  nullptr, &direct->out, work->sendbuff,
                                                                                  work->recvbuff, work->redOpArg,
                                                                                  3 * Proto::MaxGroupWidth, 1, 1);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.send(offset, nelem);
          }
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // 聚集(收集)：从各个对端收集数据
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0> prims(
        tid, nThreadsGather, direct->up, NULL, work->sendbuff, work->recvbuff, work->redOpArg, 0 * Proto::MaxGroupWidth,
        0, 0, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.directGather(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      if (hasDn) {
        // 从网络接收，然后广播
        // Coverity 静态检查报告下面这个类中可能存在越界，但这实际上是
        // 误报。
        // coverity[identity_transfer:假]
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0> prims(
          tid - tidStartBcast, nThreadsBcast, &direct->out, direct->down, work->sendbuff, work->recvbuff,
          work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize :
                                              gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      } else {
        if (work->netRegUsed) {
          if (tid == tidStartBcast) {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
          }
          __syncwarp();
        } else {
          // 从网络接收(无需 后 线程)
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0> prims(tid - tidStartBcast, nThreadsBcast,
                                                                                  &direct->out, nullptr, work->sendbuff,
                                                                                  work->recvbuff, work->redOpArg,
                                                                                  1 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.recv(offset, nelem, /*postOp=*/true);
          }
        }
      }
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int /*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const bool hasOut = nvls->out != -1;
    const int nranks = ncclShmem.comm.nRanks;
    const int totalWarps = NCCL_MAX_NTHREADS / WARP_SIZE;
    const int bcastWarps = hasOut ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;

    const int nThreadsScatter = scatterWarps * WARP_SIZE;
    const int nThreadsGather = gatherWarps * WARP_SIZE;
    const int nThreadsReduce = reduceWarps * WARP_SIZE;
    const int nThreadsBcast = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (work->oneNode) {
      ssize_t gridOffset, channelCount, chunkSize;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset,
                      &channelCount, &chunkSize);
      const ssize_t loopCount = nvls->nHeads * chunkSize;
      int remCount = channelCount % (nvls->nHeads * chunkSize);
      int lastChunkSize = alignUp(divUp(remCount, nvls->nHeads), 16384 / sizeof(T));

      if (tid < tidEndScatter) {
        // 散射(散播)：把数据分发到各个对端
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0> prims(
          tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL, work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // 聚集(收集)：从各个对端收集数据
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0> prims(
          tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff, work->redOpArg,
          1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // 规约后通过 NVLS 广播
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(tid - tidEndGather, nThreadsReduce,
                                                                            &nvls->down, &nvls->down, NULL, NULL,
                                                                            work->redOpArg, 2 * Proto::MaxGroupWidth, 0,
                                                                            0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset, offset;
          int nelem;
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          chunkOffset = elemOffset + nvls->headRank * chunkSize;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkSize, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else {
      const int bid = ncclShmem.channelId - work->channelLo;
      const int nChannels = work->channelHi - work->channelLo + 1;
      const ssize_t chunkSize = work->collnet.chunkCount;
      const ssize_t loopSize = nChannels * nvls->nHeads * chunkSize;
      const ssize_t size = work->collnet.count;

      if (tid < tidEndScatter) {
        // 散射(散播)：把数据分发到各个对端
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0> prims(
          tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL, work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
        // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
      } else if (tid < tidEndGather) {
        // 聚集(收集)：从各个对端收集数据
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0> prims(
          tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff, work->redOpArg,
          1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // 规约后发送到网络
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity 静态检查报告下面这个类中可能存在越界，但这实际上是
        // 误报。
        // coverity[identity_transfer:假]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(tid - tidEndGather, nThreadsReduce,
                                                                            &nvls->down, &nvls->out, NULL,
                                                                            work->recvbuff, work->redOpArg,
                                                                            2 * Proto::MaxGroupWidth, 0, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ?
                             gridOffset + (nvls->headRank * nChannels + bid) * chunkSize :
                             gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else if (tid < tidEndBcast && nvls->headRank != -1) {
        // 从网络接收，然后广播
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        // Coverity 静态检查报告下面这个类中可能存在越界，但这实际上是
        // 误报。
        // coverity[identity_transfer:假]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(tid - tidEndReduce, nThreadsBcast,
                                                                            &nvls->out, &nvls->down, NULL,
                                                                            work->recvbuff, work->redOpArg,
                                                                            3 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ?
                             gridOffset + (nvls->headRank * nChannels + bid) * chunkSize :
                             gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int /*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const int treeUp = nvls->treeUp;
    const int* treeDown = nvls->treeDown;
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset,
                    &channelCount, &chunkCount);
    const ssize_t loopCount = nvls->nHeads * chunkCount;
    const int nranks = ncclShmem.comm.nRanks;
    const bool hasUp = treeUp != -1;
    const int totalWarps = NCCL_MAX_NTHREADS / WARP_SIZE;
    const int bcastWarps = hasUp ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 4) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasUp ? 5 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
    ssize_t offset;
    int nelem;
    int remCount = channelCount % (nvls->nHeads * chunkCount);
    int lastChunkCount = alignUp(divUp(remCount, nvls->nHeads), 16 / sizeof(T));

    const int nThreadsScatter = scatterWarps * WARP_SIZE;
    const int nThreadsGather = gatherWarps * WARP_SIZE;
    const int nThreadsReduce = reduceWarps * WARP_SIZE;
    const int nThreadsBcast = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (tid < tidEndScatter) {
      // 散射(散播)：把数据分发到各个对端
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0> prims(
        tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL, work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.scatter(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // 聚集(收集)：从各个对端收集数据
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0> prims(
        tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff, work->redOpArg,
        1 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.gather(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndReduce && nvls->headRank != -1) {
      if (!hasUp) {
        // 规约与广播
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<3>, /*Direct=*/1, Proto, 0> prims(tid - tidEndGather, nThreadsReduce,
                                                                            treeDown, treeDown, NULL, NULL,
                                                                            work->redOpArg, 2 * Proto::MaxGroupWidth, 0,
                                                                            0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else {
        // 规约后发送到网络
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity 认为被调用方把 &treeUp 当作数组使用；但由于使用了
        // FanAsymmetric<3, 1>，实际只会访问第一个元素，因此是安全的。
        // coverity[callee_ptr_arith:假]
        Primitives<T, RedOp, FanAsymmetric<3, 1>, /*Direct=*/1, Proto, 0> prims(tid - tidEndGather, nThreadsReduce,
                                                                                treeDown, &treeUp, NULL, NULL,
                                                                                work->redOpArg,
                                                                                2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else if (tid < tidEndBcast && nvls->headRank != -1) {
      // 从网络接收，然后广播
      using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
      // Coverity 认为被调用方把 &treeUp 当作数组使用；但由于使用了
      // FanAsymmetric<1, 3>，实际只会访问第一个元素，因此是安全的。
      // coverity[callee_ptr_arith:假]
      Primitives<T, RedOp, FanAsymmetric<1, 3>, /*Direct=*/1, Proto, 0> prims(tid - tidEndReduce, nThreadsBcast,
                                                                              &treeUp, treeDown, NULL, NULL,
                                                                              work->redOpArg, 3 * Proto::MaxGroupWidth,
                                                                              0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t chunkOffset;
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        chunkOffset = elemOffset + nvls->headRank * chunkCount;
        offset = gridOffset + chunkOffset;
        nelem = min(chunkCount, channelCount - chunkOffset);
        prims.directRecvDirectSend(offset, offset, nelem);
      }
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    ncclTree* tree = &ncclShmem.channel.collnetChain;
    ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t loopSize = int(nChannels * chunkSize);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t size = work->collnet.count;

    int nthreadsSplit = nthreads / 2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, connIndex, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      // 沿链向上规约
      group = 0;
      connIndex = 1;
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      // 沿链向下广播
      group = 1;
      connIndex = 0;
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads - nthreadsSplit;
    }

    if (tid < nthreadsSplit) {
      if (recv == -1) {
        if (work->netRegUsed) {
          if (groupTid == 0) {
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::sendPeerNotify(send, connIndex, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
            groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff, work->redOpArg,
            group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
            prims.directSend(offset, offset, nelem);
          }
          // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
        }
      } else {
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
          groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff, work->redOpArg,
          group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * int(chunkSize);
          int nelem = min(chunkSize, size - offset);
          // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
        // coverity[overrun-调用] => Coverity 认为 prims.索引 可以 greater than 1
      }
    } else {
      if (recv == nranks) {
        // 我是广播链的第一个节点，需要在这里执行除法(postOp，例如 Avg 求平均)
        if (send == -1) {
          if (work->netRegUsed) {
            if (groupTid == 0) {
              Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::recvPeerNotify(recv, connIndex, 1);
            }
            __syncwarp();
          } else {
            // Coverity 认为被调用方把 &发送 当作数组使用；但由于使用了
            // FanSymmetric<1>，实际只会访问第一个元素，因此是安全的。
            // coverity[callee_ptr_arith:假]
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
              groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff, work->redOpArg,
              group * Proto::MaxGroupWidth, connIndex, connIndex, work);
            for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
              ssize_t offset = gridOffset + bid * int(chunkSize);
              int nelem = min(chunkSize, size - offset);
              prims.directRecv(offset, nelem, /*postOp*/ true);
            }
          }
        } else {
          // Coverity 认为被调用方把 &发送 当作数组使用；但由于使用了
          // FanSymmetric<1>，实际只会访问第一个元素，因此是安全的。
          // coverity[callee_ptr_arith:假]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
            groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff, work->redOpArg,
            group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp*/ true);
          }
        }
      } else {
        // Coverity 认为被调用方把 &发送 当作数组使用；但由于使用了
        // FanSymmetric<1>，实际只会访问第一个元素，因此是安全的。
        // coverity[callee_ptr_arith:假]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
          groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff, work->redOpArg,
          group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        if (send == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecv(offset, nelem);
          }
        } else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem);
          }
        }
      }
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
