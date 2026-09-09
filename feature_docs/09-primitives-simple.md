# 09 —— 通信原语：Simple 协议（`src/device/prims_simple.h`）

> 本文是 mini-nccl（从 NVIDIA NCCL 2.30.7 抽取的单机多卡最小通信库，只保留 AllReduce 全链路）的 feature 文档之一。
> 所有代码引用均指向本仓库源码，行号已逐条核对。

## 本文覆盖的源文件

| 文件 | 作用 | 本文引用位置 |
| --- | --- | --- |
| [`src/device/primitives.h`](../src/device/primitives.h) | 协议常量类 `ProtoSimple`/`ProtoLL`/`ProtoLL128`、`Fan` 类、`Primitives` 主模板声明、`checkAbort` | 主题 1 |
| [`src/device/prims_simple.h`](../src/device/prims_simple.h) | Simple 协议的 `Primitives` 特化：连接加载、FIFO 流控、`genericOp`、Direct、线程角色 | 主题 2~7 |
| [`src/device/common_kernel.h`](../src/device/common_kernel.h) | `reduceCopy`（真正做多源规约 + 多目标写出的内核）、`loadInt` | 主题 4、7 |
| [`src/device/common.h`](../src/device/common.h) | `ncclShmemGroup`（shared memory 里的收发指针/用户指针）、`barrier_sync`/`barrier_red_or` 的 PTX 封装 | 主题 2、3 |
| [`src/include/device.h`](../src/include/device.h) | `NCCL_STEPS`、`ncclConnInfo`、`WARP_SIZE`、线程数常量、`buffSizes` | 主题 2、3、7 |
| [`src/device/all_reduce.h`](../src/device/all_reduce.h) | `runRing` 如何按 `directSend / directRecvReduceDirectSend / ...` 调用原语；`ProtoSimple<2,2>` 的实例化 | 主题 1、4、5 |
| [`src/include/collectives.h`](../src/include/collectives.h) | `ALLREDUCE_CHUNKSTEPS` / `ALLREDUCE_SLICESTEPS` | 主题 5 |
| [`src/init.cc`](../src/init.cc) | `DEFAULT_BUFFSIZE` 等缓冲区大小常量与 `computeBuffSizes` | 主题 3、7 |
| [`src/graph/tuning.cc`](../src/graph/tuning.cc) | 三协议的延迟/带宽建模常数 | 主题 7 |

---

## 主题 1：`Primitives` 抽象层 —— 用模板参数做协议多态

### ① 解决什么问题

`all_reduce.h` 里的 `runRing` / `runTreeSplit` 只描述"沿环收一圈、发一圈"这类**算法语义**，它不应该关心"数据到底是靠 flag 搬运还是靠 head/tail 搬运"。但三种协议（Simple / LL / LL128）的数据布局、同步方式、切片粒度完全不同。如果写成运行时 `switch`，每次收发都要付分支代价，且无法让编译器按协议常量展开循环。

### ② 一句话本质

**`Primitives<T, RedOp, Fan, Direct, Proto, P2p, isNetOffload>` 是一个只在编译期实例化的模板类；`Proto` 位置放 `ProtoSimple<...>` / `ProtoLL` / `ProtoLL128` 之一，通过偏特化在编译期选中一整套收发实现。**

### ③ 代码链路

1. 主模板只有声明、没有定义 —— [`primitives.h:128-129`](../src/device/primitives.h#L128)
   ```cpp
   template <typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
   class Primitives;
   ```
2. 三个偏特化实现分别落在 [`prims_simple.h:38-41`](../src/device/prims_simple.h#L38)、[`prims_ll.h:16-18`](../src/device/prims_ll.h#L16)、[`prims_ll128.h:20-22`](../src/device/prims_ll128.h#L20)，由 [`primitives.h:179-181`](../src/device/primitives.h#L179) 三个 `#include` 拼装。
3. 算法层只写协议无关代码 —— [`all_reduce.h:62-63`](../src/device/all_reduce.h#L62) 构造 `prims`，[`all_reduce.h:97`](../src/device/all_reduce.h#L97) 起的循环里调用 `prims.directSend / directRecvReduceDirectSend / ...`。
4. 协议的具体常量在 `RunWorkColl` 特化里钉死 —— [`all_reduce.h:348-355`](../src/device/all_reduce.h#L348)：`RunWorkColl<AllReduce, T, RedOp, RING, SIMPLE>` 里 `using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;`。

### ④ 关键代码逐行解读

[`primitives.h:37-57`](../src/device/primitives.h#L37)：`ProtoSimple` 携带 Simple 协议独有的三个编译期数字。

```cpp
template <int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0,
          int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  // 数据 字节 (无 标志 etc) 入 one 步骤 的 fifo 队列.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
  }
  // Granularity of 数据 字节 transferred 每个 线程.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // 组 width is 如何 许多 consecutive 组 值 a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
};
```

* `Id`：协议枚举值（`NCCL_PROTO_SIMPLE == 2`，见 [`nccl_tuner.h:43-46`](../src/include/plugin/nccl_tuner.h#L43)），算法层用它向 `ncclCollCbdPart` 询问"本协议的 chunk 该多大"。
* `SlicePerChunk` / `StepPerSlice`：一个 chunk 切几片、一片跨几个 FIFO slot（主题 5）。
* `Unroll`：传给 `reduceCopy` 的向量化展开因子（主题 4）。
* `calcBytePerStep()`：**Simple 协议一个 FIFO slot 里放的全是数据，没有 flag**，所以一个 step 的可用字节 = `buffSizes[SIMPLE] / NCCL_STEPS`。对比 `ProtoLL::calcBytePerStep()` 要 `/2`（[`primitives.h:63-65`](../src/device/primitives.h#L63)）、`ProtoLL128` 要乘 `15/16`（[`primitives.h:78-80`](../src/device/primitives.h#L78)）—— 这一行就是三协议"有效带宽"差异的根源。
* `MaxGroupWidth = 2`：Simple 的一个逻辑"组"要占 2 个 barrier 编号（因为 tree 算法会把线程块劈成上行/下行两组，见 [`all_reduce.h:294`](../src/device/all_reduce.h#L294) 和 [`all_reduce.h:320`](../src/device/all_reduce.h#L320) 的 `0 * / 1 * Proto::MaxGroupWidth`）；LL/LL128 是 1。

模板参数各自含义（对照 [`prims_simple.h:38-41`](../src/device/prims_simple.h#L38)）：

| 参数 | 含义 |
| --- | --- |
| `T` | 元素类型（`float` / `__half` / …），决定 `stepSize` 与向量化宽度 |
| `RedOp` | 规约算子（`ncclDevRedOpFull` 之上的 Functor），也决定 `applyPreOp/applyReduce/applyPostOp` |
| `Fan` | **扇入扇出上限**。`FanSymmetric<N>`：编译期保证收发个数相同，运行时只存 1 个 int（[`primitives.h:111-125`](../src/device/primitives.h#L111)）；`FanAsymmetric<MaxRecv, MaxSend>`：收发独立（[`primitives.h:95-109`](../src/device/primitives.h#L95)）。ring 用 `FanSymmetric<1>`，tree 上行用 `FanAsymmetric<TREE_ARITY, 1>`、下行用 `FanAsymmetric<1, TREE_ARITY>`（[`all_reduce.h:172`](../src/device/all_reduce.h#L172)、[`all_reduce.h:202`](../src/device/all_reduce.h#L202)）。 |
| `Direct` | 是否**允许**使用直连（零拷贝）路径。它是"能力开关"而非"一定走直连"：真正启用还要看运行时 `ipcReg`（用户 buffer 注册过）与连接 flags（`NCCL_P2P_WRITE` / `NCCL_P2P_READ`），见 [`prims_simple.h:545-567`](../src/device/prims_simple.h#L545)。`all_reduce.h` 的 ring/tree 一律传 `1`（[`all_reduce.h:62`](../src/device/all_reduce.h#L62)）。 |
| `Proto` | `ProtoSimple<...>` / `ProtoLL` / `ProtoLL128`，选实现 |
| `P2p` | 1 表示这是 P2P（sendrecv）work 而不是集合通信 work，影响 `p2pWork` / `collWork` 的读取路径（[`prims_simple.h:681-689`](../src/device/prims_simple.h#L681)） |
| `isNetOffload` | 网络设备卸载模式：工作线程跳过本地搬数据循环（[`prims_simple.h:230`](../src/device/prims_simple.h#L230) 的 `&& !isNetOffload`） |

`Direct` 在 LL / LL128 上被"退化处理"：`PrimitivesWithoutDirect`（[`primitives.h:133-165`](../src/device/primitives.h#L133)）把 `directSend` 直接转发成 `send`、`directRecv` 转发成 `recv`。也就是说 **Direct 只有 Simple 协议才真正省拷贝**，LL/LL128 只是保持接口一致。

### ⑤ 收益

* 编译期多态：三份实现各自展开各自的常量循环，运行时零分支、零虚表。
* 参数即常量：`Unroll`、`SlicePerChunk`、`MaxRecv` 都是 `constexpr`，编译器能把 `for (i = 1; i < MaxSend && i < fan.nsend(); i++)` 这类循环完全展开并在 `MaxSend==1` 时整段删掉（源码里大量 `coverity[dead_error_line]` 注释就是在标记"这段对某些实例化不可达，是预期的"）。
* 代码复用：`runRing` 一份代码同时服务三种协议（[`all_reduce.h:887-913`](../src/device/all_reduce.h#L887)）。

### ⑥ 面试考点

**Q1：为什么 NCCL 不用运行时 `switch(proto)` 而要写成模板？**
A：协议差异（slot 粒度、flag 布局、向量化宽度）都是**编译期常量**。写成模板后，`SlicePerChunk`、`Unroll`、`MaxRecv` 全部参与常量折叠与循环展开，热路径上没有分支预测失败，也没有把协议常量放寄存器再比较的开销；同时让"发 1 个目标"和"发 3 个目标"生成两份完全不同的代码，而不是一份带运行时上界的通用代码。

**Q2：`FanSymmetric` 相比 `FanAsymmetric` 优化了什么？**
A：它只存一个 `int n` 而不是 `nr/ns` 两个，[`primitives.h:93`](../src/device/primitives.h#L93) 的注释写明："省一个 32 位寄存器，更重要的是展开循环时用更少的谓词寄存器"。ring 天然收发对称，所以 `runRing` 用 `FanSymmetric<1>`。

**Q3：`Direct=1` 就一定走零拷贝吗？**
A：不是。`Direct` 只是"允许"。真正的判定在 [`prims_simple.h:545-567`](../src/device/prims_simple.h#L545)：`Direct && ipcRegFlag` 且连接 flags 含 `NCCL_P2P_WRITE`/`NCCL_P2P_READ` 才会置 `DirectWrite`/`DirectRead` 标志位。用户 buffer 没注册（`collWork->regUsed == 0`）时，仍然走 `connEltsFifo` 中转。

**Q4：`PrimitivesWithoutDirect` 存在的意义？**
A：LL/LL128 的数据单元本身带 flag，无法直接写到对端用户 buffer（对端用户 buffer 里没有 flag 位），所以直连对它们没有意义。为了让 `all_reduce.h` 能协议无关地统一调用 `directSend/directRecv...`，用这个 CRTP 辅助类把 direct* 系列"降级转发"到普通 send/recv（[`primitives.h:133-165`](../src/device/primitives.h#L133)）。

**Q5：三种协议的 `calcBytePerStep()` 分别是什么，说明什么？**
A：Simple = `buffSizes[SIMPLE]/NCCL_STEPS`（100% 有效，[`primitives.h:48-50`](../src/device/primitives.h#L48)）；LL = 再 `/2`（50%，[`primitives.h:63-65`](../src/device/primitives.h#L63)）；LL128 = 乘 `NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS` = 15/16（93.75%，[`primitives.h:78-80`](../src/device/primitives.h#L78)）。这直接决定了三者的带宽上限。

---

## 主题 2：构造函数 —— 把 `ncclConnInfo` 的 buffs/head/tail/step 拉进寄存器，并给线程分角色

### ① 解决什么问题

host 端在连接阶段（[`p2p.cc:620-625`](../src/transport/p2p.cc#L620)）已经把对端的显存指针、head/tail 计数器地址写进了 `ncclConnInfo`，但 GPU 上每次从 `ncclShmem.channel.peers[peer]->send[c]` 顺着指针链去取，是多次依赖装载（load-load 串行）。原语对象在 kernel 里会被反复调用（`runRing` 每步一次），如果每次重新解析连接就太慢了。

同时，一个线程块里的线程必须分工：少数线程负责轮询/发布 flag，其余线程全力搬数据。

### ② 一句话本质

**构造函数一次性把 `conn->buffs / head / tail / step / stepSize` 读进对象的寄存器字段，并按 `tid` 给每个线程打上 `RoleWaitRecv / RoleWaitSend / RolePostRecv / RolePostSend` 位掩码；此后所有原语调用只操作寄存器与 shared memory。**

### ③ 代码链路

1. [`prims_simple.h:618-623`](../src/device/prims_simple.h#L618) 构造函数签名与 `stepSize` 初始化。
2. [`prims_simple.h:630`](../src/device/prims_simple.h#L630) 计算 `nworkers`。
3. [`prims_simple.h:655-670`](../src/device/prims_simple.h#L655) 角色分配。
4. [`prims_simple.h:692-698`](../src/device/prims_simple.h#L692) 按角色调 `loadRecvConn` / `loadSendConn`。
5. [`loadRecvConn`](../src/device/prims_simple.h#L517) / [`loadSendConn`](../src/device/prims_simple.h#L571)。
6. [`setDataPtrs`](../src/device/prims_simple.h#L792) 把用户 buffer 指针与 direct buffer 指针落到 shared memory。

### ④ 关键代码逐行解读

先看角色分配与连接加载（[`prims_simple.h:627-698`](../src/device/prims_simple.h#L627) 中摘取）：

```cpp
  if (mode == primsModeDefault) {
    // 与 sendPeers/recvPeers 中的各个 rank 建立连接
    // 对于发送操作，需要额外一个 线程束，以便让 threadfence 与数据拷贝相互重叠
    this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

    int nrecv = 0, nsend = 0;
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend);

    constexpr int ThreadPerSync =
      // NVLS 的扇出可能超过 8。这种情况下需要增大分组的规模
      MaxSend >= 16 || MaxRecv >= 16 ? 32 :
      MaxSend >= 8 || MaxRecv >= 8 ? 16 :
      8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
    static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

    assert(2 * (nrecv + nsend) <= nthreads); // Ensure no thread is assigned more than one role.
    if (tid < nrecv) {
      flags |= RoleWaitRecv;
      index = tid;
    }
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
```

* `nworkers = nthreads - WARP_SIZE`（当 `MaxSend>0 && nthreads >= 96`，`NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE = 3*WARP_SIZE`，[`device.h:106`](../src/include/device.h#L106)）：**从线程块尾部割出一个 warp 专职做同步**，其余线程搬数据。这样 `fence/threadfence` 与 `reduceCopy` 能重叠。
* `ThreadPerSync`：扇出越大，需要越多线程来"一人盯一个对端"。`MaxSend/MaxRecv <= 8` 时 8 个线程（1/4 warp）就够覆盖 4 种角色 × 若干对端。
* 角色布局是**两头夹中间**：`[0, nrecv)` 做 `WaitRecv`，`[nrecv, nrecv+nsend)` 做 `WaitSend`，**尾部** `nsend` 个线程做 `PostSend`，再往前 `nrecv` 个做 `PostRecv`。中间剩下的就是纯工作线程。这样"同步线程"集中在 tid 头部和尾部，工作线程连续成块，便于 `subBarrier()` 只同步 `[0, nworkers)`。
* `assert(2*(nrecv+nsend) <= nthreads)` 保证一个线程最多一个角色。

再看 `loadRecvConn`（[`prims_simple.h:517-569`](../src/device/prims_simple.h#L517)）：

```cpp
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
      if (Direct) { /* ... 见主题 6 ... */ }
    }
  }
```

* `step = conn->step; step = roundUp(step, SlicePerChunk*StepPerSlice);`：从上一次 kernel 结束时的进度继续，并对齐到"整 chunk"边界。
* `RolePostRecv` 线程：`connStepPtr = conn->head`，并**立刻把对齐多出来的信用归还给对端**（`*connStepPtr = step`）。这一步很关键，否则对端会因为少收到信用而死锁。
* `RoleWaitRecv` 线程：`connStepPtr = conn->tail`，并把**当前值 prefetch 进 `connStepCache`** —— 这就是主题 3 里 cache 的初值。
* `connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE]`：**每个协议有自己的一套 buffer**，Simple 只认 `NCCL_PROTO_SIMPLE` 那一块。
* `connStepSize = conn->stepSize / sizeof(T)`：把字节步长换算成元素个数，后续 `connEltsFifo + (step % NCCL_STEPS) * connStepSize` 直接用元素寻址。

`loadSendConn`（[`prims_simple.h:571-615`](../src/device/prims_simple.h#L571)）与之对称，只是 `RolePostSend` 拿 `conn->tail`、`RoleWaitSend` 拿 `conn->head`：

```cpp
    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    }
    if (flags & RoleWaitSend) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn;
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize / sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
```

于是全库统一的语义是（与 host 端映射一致，见 [`p2p.cc:621-622`](../src/transport/p2p.cc#L621) 与 [`p2p.cc:652-653`](../src/transport/p2p.cc#L652)）：

| 计数器 | 谁写 | 谁读 | 含义 |
| --- | --- | --- | --- |
| `tail` | 发送方（`RolePostSend`，写的是**对端**的 recv tail） | 接收方（`RoleWaitRecv`） | "数据已就绪" |
| `head` | 接收方（`RolePostRecv`，写的是**对端**的 send head） | 发送方（`RoleWaitSend`） | "槽位已归还（信用）" |

### ⑤ 收益

* 连接解析只在构造时做一次，热路径上 `waitPeer` 只读一个寄存器里的指针 `connStepPtr`。
* 角色位掩码让"是否要 spin"、"是否要 fence"变成一次 `flags & RoleXxx` 的整数测试，编译器对常量 `Recv/Send` 模板参数还能把它完全常量折叠掉（例如 `Recv=0` 时 `flags & (Recv * RoleWaitRecv)` 恒为 0）。
* `2*(nrecv+nsend) <= nthreads` 的约束保证同步线程最多占一半，工作线程数量下界明确。

### ⑥ 面试考点

**Q1：为什么构造时要 `step = roundUp(step, SlicePerChunk*StepPerSlice)`，并且 `RolePostRecv` 线程立刻 `*connStepPtr = step`？**
A：`step` 必须对齐到 chunk 边界，否则一个 chunk 会横跨 chunk 边界导致后续 `SlicePerChunk` 次 `waitPeer` 的步进逻辑错乱。对齐时 `step` 被**抬高**了，意味着接收方"假装"多消费了几个 slot；如果不把这部分信用立刻归还给对端（`*head = step`），对端的 `RoleWaitSend` 会以为这些 slot 还没被消费而永久 spin。

**Q2：`connStepSize` 为什么用 `conn->stepSize / sizeof(T)` 而不是直接用字节？**
A：后续 `reduceCopy` 和 `connEltsFifo + (step % NCCL_STEPS) * connStepSize` 都是按 `T*` 做指针算术的，用元素个数可以避免每次乘除 `sizeof(T)`。`conn->stepSize` 由 host 端设为 `buffSizes[SIMPLE]/NCCL_STEPS`（[`p2p.cc:610`](../src/transport/p2p.cc#L610)）。

**Q3：为什么 `nworkers` 要减掉一个 warp？**
A：让 `fence_acq_rel_sys()`（`postPeer` 里发布 tail 前的系统级内存栅栏）与数据拷贝在不同线程上并行执行，把 fence 的延迟藏在拷贝后面。条件 `nthreads >= 3*WARP_SIZE` 说明线程少时（<=64）再割一个 warp 就不划算了（[`prims_simple.h:630`](../src/device/prims_simple.h#L630)）。

---

## 主题 3：环形 FIFO + `NCCL_STEPS` 个 slot 的流控

### ① 解决什么问题

两个 GPU 之间是一块固定大小的共享 buffer（`conn->buffs[NCCL_PROTO_SIMPLE]`），必须解决经典的生产者-消费者问题：发送方不能覆盖接收方还没读走的槽位，接收方不能读发送方还没写好的槽位。而且这个过程**不能有锁、不能 sleep**，只能靠 GPU 线程 spin 在跨卡内存上。

### ② 一句话本质

**把共享 buffer 切成 `NCCL_STEPS = 8` 个 slot，用两个单调递增的 64 位计数器 `tail`（生产者→消费者，"数据就绪"）与 `head`（消费者→生产者，"信用归还"）做跨卡流控；`step % NCCL_STEPS` 就是当前 slot 下标。**

### ③ 代码链路

1. slot 数量常量 [`device.h:36`](../src/include/device.h#L36) `NCCL_STEPS 8`。
2. 生产者等信用：[`prims_simple.h:142-157`](../src/device/prims_simple.h#L142) `waitPeer` 的 spin 循环。
3. 生产者算 slot 地址：[`prims_simple.h:159-203`](../src/device/prims_simple.h#L159)。
4. 生产者发布：[`prims_simple.h:206-215`](../src/device/prims_simple.h#L206) `postPeer`。
5. 消费者归还信用：同样是 `postPeer`（`RolePostRecv` 写 `conn->head`）。
6. 计数器的 host 端映射：[`p2p.cc:621-622`](../src/transport/p2p.cc#L621)、[`p2p.cc:652-653`](../src/transport/p2p.cc#L652)。

### ④ 关键代码逐行解读

[`prims_simple.h:142-215`](../src/device/prims_simple.h#L142)：

```cpp
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
      }
    }

    if (flags & (Recv * RoleWaitRecv | Send * RoleWaitSend)) {
      if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
        connFifo[step % NCCL_STEPS].size = nelts * sizeof(T);

      void** ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst) : (ncclShmem.groups[group].srcs + Src);
      /* ... NetRegMode / ConnFifo / Direct 分支，见主题 6 ... */
      } else {
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
```

逐段解释：

* **`isSendNotRecv`**：当 `Recv` 和 `Send` 同时为 1（`recvReduceSend` 这种"既收又发"的原语）时，同一个线程可能既是 WaitRecv 又是 WaitSend；这里用 `flags & RoleWaitSend` 区分"本次调用我扮演哪一边"。
* **等待条件**：
  * 接收（`isSendNotRecv == false`）：`while (connStepCache < step + StepPerSlice)` —— 等**对端 tail** 推进到"我这一步要读的 slot 已经写完"。
  * 发送（`isSendNotRecv == true`）：`while (connStepCache + NCCL_STEPS < step + StepPerSlice)`，等价于 `connStepCache >= step + StepPerSlice - NCCL_STEPS` —— 等**对端 head**（信用）追上来。多出来的 `+NCCL_STEPS` 就是"**我最多可以比对端领先 8 个 slot**"，这是流水线的深度，也是 Simple 能打满带宽的关键：发送方可以连续写 8 个 slot 而不需要等任何一次接收确认。
* **`checkAbort`**（[`primitives.h:167-177`](../src/device/primitives.h#L167)）：每 spin 满 `NCCL_SPINS_BEFORE_CHECK_ABORT = 10000` 次才去读一次 `ncclShmem.comm.abortFlag`，避免把 abort 检查放进最内层循环。这是"**spin 循环里不要碰全局内存**"的经典优化。
* **`step % NCCL_STEPS`**：slot 下标由逻辑步号取模得到，天然形成环形复用。`connFifo[step % NCCL_STEPS].size = nelts * sizeof(T)` 是把本 slot 的实际字节数告诉 proxy（网络路径用）。
* **`ptrs[index] = ...`**：Wait 线程算出的 slot 地址写进 **shared memory**（`ncclShmem.groups[group].srcs/dsts`，结构见 [`common.h:42-54`](../src/device/common.h#L42)），随后 `subBarrier()` 之后所有工作线程都从 shared memory 拿到指针。这是"**少数线程做同步，多数线程只从 shared memory 读指针**"的关键设计。
* **`step += StepPerSlice`**：一次 `waitPeer` 推进 `StepPerSlice` 个 slot。
* **`postPeer`**：同样 `step += StepPerSlice`；若是发送侧且真的写了数据（`dataStored`），先 `fence_acq_rel_sys()` 再 `st_relaxed_sys_global(connStepPtr, step)`。**fence 必须在写 tail 之前**，保证数据先于"数据就绪"信号对对端可见。

`loadStepValue` 为什么用 volatile 而不是 acquire（[`prims_simple.h:124-140`](../src/device/prims_simple.h#L124)）：

```cpp
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
    // 从而绕过 L1 缓存、不会读到陈旧数据。
    return ld_volatile_global(ptr);
  }
```

### ⑤ 收益（定量）

`connStepCache` 的作用（`prims_simple.h:80` 声明为 `uint64_t connStepCache;`）：

* **减少跨卡 volatile 读**。对 `connStepPtr` 的每次 volatile 读都是一次穿越 NVLink/PCIe 的往返（不能被 L1 缓存，几百 ns 量级）。把它缓存在寄存器里后：
  * spin 循环内的比较用的是寄存器值，不需要每次都重新装载；
  * 更重要：一旦对端一次推进了多个 slot（例如一次归还 8 个信用），接下来**多次 `waitPeer` 完全不需要再读远程内存**，因为 `connStepCache` 已经足够大。
* 极端情况下（8 个 slot 全空闲），一整个 chunk（`SlicePerChunk*StepPerSlice = 4` 个 step）的 4 次 `waitPeer` 只需 1 次远程读。

Simple 协议一个 slot 的同步开销：

* 缓冲区默认 4 MiB（[`init.cc:827`](../src/init.cc#L827) `DEFAULT_BUFFSIZE (1 << 22)`），`NCCL_STEPS = 8` → **每 slot 512 KiB**（`stepSize = buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)`，[`prims_simple.h:623`](../src/device/prims_simple.h#L623)）。
* 一个 slot 只付：1 次 8 字节 `tail` 写 + 1 次 8 字节 `head` 轮询。
* 数据 : 同步 = 512 KiB : 8 B = **65536 : 1**，即同步字节占比约 **0.0015%**。对比 LL 的 50%、LL128 的 6.25%（详见第 10 篇）。
* 但**字节数不是主要成本，延迟才是**：每一次"写 tail → 对端看见 → 对端回 head"是一次跨卡往返。Simple 靠 8 个 slot 的流水线深度把这 ~1 µs 的往返摊到 8×512 KiB = 4 MiB 的在途数据上。调优模型里 Simple 的 NVLink ring 硬件延迟是 3.4 µs，LL 是 0.6 µs（[`tuning.cc:176`](../src/graph/tuning.cc#L176)）—— 差的就是这"先等 tail 再读数据"的两段式依赖。

### ⑥ 面试考点

**Q1：`waitPeer` 里为什么发送方是 `connStepCache + NCCL_STEPS < step + StepPerSlice` 而接收方是 `connStepCache < step + StepPerSlice`？**
A：两者读的计数器语义不同。接收方读 `tail`（数据就绪），要求 `tail >= step+StepPerSlice`。发送方读 `head`（信用），条件是"我即将写入的 slot 在 `NCCL_STEPS` 步之前已经被消费过"，即 `head + NCCL_STEPS >= step + StepPerSlice`。`NCCL_STEPS` 就是允许的在途 slot 数，直接决定流水线深度。

**Q2：为什么必须 `fence_acq_rel_sys()` 之后再写 tail？**
A：`st_relaxed_sys_global` 是 relaxed 语义，编译器/硬件可以把它排到数据写之前。如果对端先看到 tail 前进、再看到数据，就会读到脏数据。所以 `postPeer` 在 `dataStored == true` 时先插一条系统级 acq_rel 栅栏（[`prims_simple.h:210-213`](../src/device/prims_simple.h#L210)），保证"数据写 → 全局可见 → tail 写"的次序。

**Q3：`connStepCache` 存的是什么？为什么不能直接每次都读 `connStepPtr`？**
A：存的是上一次 volatile 读到的对端计数器值。直接每次读会 (a) 在 spin 循环里产生大量跨卡内存事务，把 NVLink 带宽浪费在轮询上；(b) 这些读还会污染/占用 LSU 与 memory pipeline，拖慢真正的数据搬运用到的 load/store。缓存到寄存器后，只有 cache 值不够时才发一次真正的远程读。

**Q4：`checkAbort` 为什么要攒够 10000 次 spin 才检查？**
A：abort 标志在全局内存（`ncclShmem.comm.abortFlag`）。如果每次循环都读，spin 循环会退化成"每轮一次全局内存访问"，既慢又和真正的计数器轮询抢带宽。攒够 `NCCL_SPINS_BEFORE_CHECK_ABORT = 10000` 次（[`primitives.h:28`](../src/device/primitives.h#L28)、[`primitives.h:167-177`](../src/device/primitives.h#L167)）再查一次，把 abort 检查的摊薄成本降到可忽略。

**Q5：`NCCL_STEPS` 调大或调小会怎样？**
A：调大 → 流水线更深、更能容忍跨卡往返延迟，但每个 slot 更小（同 buffer 下），单次同步的相对开销上升，且 buffer 占用不变；调小 → slot 更大、单次传输更连续，但发送方更容易在信用上阻塞。NCCL 取 8 是两者折中，同时 `NCCL_LL_CLEAN_MASK` 之类的常量还要求它是 8 的倍数（[`device.h:118`](../src/include/device.h#L118) 的 `static_assert`）。

---

### 环形 FIFO slot 与 head/tail 流控时序

```mermaid
sequenceDiagram
    autonumber
    participant WS as 发送方 WaitSend 线程
    participant WK as 发送方工作线程<br/>(tid < nworkers)
    participant PS as 发送方 PostSend 线程
    participant FIFO as 对端共享 buffer<br/>8 个 slot (NCCL_STEPS=8)
    participant HEAD as 接收方写回的 head<br/>(信用, 发送方本地可见)
    participant TAIL as 对端 recv 的 tail<br/>(数据就绪)
    participant WR as 接收方 WaitRecv 线程
    participant RK as 接收方工作线程
    participant PR as 接收方 PostRecv 线程

    Note over WS,PR: step 是每侧单调递增的 64 位逻辑时钟; slot = step % 8

    WS->>HEAD: ld.volatile.global 读 head
    HEAD-->>WS: head = h (缓存在 connStepCache)
    loop 信用不足
        WS->>WS: while (h + 8 < step + StepPerSlice) spin<br/>cache 命中时不再发远程读
    end
    WS->>WS: srcs/dsts[index] = connEltsFifo + (step%8)*connStepSize
    WS->>FIFO: (subBarrier) 把指针发布到 shared memory
    WK->>FIFO: reduceCopy 把数据写进 slot (step%8)
    WK->>PS: (barrier) 写完成
    PS->>PS: fence_acq_rel_sys()
    PS->>TAIL: st.relaxed.sys tail = step + StepPerSlice
    Note over PS,TAIL: 数据先于"就绪"信号可见

    WR->>TAIL: ld.volatile.global 读 tail
    TAIL-->>WR: tail = t (缓存在 connStepCache)
    loop 数据未就绪
        WR->>WR: while (t < step + StepPerSlice) spin
    end
    WR->>FIFO: srcs[index] = connEltsFifo + (step%8)*connStepSize
    RK->>FIFO: reduceCopy 读 slot 并做规约
    RK->>PR: (barrier) 读完成
    PR->>HEAD: st.relaxed.sys head = step + StepPerSlice
    Note over PR,HEAD: 归还槽位信用, 允许发送方领先 8 个 slot
```

---

## 主题 4：`genericOp` —— 一个模板函数统摄全部原语

### ① 解决什么问题

ring / tree 算法需要非常多的搬运形态：`send`、`recv`、`recvReduceSend`、`recvReduceCopySend`、`directSend`、`directRecvCopy`…… 如果每个都写一份，代码量爆炸且每份都要单独维护 FIFO 流控、barrier、fence 的正确性。

### ② 一句话本质

**`genericOp<DirectRecv1, DirectSend1, Recv, Send, SrcBuf, DstBuf>` 用 6 个 `int`/枚举模板参数把"是否收、是否发、源是 Input 还是 Output（还是没有）、目标是 Input/Output（还是没有）、是否直连"编码进类型；所有 `if` 都是 `constexpr`，编译期就把不适用的分支整段裁掉，于是 30 多个原语全部退化成同一个函数体的不同实例化。**

### ③ 代码链路

1. 模板函数本体 —— [`prims_simple.h:217-346`](../src/device/prims_simple.h#L217)。
2. 30+ 个薄封装 —— [`prims_simple.h:899-1002`](../src/device/prims_simple.h#L899)。
3. 实际的多源规约/多目标写出 —— `reduceCopy`，[`common_kernel.h:206-261`](../src/device/common_kernel.h#L206)。
4. 调用方 —— [`all_reduce.h:97`](../src/device/all_reduce.h#L97)、[`all_reduce.h:108`](../src/device/all_reduce.h#L108)、[`all_reduce.h:123`](../src/device/all_reduce.h#L123)、[`all_reduce.h:132`](../src/device/all_reduce.h#L132)、[`all_reduce.h:143`](../src/device/all_reduce.h#L143)。

### ④ 关键代码逐行解读

先看薄封装层（[`prims_simple.h:899-984`](../src/device/prims_simple.h#L899) 摘录）：

```cpp
  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
                                                                 bool postOp = false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
```

六个模板参数的编码规则（[`prims_simple.h:217-222`](../src/device/prims_simple.h#L217)）：

```cpp
  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;
```

* `DirectRecv = 1 && Direct && DirectRecv1`：如果类模板参数 `Direct == 0`，无论调用哪个 `direct*` 版本，编译期都会被压成 0 —— 一行代码实现了"`Direct=0` 时 direct* 自动降级"。
* `SrcBuf == -1` → `Src = 0` → 后续 `if (Src)` 被裁掉。这就是 `send(inpIx, eltN)` 与 `recv(outIx, eltN)` 共用一份代码却行为完全不同的原因。

再看核心循环（[`prims_simple.h:262-326`](../src/device/prims_simple.h#L262)）：

```cpp
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
        /* ... NetDeviceUnpack ... */
        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // 直连接收最多只能有一个。由于 srcs[0] == dstPtr+偏移(源和目标是同一块地址)，可以省掉一次拷贝
          if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/ 0>(
              tid, nworkers, /*redArg*/ 0, /*postOp*/ false, 1, ncclShmem.groups[group].srcs, fan.nsend(),
              ncclShmem.groups[group].dsts + 1, workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // 用于 CollNet 广播场景下执行空发送
          reduceCopy</*...*/>(/*...*/);
        } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 : 1;
          if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
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
          // 当对网络对端调用 prims.directSend 时会走到这里
          workSize = 0;
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < workSize);
        offset += sliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
```

逐段解释：

* `tid == 0` 负责把用户 buffer 指针写进 shared memory 的 `srcs[0]`/`dsts[0]`。**只有 0 号线程写**，其余线程通过 `subBarrier()` 后可见，省掉 30+ 个线程重复计算同一地址。
* `waitPeer(...)` → `subBarrier()`：Wait 线程把 slot 地址写进 `srcs/dsts`，`subBarrier()` **只同步 `[0, nworkers)` 的工作线程**（[`prims_simple.h:94-100`](../src/device/prims_simple.h#L94)），同步线程不参与，从而不被拖慢。
* `reduceCopy` 的模板参数是最精妙的地方：
  * `MinSrcs = Recv + Src`，`MaxSrcs = Recv * MaxRecv + Src`：对 `recvReduceSend`（Recv=1, Src=1, MaxRecv=1）就是 2 个源（对端 slot + 本地 input）；对 `send`（Recv=0, Src=1）就是 1 个源；对 `recvCopySend`（Recv=1, Src=0）也是 1 个源。
  * `MinDsts = Send + Dst`，`MaxDsts = Send * MaxSend + Dst`：同理。
  * 实际源/目的个数 `Recv * fan.nrecv() + Src` / `Send * fan.nsend() + Dst` 是运行时的，但**上下界是编译期的**，`reduceCopyPacks` 因此能用 `MinSrcs/MaxSrcs` 做完全展开（[`common_kernel.h:38-48`](../src/device/common_kernel.h#L38)）。
  * `PreOpSrcs = SrcBuf != Input ? 0 : 1`：只有源是**用户 input** 时才需要 `applyPreOp`（如 `ncclAvg` 之外的预乘），从 Output 读数时不需要。
* `postPeer<Recv, Send>(0 < workSize)`：`workSize == 0` 时不插 fence，省一次系统级栅栏。
* `barrier()` 之后紧跟 `postPeer`，注释 `// This barrier has a counterpart in following loop` 指第二个循环（[`prims_simple.h:334-345`](../src/device/prims_simple.h#L334)）里的 `barrier()` —— 两个循环必须成对出现，否则线程会在不同迭代里错配。

为什么把循环拆成两个（[`prims_simple.h:230-253`](../src/device/prims_simple.h#L230) 的注释）：

```cpp
    if (tid < nworkers && offset < nelem && !isNetOffload) {
      /* 这个循环专供“工作线程 + 非空 slice”使用。非工作线程以及空 slice
       * 由紧随本 if 块之后的那个循环来处理。
       *
       * 为什么要把一个循环拆成两个？为了把两个分支判断移出关键路径。
       *   原实现：perf_orig = 2 * numslices  (每个 slice 都要判断 2 次)
       *   新实现：perf_new  = 2 + numslices  (2 次判断只在循环外做一次)
       */
```

### ⑤ 收益

* 一份流控/barrier/fence 逻辑服务 30+ 个原语，正确性只需维护一处。
* 所有分支在编译期裁剪：`send` 的实例里根本没有"读对端数据"的代码，也没有 `applyReduce` 调用。
* 分支数从 `2 * numslices` 降到 `2 + numslices`（源码注释给出的量化），`SlicePerChunk > 2` 时严格更优。
* `Unroll`、`MinSrcs/MaxSrcs` 等常量让 `reduceCopyPacks` 能生成完全展开、无循环的搬运代码。

### ⑥ 面试考点

**Q1：`genericOp` 的模板参数是怎么在编译期裁掉分支的？**
A：全部是 `int` 模板参数且立即转成 `constexpr`（[`prims_simple.h:219-222`](../src/device/prims_simple.h#L219)）。`if (Src)`、`if (flags & (Recv * RoleWaitRecv))` 在 `Recv=0`/`Src=0` 时恒为常量假，编译器直接删除整段。甚至 `Recv * MaxRecv + Src` 这种算术也是折叠成常量的，所以 `reduceCopy` 的 `MinSrcs/MaxSrcs` 是编译期常量。

**Q2：`subBarrier()` 和 `barrier()` 的区别？为什么这里两个都要？**
A：`barrier()` 同步全部 `nthreads`（含同步线程，[`prims_simple.h:87-93`](../src/device/prims_simple.h#L87)），`subBarrier()` 只同步 `[0, nworkers)` 的工作线程（[`prims_simple.h:94-100`](../src/device/prims_simple.h#L94)）。`waitPeer` 之后只需让工作线程看到 shared memory 里的指针 → 用 `subBarrier()`，让同步线程继续去 spin，不被工作线程拖住。`postPeer` 之前必须用全 `barrier()`，因为要确保**所有工作线程都写完数据**才能发布 tail。

**Q3：为什么第二个循环（空 slice 路径）里没有 `reduceCopy`？**
A：源码注释（[`prims_simple.h:336`](../src/device/prims_simple.h#L336)）写得很清楚："Only workers could have Wait roles so we know the slice must be empty" —— 走到这个循环的要么是同步线程（没有 Wait 角色），要么是所有 slice 都为空的工作线程。但**流控不能跳过**：`step` 必须继续推进、`head`/`tail` 必须继续发布，否则对端会永久等待。

**Q4：`PreOpSrcs = SrcBuf != Input ? 0 : 1` 是什么意思？**
A：`applyPreOp`（如 `PreMulSum` 的预乘）只对来自**用户 input buffer** 的数据做一次；如果源是 Output（例如 `sendFromOutput`）或对端 slot，数据已经被预处理过或不属于本地输入，再做一次就重复了。

**Q5：`reduceCopy` 的 `MinSrcs/MaxSrcs` 为什么是编译期常量而 `nSrcs` 是运行时？**
A：`nSrcs` 取决于运行时 `fan.nrecv()`（tree 上不同节点子节点数不同），但**上下界**由模板参数决定。`reduceCopyPacks` 用 `MaxSrcs` 静态展开加载循环、用 `MinSrcs` 做 `static_assert`（[`common_kernel.h:211-212`](../src/device/common_kernel.h#L211)），运行时只承担一个 `if (s < nSrcs)` 的谓词，既保住展开又支持运行时变化。

---

## 主题 5：slice / chunk 分层 —— 让"收对端"和"发给下一跳"流水线重叠

### ① 解决什么问题

ring AllReduce 的第 k 步要做：`从 prev 收一块 → 与本地规约 → 发给 next`。如果严格串行，每一步都要先付满一次跨卡接收延迟，才能开始发送，链路有一半时间空闲。

### ② 一句话本质

**把一个 chunk 再切成 `SlicePerChunk` 个 slice，每个 slice 只跨 `StepPerSlice` 个 FIFO slot；于是第 i 个 slice 在发送的同时，第 i+1 个 slice 已经在接收 —— 收发在同一个 chunk 内部流水重叠。**

### ③ 代码链路

1. 常量来源：[`collectives.h:27-28`](../src/include/collectives.h#L27) `ALLREDUCE_SLICESTEPS (NCCL_STEPS/4) = 2`、`ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2) = 4`。
2. 实例化：[`all_reduce.h:352`](../src/device/all_reduce.h#L352) `ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>` = `ProtoSimple<2, 2>`。
3. sliceSize 计算：[`prims_simple.h:225-226`](../src/device/prims_simple.h#L225)。
4. 循环：[`prims_simple.h:262-326`](../src/device/prims_simple.h#L262)。
5. host 侧 chunk 大小：[`enqueue.cc:2294-2299`](../src/enqueue.cc#L2294)。

### ④ 关键代码逐行解读

[`prims_simple.h:224-229`](../src/device/prims_simple.h#L224)：

```cpp
    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize * StepPerSlice;
    sliceSize = max(divUp(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
    int slice = 0;
    int offset = 0;
```

* 第一项 `stepSize * StepPerSlice`：一个 slice 最多跨 `StepPerSlice` 个 slot（Simple ring 是 2）。
* 第二项是**下界保护**：`divUp(nelem, 16*SlicePerChunk)*16` 保证即使 `nelem` 很小，也能把数据均分到 `SlicePerChunk` 片里（并且 16 元素对齐以保住向量化）；`sliceSize/32` 则是"至少要有 1/32 个 slot 那么大"，避免为了切太细而产生大量极小传输。
* 两者取 `max`：**既不切得比 slot 还大（否则无法流水），也不切得比 16 元素还小（否则失去向量化收益）**。

循环体里 `sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;`（[`prims_simple.h:263`](../src/device/prims_simple.h#L263)）处理最后一片的尾数。

host 侧对应的粒度（[`enqueue.cc:2294-2299`](../src/enqueue.cc#L2294)）：

```cpp
  int stepSize = comm->buffSizes[info->protocol] / NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize * chunkSteps;
  if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
```

* 只有 `SIMPLE + RING` 才启用 chunk/slice 分层（tree 与 LL/LL128 都是 `1/1`）。
* `chunkSize = stepSize * chunkSteps = 512 KiB × 4 = 2 MiB`；`sliceSize = stepSize * sliceSteps = 1 MiB`。
* LL 要 `/2`、LL128 要 `*15/16`，又一次印证了"协议决定有效载荷比例"。

对应到 `runRing`（[`all_reduce.h:103-109`](../src/device/all_reduce.h#L103)）：

```cpp
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }
```

一次 `directRecvReduceDirectSend` 内部就把 2 MiB 的 chunk 拆成 2 个 1 MiB 的 slice，每个 slice 跨 2 个 slot。第一个 slice 在 `postPeer` 发布 tail 后立即进入第二个 slice 的 `waitPeer`，而对端此时正在消费第一个 slice —— **收发重叠**。

### ⑤ 收益

* 流水线深度从"1 个 chunk"提升到"SlicePerChunk 个 slice"，链路空闲时间从 ~50% 降到 ~1/SlicePerChunk。
* `SlicePerChunk * StepPerSlice = 2 * 2 = 4 = ALLREDUCE_CHUNKSTEPS`，正好用满 chunkSteps 个 slot，不浪费 buffer。
* 与主题 3 的 "最多领先 NCCL_STEPS 个 slot" 配合：ring 上一共可以在途 `NCCL_STEPS / StepPerSlice = 4` 个 chunk，跨卡往返延迟被完全掩盖。

### ⑥ 面试考点

**Q1：为什么要分 slice，直接把一个 chunk 当整体收发不行吗？**
A：不行。整体收发意味着"收完整个 chunk 才开始发"，一次跨卡往返（~1 µs）内链路有一半时间空闲。切成 slice 后，第 i 片发出的同时第 i+1 片在收，把串行变成流水，带宽利用率显著提升。

**Q2：`SlicePerChunk` 和 `StepPerSlice` 的乘积为什么要等于 `chunkSteps`？**
A：`chunkSteps` 是 host 侧给一个 chunk 分配的 slot 数（[`enqueue.cc:2295`](../src/enqueue.cc#L2295)）。`SlicePerChunk * StepPerSlice` 是 device 侧一次 `genericOp` 消耗的 slot 数。两者必须相等，否则 `step` 的推进与 host 侧对 FIFO 容量的假设不一致，会导致提前覆盖未消费的 slot。

**Q3：`sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32)` 的两个下界分别防什么？**
A：第一个防"nelem 太小导致只有第一片有数据、其余片空转"（保证均分且 16 元素对齐以保住 128-bit 向量化）；第二个防"slot 太大而 nelem 很小时 sliceSize 被压得过小，产生大量极小传输"。

**Q4：LL / LL128 为什么不需要 slice 分层？**
A：它们的 `sliceSteps/chunkSteps` 恒为 1（[`enqueue.cc:2295-2296`](../src/enqueue.cc#L2295)）。LL/LL128 本身靠 in-band flag 把延迟压到极低，一次传输的粒度就是用户数据本身，再用 slot 切分反而引入不必要的步进管理；而且它们的正确性依赖 flag 与数据的严格配对，切分逻辑会更复杂。

---

## 主题 6：Direct 模式 —— 注册过内存时直写对端用户 buffer

### ① 解决什么问题

默认情况下，一次 `recvReduceSend` 的数据路径是：`对端 slot(FIFO) → 本地寄存器 → 规约 → 本地 slot(FIFO) → 对端读取`，即数据要"落地"到中间 buffer 两次。如果用户 buffer 已经通过 CUDA IPC / cuMem 注册并交换了远端地址，完全可以让发送方**直接写到接收方的用户 output buffer**，省掉一次中转。

### ② 一句话本质

**`Direct=1` 且 `regUsed != 0` 且连接支持 P2P 时，通过 `conn->ptrExchange` 与对端交换用户 buffer 的远端地址，存到 `directBuff`；此后 `waitPeer` 把 `srcs/dsts[index]` 直接指向 `directBuff + offset`，数据在一次 `reduceCopy` 里从"对端输入"直接落到"对端输出"。**

### ③ 代码链路

1. 能力判定：[`prims_simple.h:545-567`](../src/device/prims_simple.h#L545)（recv）、[`prims_simple.h:592-613`](../src/device/prims_simple.h#L592)（send）。
2. 地址交换：`setDataPtrs`，[`prims_simple.h:792-890`](../src/device/prims_simple.h#L792)。
3. 使用：`waitPeer` 的 Direct 分支，[`prims_simple.h:177-193`](../src/device/prims_simple.h#L177)。
4. 特例优化：`genericOp` 里 `srcs[0] == dsts[0]` 时跳过拷贝，[`prims_simple.h:283-293`](../src/device/prims_simple.h#L283)。
5. 收尾等待：析构函数里 DirectRead 的排空，[`prims_simple.h:779-789`](../src/device/prims_simple.h#L779)。

### ④ 关键代码逐行解读

[`prims_simple.h:545-567`](../src/device/prims_simple.h#L545)（recv 侧能力判定）：

```cpp
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
```

* `NCCL_P2P_WRITE = 0x01`、`NCCL_P2P_READ = 0x02`（[`device.h:130-131`](../src/include/device.h#L130)）。**Write = 我写对端显存；Read = 对端读我的显存**。
* `DirectWrite` 与 `DirectRead` 是**提供方/接受方**的关系：对同一个连接，一侧是 provider（提供自己的 buffer 给对端访问），另一侧是 acceptor（`setDataPtrs` 里的 `recvProvider/sendAcceptor/sendProvider/recvAcceptor`，[`prims_simple.h:800-805`](../src/device/prims_simple.h#L800)）。

`setDataPtrs` 的 provider 一侧（[`prims_simple.h:806-824`](../src/device/prims_simple.h#L806)）：

```cpp
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite);
      // 发送方提供直连缓冲区(供对端拉取)
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead);
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead);
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
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
```

* `ptrExchange` 是一个 **1 槽的握手邮箱**：provider 必须 `while (*slot != nullptr)` 等对端取走旧值，才能写入新值；acceptor 侧（[`prims_simple.h:825-841`](../src/device/prims_simple.h#L825)）spin 等 `ptr != nullptr`，取走后置 `nullptr`。
* `recvbuffRmtAddrs[localPeer]` 是 host 端注册时拿到的"我的 buffer 在对端进程地址空间里的虚拟地址"—— 这就是 CUDA IPC 的价值所在。
* 非 P2P 场景要用 `rankToLocalRank[peer]`（或跨 clique 时直接用 peer rank）来索引，因为共享内存是按 localRank 组织的。

`waitPeer` 里真正用上 `directBuff`（[`prims_simple.h:177-193`](../src/device/prims_simple.h#L177)）：

```cpp
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
        ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * connStepSize;
      }
```

* `DirectRead` 的发送侧是 `nullptr` —— **因为数据不用发，对端自己会来我的 output buffer 读**，本端只需推进流控。
* `DirectWrite` 的接收侧指向 `directBuff + dstIx` —— 因为"接收"这个动作已经由对端直接写我的 output 完成了。

`genericOp` 里还有一个 Direct 特有的捷径（[`prims_simple.h:283-293`](../src/device/prims_simple.h#L283)）：

```cpp
        if (DirectRecv &&
            ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // 直连接收最多只能有一个。由于 srcs[0] == dstPtr+偏移(源和目标是同一块地址)，可以省掉一次拷贝
          if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/ 0>(
              tid, nworkers, /*redArg*/ 0, /*postOp*/ false, 1, ncclShmem.groups[group].srcs, fan.nsend(),
              ncclShmem.groups[group].dsts + 1, workSize);
          }
        }
```

即 `directRecvCopyDirectSend` 场景下源和目的是同一块地址时，**整段拷贝被跳过**（`nDsts` 从 `dsts+1` 开始，把自己排除掉）。

### ⑤ 收益

* 省掉一次中间 buffer 落地：数据路径从"读 FIFO + 写 FIFO"（2 次跨卡 + 2 次本地）变成"读对端用户 buffer + 写对端用户 buffer"，跨卡事务数不变但**本地显存读写带宽压力减半**。
* `srcs[0] == dsts[0]` 的捷径在最理想情况下连本地拷贝都省了。
* 代价：需要用户 buffer 注册（`ncclCommRegister` 或 CUDA 图捕获路径），且要在 `ptrExchange` 上做一次跨卡握手（只在 kernel 启动时一次，不在热路径）。

### ⑥ 面试考点

**Q1：`DirectWrite` 和 `DirectRead` 的区别？**
A：`DirectWrite` = 我有能力**写对端的显存**（我 push）；`DirectRead` = 对端有能力**读我的显存**（对端 pull）。两者由连接 flags `NCCL_P2P_WRITE / NCCL_P2P_READ` 决定（[`prims_simple.h:548-555`](../src/device/prims_simple.h#L548)），并且在同一条连接上必须配成 provider/acceptor 一对。

**Q2：`ptrExchange` 为什么是"单槽邮箱"而不是直接写？**
A：它是一个 `void* volatile` 的单元素握手区。provider 必须等对端把旧指针取走（`while (*slot != nullptr)`）才能写新值，否则会覆盖对端还没读的地址；acceptor 取走后写回 `nullptr` 表示"已消费"（[`prims_simple.h:836`](../src/device/prims_simple.h#L836)）。这只在 kernel 启动时发生一次，代价可忽略。

**Q3：为什么 `DirectRead` 的发送侧要把 `ptrs[index]` 设成 `nullptr`？**
A：因为这种模式下**接收方会主动来读我的 output buffer**，发送方不需要（也不能）再写一份数据。`nullptr` 会让 `genericOp` 走到 [`prims_simple.h:314-319`](../src/device/prims_simple.h#L314) 的 `else` 分支把 `workSize = 0`，从而完全跳过数据搬运，只推进流控。

**Q4：析构函数里 DirectRead 为什么要 `while (*tail > *head)`？**
A：见 [`prims_simple.h:779-789`](../src/device/prims_simple.h#L779)：DirectRead 下我的 output buffer 正被对端读取，kernel 返回后这块 buffer 可能被下一个 kernel 覆盖。必须等到"我已发布的所有 tail 都被对端用 head 确认消费完"。这个检查放在 `barrier()` **之后**，因为 Post 线程与这个检查存在竞争。

---

## 主题 7：memory ordering —— `volatile` / `fence` 放在哪、为什么必要

### ① 解决什么问题

跨 GPU 通信没有硬件缓存一致性协议帮忙（NVLink 上 P2P 显存访问不保证 L1 一致），编译器和 SM 都会重排 load/store。必须精确回答三个问题：数据什么时候对对端可见？"数据就绪"信号什么时候可见？两者的先后如何保证？

### ② 一句话本质

**"数据写 → `fence_acq_rel_sys()` → `st.relaxed.sys` 写 tail" 是发布侧的三段式；"volatile 读 tail → volatile 读数据" 是获取侧的两段式；接收方还完数据后 `st.relaxed.sys` 写 head 归还信用。全链路刻意用 `volatile`（而非 `acquire`）换取速度，代价是要求数据读取端也全程 volatile 以绕过 L1。**

### ③ 代码链路

1. 获取侧 volatile 读计数器：[`prims_simple.h:124-140`](../src/device/prims_simple.h#L124) → `ld_volatile_global`，[`op128.h:349-353`](../src/device/op128.h#L349)。
2. 发布侧 fence：[`prims_simple.h:210-213`](../src/device/prims_simple.h#L210) → `fence_acq_rel_sys`，[`op128.h:400-406`](../src/device/op128.h#L400)。
3. 发布侧 relaxed 写：[`op128.h:385-391`](../src/device/op128.h#L385) `st_relaxed_sys_global`。
4. 数据搬运全程 volatile：[`common_kernel.h:96-98`](../src/device/common_kernel.h#L96)、[`common_kernel.h:116-119`](../src/device/common_kernel.h#L116)、[`common_kernel.h:136-138`](../src/device/common_kernel.h#L136)。
5. NVLS 特例（acquire + multimem）：[`prims_simple.h:126-133`](../src/device/prims_simple.h#L126)。

### ④ 关键代码逐行解读

发布侧（[`prims_simple.h:206-215`](../src/device/prims_simple.h#L206)）：

```cpp
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
```

* `dataStored` 为假（例如 `workSize == 0` 的空发送）时**跳过 fence** —— 没有数据就不需要保序。
* 接收侧（`RolePostRecv`）写 head 时**不需要 fence**：head 是"我读完了"的信号，不携带数据，即使早到也只是让对端提前拿到信用，不会造成数据错误（对端写新数据发生在本端读完之后）。

PTX 实现（[`op128.h:385-406`](../src/device/op128.h#L385)）：

```cpp
__device__ __forceinline__ void st_relaxed_sys_global(uint64_t* ptr, uint64_t val) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(cvta_to_global(ptr)), "l"(val) : "memory");
#else
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(cvta_to_global(ptr)), "l"(val) : "memory");
#endif
}
__device__ __forceinline__ void fence_acq_rel_sys() {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  asm volatile("membar.sys;" ::: "memory");
#endif
}
```

* `.sys` 作用域是关键：跨 GPU 的 P2P 内存需要用 system scope，只保证 GPU 内部可见的 `.gpu` scope 不够。
* Volta 之前（`< sm_70`）没有 `relaxed`/`acq_rel` 语义，退化为 `volatile` / `membar.sys`。

获取侧为什么可以用 volatile（[`common_kernel.h:95-98`](../src/device/common_kernel.h#L95)）：

```cpp
        } else {
          // 使用易变加载，以防 credits 是通过易变（而非获取）语义轮询的。
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(redFn, acc[u]);
        }
```

注释直译过来就是：**因为计数器是用 volatile（而非 acquire）轮询的，数据也必须用 volatile 加载**，否则 volatile 的计数器已经"看见"了新值，而普通 `ld.global` 的数据读可能命中 L1 里的旧副本 —— 这会造成"信号到了、数据没到"的经典 bug。

NVLS 特例（[`prims_simple.h:126-133`](../src/device/prims_simple.h#L126)）用的是 `multimem.ld_reduce.acquire.sys.global.min.u64`，一条指令完成"多播读 + 求最小 + acquire"，用于在 NVLS 场景下从多个对端的计数器中取最小值。

### ⑤ 收益（定量）

* Simple 协议每个 slot 的 fence 次数：**发送侧 1 次**（`dataStored==true` 时），接收侧 0 次。一个 512 KiB 的 slot 摊 1 次 `fence.acq_rel.sys`。
* `checkAbort` 把 abort 检查从"每轮一次全局读"降到"每 10000 轮一次"，即 **1/10000**。
* `connStepCache` 让大量 `waitPeer` 完全不需要远程读（主题 3）。
* 调优模型给出的端到端差异：NVLink 上 ring 的硬件延迟 LL 0.6 µs / LL128 1.9 µs / Simple 3.4 µs（[`tuning.cc:176`](../src/graph/tuning.cc#L176)）；PCI 上是 1.0 / 2.5 / 5.7 µs（[`tuning.cc:182`](../src/graph/tuning.cc#L182)）。Simple 的延迟劣势正是"等 tail + 读数据"两段式依赖带来的，而它的带宽优势由 0% 的 flag 开销保证（[`tuning.cc:344-346`](../src/graph/tuning.cc#L344) 对 LL 乘 0.5、LL128 乘 0.92，Simple 不打折）。

### ⑥ 面试考点

**Q1：为什么写 tail 之前要 fence，写 head 之前不用？**
A：tail 携带"数据已就绪"的语义，必须保证数据写在 tail 写之前对对端可见。head 只携带"槽位已释放"，即使提前可见也不会造成数据错误（对端重写该 slot 一定发生在本端读完之后的因果链上）。

**Q2：为什么用 `relaxed` 而不是 `release` 写 tail？**
A：因为 fence 已经显式插在前面了，`fence.acq_rel.sys + st.relaxed.sys` 的语义等价于 `st.release.sys`，但把 fence 提到循环外/条件化更灵活（`dataStored==false` 时可以整条省掉），而且 relaxed store 本身生成的代码更轻。

**Q3：为什么数据读取必须用 `ld.volatile.global`？**
A：`connStepPtr` 是用 volatile 轮询的，没有 acquire 语义，不会建立"后续读必须看到新数据"的 happens-before 关系。普通 `ld.global` 可能命中本 SM L1 中的陈旧副本。源码在 [`common_kernel.h:96`](../src/device/common_kernel.h#L96) 直接写明："使用易变加载，以防 credits 是通过易变（而非获取）语义轮询的"。

**Q4：`.sys` 和 `.gpu` scope 的区别，这里为什么必须 `.sys`？**
A：`.gpu` 只保证本 GPU 内（所有 SM + L2）可见；`.sys` 保证对整个系统（包括其他 GPU、通过 NVLink/PCIe 映射的 P2P 内存、以及 host）可见。跨卡通信显然需要 `.sys`。

**Q5：析构函数里为什么要等 `connFifo[prevStep % NCCL_STEPS].size != -1`？**
A：见 [`prims_simple.h:759-769`](../src/device/prims_simple.h#L759)。NetRegMode 下发送缓冲区被 NIC 直接 DMA 读，proxy 消费完后把 `size` 置 `-1`。kernel 返回前必须确认 proxy 已经发完，否则下一个 kernel 会覆盖这块被 NIC 直接访问的 buffer。

**Q6：`barrier()` 用 `barrier.sync.aligned` 且编号是 `15 - group`，为什么？**
A：`barrier_sync` 的 PTX 封装在 [`common.h:92-105`](../src/device/common.h#L92)，用 `.aligned` 变体（编译器可假设线程块内线程已对齐，省一次运行时检查）。编号 `15 - group` 是给不同同步组分配独立的命名 barrier（[`prims_simple.h:85-93`](../src/device/prims_simple.h#L85)），避免 tree 算法的上行组/下行组互相干扰；**0 号 barrier 被保留给 kernel 结束时的最终同步**，所以从 15 开始往下分配。

---

## 与其他章节的衔接

| 章节 | 关系 |
| --- | --- |
| [06-transport-p2p-shm.md](./06-transport-p2p-shm.md) | 本文主题 2/3/6 中 `conn->buffs / head / tail / ptrExchange / stepSize` 的**来源**都在 transport 层：`p2pSendConnect`/`p2pRecvConnect` 负责把它们指向对端显存。建议先读 06 再读本文的"连接加载"部分。 |
| [08-device-kernel-allreduce.md](./08-device-kernel-allreduce.md) | 本文是 08 的下一层：08 讲 `runRing`/`runTreeSplit` 的算法步骤，本文讲这些步骤里 `prims.directSend(...)` 一次调用内部发生了什么。 |
| [10-primitives-ll-ll128.md](./10-primitives-ll-ll128.md) | 本文的姊妹篇。三协议的差异全部收敛在 `ProtoSimple`/`ProtoLL`/`ProtoLL128` 三个类与对应的 `Primitives` 偏特化上；建议对照 `calcBytePerStep()`（本文主题 1）与 10 篇的"三协议对比表"一起看。 |
| [11-reduce-and-vectorization.md](./11-reduce-and-vectorization.md) | 本文主题 4 反复调用的 `reduceCopy` 是 11 的主角：`Unroll`、`BytePerPack`、`BigPackSize` 的选择与 `applyPreOp/applyReduce/applyPostOp` 的实现都在 11。 |
| [13-bandwidth-saturation.md](./13-bandwidth-saturation.md) | 本文主题 3/5 解释"为什么 Simple 能打满带宽"（8 slot 流水线 + slice 重叠）；13 从实测角度给出打满/打不满的现象与调参手段。 |
| [14-latency-optimization.md](./14-latency-optimization.md) | 本文主题 7 给出 Simple 的延迟劣势来源（等 tail + 读数据两段式）；14 会对比 LL/LL128 如何把这两段合成一段，以及 `NCCL_PROTO` 的选择策略（判定逻辑在 [`tuning.cc:459-558`](../src/graph/tuning.cc#L459)）。 |
