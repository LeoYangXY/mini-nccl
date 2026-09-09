# 小消息延迟优化

> 本文回答一个问题：**一次 8 KiB 的 AllReduce 为什么还要几微秒，以及这些微秒花在哪里。**
> 所有结论都给出源码位置；数字要么直接来自源码常量，要么明确标注为"量级估算"。

---

## 本文覆盖的源文件

| 文件 | 在本文中的作用 |
|---|---|
| [src/graph/tuning.cc](../src/graph/tuning.cc) | `baseLatencies` / `hwLatencies` 表、延迟累加公式、`ncclTopoGetAlgoTime` |
| [src/device/all_reduce.h](../src/device/all_reduce.h) | `runRing`（2(N-1) 次原语调用）vs `runTreeSplit`（1 次上行 + 1 次下行） |
| [src/device/prims_simple.h](../src/device/prims_simple.h) | device-side 的 spin 等待与信用推进（不经过 CPU） |
| [src/device/prims_ll.h](../src/device/prims_ll.h) | LL 的 flag 轮询读取（`readLL`） |
| [src/device/prims_ll128.h](../src/device/prims_ll128.h) | LL128 的 128B 行 + flag 线程 |
| [src/device/primitives.h](../src/device/primitives.h) | 三种协议的每 step 有效载荷 |
| [src/include/device.h](../src/include/device.h) | `NCCL_STEPS`、`NCCL_MIN_NTHREADS`、`NCCL_MAX_DEV_WORK_BATCH_COLLS` |
| [src/group.cc](../src/group.cc) | group 语义：把多个 collective 攒成一次 launch |
| [src/enqueue.cc](../src/enqueue.cc) | plan / work batch 构建、按 size 收缩 nc/nt、kernel launch |
| [src/misc/strongstream.cc](../src/misc/strongstream.cc) | CUDA Graph capture 检测与 per-graph stream |
| [src/register/coll_reg.cc](../src/register/coll_reg.cc) | 用户 buffer 注册（direct 路径、Graph 自动注册） |
| [src/transport/p2p.cc](../src/transport/p2p.cc) | 默认 P2P 不走 copy engine，`proxyProgress = NULL` |
| [src/graph/connect.cc](../src/graph/connect.cc) | 双树（double binary tree）的构建 |

---

## 1. 延迟的组成拆解

### 1.1 延迟瀑布图

```mermaid
flowchart LR
    T0["t=0<br/>主机侧 ncclAllReduce"]
    T1["① API + 入队<br/>ncclEnqueueCheck<br/>+0.5~2 µs"]
    T2["② 组结束 / plan 构建<br/>选算法·切 channel·算 chunk<br/>+1~2 µs"]
    T3["③ kernel launch<br/>cudaLaunchKernel<br/>+1~3 µs"]
    T4["④ 建链握手 / 首 step 信用<br/>waitPeer 首次 spin<br/>+0.6~4 µs × 次数"]
    T5["⑤ 链路传输<br/>size/BW<br/>8KiB ÷ 281GB/s ≈ 0.03 µs"]
    T6["⑥ 归约 + 写回 recvbuff<br/>+0.1~0.5 µs"]
    T7["t=总计<br/>≈ 6~10 µs"]

    T0 --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7

    style T0 fill:#e9ecef
    style T1 fill:#fff3cd
    style T2 fill:#fff3cd
    style T3 fill:#ffe0b2
    style T4 fill:#d1ecf1
    style T5 fill:#d4edda
    style T6 fill:#d4edda
    style T7 fill:#343a40,color:#fff
```

关键观察：**⑤ 传输时间在 8 KiB 时只有约 0.03 µs**（`8KiB / 281 GB/s`，量级估算），可以忽略；真正的耗时全在 ①~④ 这些**固定开销**上。这就是"8KB 以下延迟仍有几微秒"的根本原因。

### 1.2 源码里的延迟数字（baseLat / hwLat）

NCCL 自己的性能模型把延迟拆成"常数项 + 每步项"（[tuning.cc:647-672](../src/graph/tuning.cc#L647)）：

```669:671:src/graph/tuning.cc
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
```

（`bw` 单位 GB/s，所以 `nBytes/(1000*bw)` 的单位是 **µs**；`lat` 同样是 µs。）

**常数项 `baseLatencies[algo][proto]`**（[tuning.cc:166-172](../src/graph/tuning.cc#L166)），proto 顺序为 **LL / LL128 / Simple**：

| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | **6.8** | 14.0 | 8.4 |
| Ring | **6.6** | 14.0 | 8.4 |
| PAT | 8.0 | 8.0 | 8.0 |

> 单位：µs。这三行是"**再怎么优化也至少要走这么多**"的地板（除 CollNet/NVLS 外的其它算法在源码里是 0，因为本仓库未启用）。

**每步项 `hwLatencies[hw][algo][proto]`**

`NCCL_HW_NVLINK`（[tuning.cc:175-180](../src/graph/tuning.cc#L175)）：

| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 0.6 | 1.25 | 4.0 |
| Ring | 0.6 | 1.9 | **3.4** |

`NCCL_HW_PCI`（[tuning.cc:182-185](../src/graph/tuning.cc#L182)）：

| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 1.0 | 1.9 | 4.0 |
| Ring | 1.0 | 2.5 | **5.7** |

`NCCL_HW_NET`（[tuning.cc:188-191](../src/graph/tuning.cc#L188)）：Tree `5.0/8.5/14`，Ring `2.7/4.0/14.0`。

选哪套由拓扑决定（[tuning.cc:301-303](../src/graph/tuning.cc#L301)）：

```301:303:src/graph/tuning.cc
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;
```

**每步项乘多少步**

- `nsteps = 2*(nRanks-1)`（AllReduce，[tuning.cc:306](../src/graph/tuning.cc#L306)）
- Ring：`latencies += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat`（[tuning.cc:429](../src/graph/tuning.cc#L429)），单节点时就是 `nsteps × intraLat`
- Tree AllReduce：`latencies += 2 * ((nRanks/nNodes - 1) * intraLat + log2i(nNodes) * interLat)`（[tuning.cc:433](../src/graph/tuning.cc#L433)）

### 1.3 一个具体算例：2 卡 NVLink、8 KiB

| 组合 | 计算 | 总延迟（模型值） |
|---|---|---|
| **Ring + LL** | 6.6 + 2×(1)×0.6 | **7.8 µs** |
| Tree + LL | 6.8 + 2×0.6 | 8.0 µs |
| Ring + Simple | 8.4 + 2×3.4 | 15.2 µs |
| Tree + Simple | 8.4 + 2×4.0 | 16.4 µs |
| Ring + LL128 | 14.0 + 2×1.9 | 17.8 µs |

**结论**：在小消息区，模型给出的最优解是 **LL**（常数项 6.6/6.8 最低，每步 0.6 最低）。这也和"LL = Low Latency"的命名一致。

---

## 2. LL 协议为什么低延迟

**问题**：同样是一条 NVLink，为什么 LL 的每步延迟 0.6 µs，Simple 是 3.4 µs？

**做法**：Simple 的收端要等"发送方 post 了 tail"（一次跨卡可见的写 + 一次 fence），而 LL 把 **flag 和数据打包在同一条 16B 原子访问里**，收端一次 `ld.volatile.global.v4.u32` 就把 data 和 flag 一起拿到，flag 对了就是数据到了：

```116:129:src/device/prims_ll.h
  __device__ uint64_t readLL(int offset, int i) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    int spins = 0;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                   : "l"(&src->i4)
                   : "memory");
      if (checkAbort(abort, 1, spins)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
  }
```

发送端同理，一条 `storeLL` 同时写数据和翻转 flag（[prims_ll.h:303-304](../src/device/prims_ll.h#L303)）。

对比 Simple：

- Simple 的发端写完数据后要 `fence_acq_rel_sys()` 再 `st_relaxed_sys_global(connStepPtr, step)`（[prims_simple.h:207-215](../src/device/prims_simple.h#L207)），收端还要在 `waitPeer` 里 spin 读 `connStepPtr`（[prims_simple.h:149](../src/device/prims_simple.h#L149)）——**数据到位和"数据到位"这个信号是两次独立的跨卡访问**。
- LL 把两件事合成一条原子 16B 访问，**免掉了一次信号往返**。

**定量收益**：每步 0.6 µs vs 3.4 µs（[tuning.cc:176](../src/graph/tuning.cc#L176)），2 卡 AllReduce 共 2 步 → 总延迟 7.8 µs vs 15.2 µs，**差约 2 倍**。

**代价**：LL 的 16B 里只有 8B 是数据，有效带宽 **50%**（[primitives.h:63-65](../src/device/primitives.h#L63)，注释直接写 "Half is data"）。所以 LL **只适合小消息**——大消息上 50% 的带宽损失远大于省下的几微秒。

LL128 是折中：128B 一行里 15/16 是数据（93.75%），但常数项高达 14.0 µs（[tuning.cc:168-169](../src/graph/tuning.cc#L168)），**延迟上并不占优**，它的价值在大消息且 Simple 不可用时。

**调错方向**：`NCCL_PROTO=Simple` 跑 8 KiB → 延迟从 ~7.8 µs 涨到 ~15.2 µs；反过来 `NCCL_PROTO=LL` 跑 128 MB → 带宽腰斩（详见 13 章杠杆 5）。

---

## 3. Tree 算法为什么小消息延迟低

### 3.1 跳数：O(log N) vs O(N)

**做法（设备侧代码）**：

- **Ring**：每个 chunk 要经历 `2(nranks-1)` 次原语调用——`directSend` 1 次 + `directRecvReduceDirectSend` (nranks-2) 次 + `directRecvReduceCopyDirectSend` 1 次 + `directRecvCopyDirectSend` (nranks-2) 次 + `directRecv` 1 次（[all_reduce.h:93-143](../src/device/all_reduce.h#L93)）。
- **Tree**：`runTreeSplit` 把线程一分为二，上行组只做 1 次 `directSend`/`directRecvReduceDirectSend`，下行组只做 1 次 `directRecv`/`directRecvCopyDirectSend`（[all_reduce.h:277-336](../src/device/all_reduce.h#L277)）。**每个 chunk 只经过"上行 1 跳链 + 下行 1 跳链"**，中间节点的扇入扇出由 `FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>` 一次处理掉。

**定量收益**：8 卡时 Ring 是 14 步，Tree 是 `2×log2(8) = 6` 跳——**步数随 N 从线性降为对数**。

> **诚实说明（本仓库的调优模型没体现这一点）**：`tuning.cc` 里单机（nNodes=1）Tree 的延迟公式是
> `2 * ((nRanks/nNodes - 1) * intraLat + log2i(nNodes) * interLat)`（[tuning.cc:433](../src/graph/tuning.cc#L433)），
> `nNodes=1` 时化为 `2*(nRanks-1)*intraLat`，**和 Ring 的步数完全一样**。也就是说 Tree 在模型里只靠不同的 `hwLat` 常量（Tree Simple 4.0 vs Ring Simple 3.4）和 `treeCorrectionFactor`（[tuning.cc:640-644](../src/graph/tuning.cc#L640)）来区分，**O(log N) 的优势在模型中没有被建模出来**。上面的 O(log N) 结论来自设备侧代码的实际调用次数，不是来自模型。

### 3.2 线程分组让上下行重叠

`runTreeSplit` 不是"先做完上行再做下行"，而是**把线程块切成两组并行跑**（[all_reduce.h:242-263](../src/device/all_reduce.h#L242)）：

```252:263:src/device/all_reduce.h
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
```

两组用**不同的同步组**（`0 * Proto::MaxGroupWidth` vs `1 * Proto::MaxGroupWidth`，[all_reduce.h:295](../src/device/all_reduce.h#L295)、[all_reduce.h:320](../src/device/all_reduce.h#L320)），避免 barrier 互相干扰。

源码注释也说明了为什么要走 Split 而不是 UpDown：UpDown 版本"把两个阶段严格串行执行，实现简单但流水利用率不如 runTreeSplit"（[all_reduce.h:156](../src/device/all_reduce.h#L156)）；只有在 CUDA 11.2~11.3 + sm_80+ 上才会退回 UpDown（[all_reduce.h:364-368](../src/device/all_reduce.h#L364)）。

### 3.3 double binary tree（双树）

**做法**：`connectTrees` 对每个 channel 建**两棵互补的树**——`channel0 = comm->channels[c]` 与 `channel1 = channel0 + nChannels`（[connect.cc:192-194](../src/graph/connect.cc#L192)），两棵树各自设置独立的 `tree.up` / `tree.down`：

```199:220:src/graph/connect.cc
    if (comm->rank == ttp[node]) {
      NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
      NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
    }
    ...
    // 打印本 rank 在两棵树上的连接关系（调试用）
    if (comm->rank == ttp[node] || comm->rank == ttc0[node] || comm->rank == ttc1[node]) {
      INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c, channel0->tree.up, comm->rank, channel0->tree.down[0],
           channel0->tree.down[1], channel0->tree.down[2]);
      INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c + nChannels, channel1->tree.up, comm->rank,
           channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
    }
```

**收益**：每个 rank 在树 A 里是叶子、在树 B 里就是中间节点（或反之），于是**所有 rank 的收/发负担被摊平**，不会出现"某个 rank 既要收 2 个孩子又要发给父节点"的热点。这是 Tree 在多卡下能把带宽也做上去的关键（在延迟之外的收益）。

> 本仓库 `NCCL_MAX_TREE_ARITY = 3`（二叉树最多 2 个孩子 + 本地），`connectTrees` 的注释也写明"节点内用 NCCL_MAX_TREE_ARITY(=3) 个子；跨节点用 ncclGetDtree 生成两棵互补的树"（[connect.cc:180-181](../src/graph/connect.cc#L180)）。

---

## 4. 减少 kernel launch：group 批处理与 CUDA Graph

### 4.1 group 语义：多个 collective → 一个 plan → 一次 launch

**问题**：每个 `ncclAllReduce` 都 launch 一次 kernel，N 个就是 N × (launch 开销)。

**做法**：

1. `ncclGroupStart()` / `ncclGroupEnd()` 用线程局部的 `ncclGroupDepth` 记录嵌套层数（[group.cc:35](../src/group.cc#L35)、[group.cc:104-122](../src/group.cc#L104)），组内的调用先入队不 launch。
2. 组结束时 `ncclGroupEndInternal`（[group.cc:773](../src/group.cc#L773)）统一调度，在一个循环里把攒下的 plan 逐个 launch（[group.cc:355-381](../src/group.cc#L355)）。
3. 入队侧还会把**大小相近的操作聚合**：`enqueue.cc` 里"我们会把大小相差在 4 倍以内的操作聚合到一起"（[enqueue.cc:459-461](../src/enqueue.cc#L459)）。
4. 多个 task 被 `scheduleCollTasksToPlan`（[enqueue.cc:601](../src/enqueue.cc#L601)）塞进**同一个 plan**，而**一个 plan = 一次 `cudaLaunchKernel`**（[enqueue.cc:1789-1790](../src/enqueue.cc#L1789)）。

**batch 预算**：每个 work batch 的字节上限是 1024 B，装多少 collective 取决于 `sizeof(ncclDevWorkColl)`（[device.h:399-400](../src/include/device.h#L399)）：

```399:401:src/include/device.h
#define NCCL_MAX_DEV_WORK_BATCH_BYTES 1024
#define NCCL_MAX_DEV_WORK_BATCH_COLLS (NCCL_MAX_DEV_WORK_BATCH_BYTES / sizeof(ncclDevWorkColl))
#define NCCL_MAX_DEV_WORK_P2P_PER_BATCH 8
```

`ncclAddWorkBatchToPlan` 的判定逻辑见 [enqueue.cc:181](../src/enqueue.cc#L181)。按 `ncclDevWorkColl` 一百多字节（[device.h:296-326](../src/include/device.h#L296)）估算，**一个 batch 大致能装下 5~8 个 collective**（量级估算）。

**定量收益（模型里看得见）**：批处理会**摊销**延迟项——

```669:669:src/graph/tuning.cc
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
```

非 Ring 算法下，`numPipeOps` 个操作的延迟被压成 `DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)` 份——**同批的第 2~8 个操作几乎不再付常数延迟**。（Ring 不摊销，注释说明见 [tuning.cc:668](../src/graph/tuning.cc#L668)。）

### 4.2 CUDA Graph capture：把 launch 开销压到近零

**做法**：

1. **检测自己是否被 capture**：`ncclCudaGetCapturingGraph` 用 `cudaStreamGetCaptureInfo_v2/v3` 拿到当前 graph 与 graphId（[strongstream.cc:84-122](../src/misc/strongstream.cc#L84)）。
2. **每个 graph 一个专用 capture stream**：`ncclStrongStreamAcquire` 维护 `captureHead` 链表，按 `graphId` 复用/新建 captureStream，并用 `cudaStreamUpdateCaptureDependencies` 把 NCCL 内部流的节点接进用户的图里（[strongstream.cc:171-259](../src/misc/strongstream.cc#L171)）。
3. **capture 期间自动注册用户 buffer**：`NCCL_GRAPH_REGISTER` 默认 1（[enqueue.cc:308](../src/enqueue.cc#L308)），在 `comm->planner.persistent && ncclParamGraphRegister()` 时走 `ncclIpcGraphRegisterBuffer`（[coll_reg.cc:265-271](../src/register/coll_reg.cc#L265)、[coll_reg.cc:390-393](../src/register/coll_reg.cc#L390)），注册只做一次，之后每次 replay 都直接复用。

**定量收益**：`cudaLaunchKernel` 的每次调用开销（微秒级，见瀑布图 ③）变成图里的一个节点下发——**replay 时 CPU 侧的 per-op 开销接近零**（原理性说明；本仓库未见 benchmark 数据）。

**调错方向**：Graph capture 期间调用会分配显存的 CUDA API 会破坏 capture；另外注册有代价（`ipcRegisterBuffer` 要建 IPC 句柄），**只跑一次的 buffer 不值得注册**。

---

## 5. 单机 NVLink 下绕过 proxy / CPU 参与

**问题**：proxy 线程是 NCCL 里的"后台进度引擎"，它会拖慢小消息吗？

**做法**：**默认不会**——因为单机 P2P 的 `proxyProgress` 是 `NULL`：

```1531:1541:src/transport/p2p.cc
static void initCeOperation() {
  static int init = 0;
  if (!init) {
    useMemcpy = ncclParamP2pUseCudaMemcpy();
    if (useMemcpy) {
      p2pTransport.send.proxyConnect = p2pSendProxyConnect;
      p2pTransport.send.proxyProgress = p2pSendProxyProgress;
    }
    init = 1;
  }
}
```

也就是说，只有当 `NCCL_P2P_USE_CUDA_MEMCPY=1`（默认 0，[p2p.cc:141](../src/transport/p2p.cc#L141)）时，P2P 才有 proxy 进度函数。默认路径下：

- **同步全在 device 侧**：收端在 `waitPeer` 里 spin 读 `connStepPtr`（[prims_simple.h:149](../src/device/prims_simple.h#L149)），发端写完直接 `st_relaxed_sys_global(connStepPtr, step)`（[prims_simple.h:213](../src/device/prims_simple.h#L213)）——**没有一次 GPU→CPU→GPU 的往返**。
- proxy 只在**建连/setup** 阶段出现（`ncclProxyConnect` + `ncclProxyMsgSetup`，[p2p.cc:503](../src/transport/p2p.cc#L503)、[p2p.cc:575](../src/transport/p2p.cc#L575)），不在每次通信的数据通路上。
- 主机侧每个 group 只做**一次** `cudaLaunchKernel`（[group.cc:370](../src/group.cc#L370)）。

**定量收益**：省掉的是每次传输的"GPU 通知 CPU → CPU 提交 → CPU 通知 GPU"往返，量级是**微秒级**，对小消息（总延迟才几微秒）是决定性的（量级估算）。

**调错方向**：开 `NCCL_P2P_USE_CUDA_MEMCPY=1` 会把数据通路拖回 proxy + CUDA event（[p2p.cc:612-619](../src/transport/p2p.cc#L612)、[p2p.cc:846](../src/transport/p2p.cc#L846)），小消息延迟显著变差。多机场景（走 `net`）则是另一回事——那里 proxy 是必需的。

---

## 6. 小消息的 chunk / 线程数收缩

**问题**：8 KiB 的数据如果也开 32 个 block × 512 线程，绝大多数线程会发现自己无数据可搬，但 barrier 一个都不少。

**做法**：`topoGetAlgoInfo` 的第二步（[enqueue.cc:2120-2181](../src/enqueue.cc#L2120)）：

```2127:2157:src/enqueue.cc
  int nc = comm->nChannels;                                          // 起始 channel 数 = 通信域可用的全部 channel
  int nt = comm->maxThreads[info->algorithm][info->protocol];        // 该算法/协议下每个 block 的最大线程数
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol]; // 单线程至少应处理的字节数
  ...
  } else {
    // 环/树 的 通道 调优：数据量不够就逐个减少 通道，但至少保留 1 个
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }
```

通道数减到 1 之后还嫌小，就**折半减线程**（要求 `nt % 128 == 0` 以保持 warp 对齐，[enqueue.cc:2162-2168](../src/enqueue.cc#L2162)）：

```2164:2167:src/enqueue.cc
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
```

阈值常量（[comm.h:59-61](../src/include/comm.h#L59)）：`NCCL_LL_THREAD_THRESHOLD = 8`、`NCCL_LL128_THREAD_THRESHOLD = 8`、`NCCL_SIMPLE_THREAD_THRESHOLD = 64`（字节/线程）；Ring+LL 还要再乘 `nRanks`（[tuning.cc:610](../src/graph/tuning.cc#L610)）。可用 `NCCL_THREAD_THRESHOLDS`（[tuning.cc:615](../src/graph/tuning.cc#L615)）覆盖。

最后是 Simple 协议追加同步 warp 与 3-warp 下限（[enqueue.cc:2170-2178](../src/enqueue.cc#L2170)）：

```2170:2178:src/enqueue.cc
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // 额外增加一个 warp 专门负责同步
    // 树 采用了“线程分组(split)”模型：上行组与下行组各自需要同步 线程束，因此追加更多
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4 * WARP_SIZE;
  }
  // 兜底：无论怎么削减，至少保证 3 个 线程束，否则连基本的收/发/同步分工都无法完成
  nt = nt / WARP_SIZE < 3 ? 3 * WARP_SIZE : nt;
  if (info->algorithm == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree 现在恒定使用全部线程
```

**算例（假设 `comm->nChannels = 32`，Ring + Simple，`nBytes = 8 KiB = 8192 B`）**：

| 步骤 | 判据 | 结果 |
|---|---|---|
| 初始 | nc=32, nt=512, thr=64 | 32×512×64 = 1,048,576 B |
| 减 channel | `8192 < nc×512×64` 一直成立 → nc: 32 → 1 | nc = 1 |
| 减线程 | `8192 < 1×nt×64`：nt 512→256→128（128 时 1×128×64 = 8192，不再成立） | nt = 128 |
| Simple 追加 | Ring + 1 warp | nt = 160 |
| 下限检查 | 160/32 = 5 ≥ 3 | **1 block × 160 线程 = 5 warps** |

对比 128 MiB（134,217,728 B）：`134217728 ≥ 32×512×64` → **完全不收缩**，nc=32、nt=512(+32)。

**定量收益**：8 KiB 时只启动 1 个 block × 5 warps，而不是 32 个 block × 17 warps —— **barrier 次数与空转线程数都降了 1~2 个数量级**（量级估算）。

**调错方向**：`NCCL_MIN_NCHANNELS=32` 强开 32 个 channel 跑 8 KiB → 31 个 block 几乎空转，延迟反而变差；`NCCL_THREAD_THRESHOLDS` 设得过大 → 中等消息被过早收缩，带宽上不去。

---

## 7. 面试常见问题

### Q1：为什么 8KB 以下 NCCL 延迟仍有几微秒？

**答（拆三段）**：

1. **传输时间根本不是瓶颈**：`8 KiB / 281 GB/s ≈ 0.03 µs`（量级估算）。
2. **固定开销才是**：模型里 `baseLatencies` 一项就是 **6.6 µs（Ring+LL）/ 6.8 µs（Tree+LL）**（[tuning.cc:166-172](../src/graph/tuning.cc#L166)），这是"kernel 起来 + 装载参数 + 走完流程"的地板，和 size 无关。
3. **再加上每步链路延迟**：2 卡 AllReduce 是 `nsteps = 2(nRanks-1) = 2` 步，NVLink+LL 每步 0.6 µs → 再 +1.2 µs。合计 **≈ 7.8 µs**。Simple 则是 `8.4 + 2×3.4 = 15.2 µs`。

一句话：**小消息延迟 ≈ 常数项 + 步数 × 每步延迟，传输项可忽略**。

### Q2：怎么进一步降？NCCL 实际有哪些手段？

| 手段 | 做法 | 源码 | 收益 |
|---|---|---|---|
| 用 LL 协议 | 让小消息走 `proto LL` | [prims_ll.h:116-129](../src/device/prims_ll.h#L116) | 每步 3.4 → 0.6 µs，常数项 8.4 → 6.6 µs |
| 减少 launch | `ncclGroupStart/End` 把多个 collective 合成一个 plan 一次 launch | [group.cc:773](../src/group.cc#L773)、[enqueue.cc:1789](../src/enqueue.cc#L1789) | 省掉 N-1 次 `cudaLaunchKernel` |
| 批处理摊销延迟 | 同批操作共享延迟项 | [tuning.cc:669](../src/graph/tuning.cc#L669) | `latCount = DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)` |
| CUDA Graph | capture 后 replay，CPU 侧 per-op 开销近零 | [strongstream.cc:84-122](../src/misc/strongstream.cc#L84) | 省掉瀑布图 ②③ 的重复开销 |
| 用户 buffer 注册 → direct | 省一次 staging 读 + 一次 staging 写 | [coll_reg.cc:264-281](../src/register/coll_reg.cc#L264)、[prims_simple.h:800](../src/device/prims_simple.h#L800) | 少两次显存访问 |
| Graph 自动注册 | `NCCL_GRAPH_REGISTER=1`（默认）在 capture 时注册一次 | [enqueue.cc:308](../src/enqueue.cc#L308) | 注册成本不进关键路径 |
| 保证 16B 对齐 | 否则掉到 `sizeof(T)` 粒度 | [common_kernel.h:226-231](../src/device/common_kernel.h#L226) | 避免访存指令数 ×4 |
| 不引入 proxy | 默认不走 copy engine | [p2p.cc:1531-1541](../src/transport/p2p.cc#L1531) | 省 GPU↔CPU 往返 |

### Q3：那极限在哪？为什么不能降到 1 µs？

**答（NCCL 的实际限制）**：

1. **`baseLatencies` 是模型里的常数项**，无法通过环境变量消除——它代表"一次 collective 从 API 进来到 kernel 跑完"的最小固定成本（[tuning.cc:166-172](../src/graph/tuning.cc#L166)）。
2. **并行度有下限**：`NCCL_MIN_NTHREADS = 128`（4 warps，[device.h:104](../src/include/device.h#L104)），代码里还有"至少 3 个 warp，否则连收/发/同步的分工都完不成"的兜底（[enqueue.cc:2176](../src/enqueue.cc#L2176)）。再少就无法分工。
3. **`NCCL_STEPS = 8` 是编译期常量**（[device.h:36](../src/include/device.h#L36)），流水线深度和 FIFO 布局都依赖它，改它需要重新编译。
4. **同步本身要跨卡可见**：即使是 LL，收端也要 spin 到对端的 flag 写进来（[prims_ll.h:121-127](../src/device/prims_ll.h#L121)），这是一次跨卡访存的物理下限。
5. **kernel launch 不在模型里**：`*time = lat*latCount + nBytes/(1000*bw)` 只建模了 kernel 内部，launch 开销是额外付的（[tuning.cc:670](../src/graph/tuning.cc#L670)）。所以 Graph capture 能拿到"模型之外"的收益。

### Q4：小消息该用 Ring 还是 Tree？

**答**：

- **设备侧**：Tree 每 chunk 只需 1 次上行 + 1 次下行（[all_reduce.h:277-336](../src/device/all_reduce.h#L277)），Ring 需要 `2(nranks-1)` 次原语调用（[all_reduce.h:93-143](../src/device/all_reduce.h#L93)），**跳数上 Tree 是 O(log N)、Ring 是 O(N)**。
- **但模型里单机 Tree/Ring 的步数公式是一样的**（都是 `2(nRanks-1)×intraLat`，[tuning.cc:429 vs 433](../src/graph/tuning.cc#L429)），Tree 只靠更小的 `hwLat` 和 `treeCorrectionFactor` 区分。所以 2 卡场景下模型算出来两者几乎打平（7.8 vs 8.0 µs），卡数多时才体现 Tree 的优势。
- **实践**：小消息让 NCCL 自己选（`ncclTopoGetAlgoTime` 挑最小 time，[enqueue.cc:2083-2098](../src/enqueue.cc#L2083)）；想做对照实验用 `NCCL_ALGO=Tree` / `NCCL_ALGO=Ring`。

---

## 与其他章节的衔接

| 章节 | 关系 |
|---|---|
| [03-channel-ring-tree.md](./03-channel-ring-tree.md) | 本文 §3 的 Tree 拓扑（双树的父/子关系、ring 的 prev/next）与 §6 的 `comm->nChannels` 来源在那里展开 |
| [04-algo-protocol-tuning.md](./04-algo-protocol-tuning.md) | 本文 §1 的延迟模型、§2 的协议对比、§6 的阈值常量，都是那一章"算法×协议选择"的输入 |
| [08-device-kernel-allreduce.md](./08-device-kernel-allreduce.md) | 本文 §3 的 `runRing` / `runTreeSplit` 在那里逐行解读（线程分组、同步组编号、postOp 时机） |
| [09-primitives-simple.md](./09-primitives-simple.md) | 本文 §5 的 device-side spin、§6 的 warp 分工，来自 Simple 原语的 `waitPeer`/`postPeer` 与角色标志 |
| [10-primitives-ll-ll128.md](./10-primitives-ll-ll128.md) | 本文 §2 的 LL 免同步往返与 LL128 的 93.75%，在那里给出 flag 布局与收发的完整实现 |
| [13-bandwidth-saturation.md](./13-bandwidth-saturation.md) | 那一章讲"大消息怎么把带宽打满"；两章共用同一组"按 size 收缩 nc/nt"的代码，只是看的方向相反（延迟看收缩是否生效，带宽看收缩是否过早） |
