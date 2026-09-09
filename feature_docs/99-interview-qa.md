# 99 · 面试速查与高频问答

> 用法：先看 §1 必背数字与公式，再按 §2 的题目自问自答。
> 每题答案都给出"可复述的骨架"，深挖时点链接跳源码。

---

## 1. 必背数字与公式

### 1.1 核心常量（都来自源码）

| 常量 | 值 | 出处 |
|---|---|---|
| `NCCL_STEPS` | `8` | [device.h:L36](../src/include/device.h#L36) —— 每 channel 环形 FIFO 的 slot 数 |
| `ALLREDUCE_CHUNKSTEPS` | `NCCL_STEPS/2 = 4` | [collectives.h:L28](../src/include/collectives.h#L28) |
| `ALLREDUCE_SLICESTEPS` | `NCCL_STEPS/4 = 2` | [collectives.h:L27](../src/include/collectives.h#L27) |
| ⇒ AllReduce 的 `ProtoSimple<2,2>` | `SlicePerChunk=2, StepPerSlice=2` | [all_reduce.h:L352](../src/device/all_reduce.h#L352) |
| `NCCL_MAX_NTHREADS` | `640` | [device.h:L103](../src/include/device.h#L103) |
| `NCCL_SIMPLE_MAX_NTHREADS` | `512` | [device.h:L105](../src/include/device.h#L105) |
| 最少线程数兜底 | `3 × WARP_SIZE = 96` | [enqueue.cc:L2176](../src/enqueue.cc#L2176) |
| Simple+Ring 额外同步线程 | `+1 warp` | [enqueue.cc:L2171](../src/enqueue.cc#L2171) |
| Simple+Tree 额外同步线程 | `+4 warp` | [enqueue.cc:L2173](../src/enqueue.cc#L2173) |
| `DEFAULT_BUFFSIZE` | `1<<22 = 4 MiB` | [init.cc:L827](../src/init.cc#L827) —— **每 channel 每协议**一份 |
| ⇒ 每 slot 大小 | `4 MiB / 8 = 512 KiB` | `stepSize = buffSizes[SIMPLE]/NCCL_STEPS`（[prims_simple.h:L623](../src/device/prims_simple.h#L623)） |
| `MAXCHANNELS` | `64` | channel 硬上限 |
| Ring 图的 `maxChannels` | `MAXCHANNELS/2 = 32` | [init.cc:L1206](../src/init.cc#L1206) |

> **顺手算一笔显存**：`nChannels × 3 协议 × buffSize`。8 个 channel 光 Simple 就是
> `8 × 4 MiB = 32 MiB` —— 这就是"channel 不能无脑加"的第二个代价（第一个是抢 SM）。

### 1.2 tuning 代价模型常量（单位：微秒 / GB/s）

`baseLatencies[algo][proto]`（[tuning.cc:L166-L172](../src/graph/tuning.cc#L166)）：

| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 6.8 | 14.0 | 8.4 |
| Ring | 6.6 | 14.0 | 8.4 |

`hwLatencies[NVLINK][algo][proto]`（每跳，[tuning.cc:L176](../src/graph/tuning.cc#L176)）：

| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 0.6 | 1.25 | 4.0 |
| Ring | 0.6 | 1.9 | 3.4 |

对比 PCI（[tuning.cc:L182](../src/graph/tuning.cc#L182)）：Ring/Simple 是 **5.7**（NVLink 是 3.4），
NET（[tuning.cc:L188](../src/graph/tuning.cc#L188)）是 **14.0** —— 这就是"跨节点延迟高一个量级"的量化依据。

带宽上限（Hopper / N1，[tuning.cc:L198](../src/graph/tuning.cc#L198)、[L205](../src/graph/tuning.cc#L205)）：

| 项 | 值 | 含义 |
|---|---|---|
| `llMaxBws[Hopper][N1]` | `141.0 GB/s` | LL 协议整体带宽天花板 |
| `perChMaxRingLL128Bws[Hopper]` | `36.7 GB/s` | LL128 **单 channel** 带宽上限 |

> **这两个数字本身就是最好的答案素材**：LL 有全局天花板（因为只有一半带宽有效），
> LL128 的上限是"每 channel"的（说明它靠加 channel 就能扩展）。

### 1.3 必会公式

**（1）Ring AllReduce 每 rank 收发量**

数据量 `S`，`N` 个 rank，切成 `N` 个 chunk（每块 `S/N`）：

- ReduceScatter 阶段：`N-1` 步，每步收/发 `S/N`
- AllGather 阶段：`N-1` 步，每步收/发 `S/N`

$$\text{每 rank 单向流量} = 2(N-1)\cdot \frac{S}{N} = \frac{2(N-1)}{N}S$$

`N→∞` 时趋于 `2S`，**与 N 无关** —— 这是 Ring 相对朴素做法（O(N·S)）的根本优势。

**（2）bus bandwidth**

$$\text{algbw} = \frac{S}{t},\qquad \text{busbw} = \text{algbw}\times\frac{2(N-1)}{N}$$

源码就是这两行（[tests/src/all_reduce.cu:L81-L87](../tests/src/all_reduce.cu#L81)）：

```c
void AllReduceGetBw(size_t count, size_t typesize, double sec,
                    double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}
```

**为什么要引入 busbw**：`algbw` 会随 N 变化（N 越大同样时间搬的有效数据越少），
不同规模无法横向比较；乘上校正因子后得到的 busbw ≈ **硬件链路的实际利用率**，
可以直接和 NVLink 标称带宽对比。

**本仓库的数字**：2 卡时 `factor = 2×1/2 = 1`，所以 `busbw == algbw`。
128MB 档 busbw ≈ 281 GB/s。

**（3）tuning 的时间模型**

$$t(\text{algo},\text{proto}) = \text{latency} + \frac{S}{\text{bandwidth}}$$

`latency = baseLatency + nHops × hwLatency`。小 `S` 时 latency 主导 → 选 latency 小的
（LL / Tree）；大 `S` 时 `S/bw` 主导 → 选 bandwidth 大的（Simple / Ring）。
**"阈值"不是硬编码的，而是两条直线的交点。**

**（4）LL / LL128 有效带宽**

| 协议 | 数据 : 总量 | 有效带宽 |
|---|---|---|
| Simple | 无 inline flag | ~100% |
| LL | 4B data + 4B flag | **50%** |
| LL128 | 120B data + 8B flag（每 128B） | **15/16 = 93.75%** |

---

## 2. 高频问答

### Q1 讲一下 `ncclAllReduce` 从调用到完成发生了什么？

**骨架答案（4 层）**：

1. **API 层**：`ncclAllReduce` 只把参数打包成 `ncclInfo`
   （[collectives.cc:L176](../src/collectives.cc#L176)），立刻交给 `ncclEnqueueCheck`。
2. **入队层**：`ncclEnqueueCheck`（[enqueue.cc:L3209](../src/enqueue.cc#L3209)）
   套一个**隐式 group**，校验参数后 `taskAppend` 把请求挂到 `comm->planner` 队列，
   **不做任何实际通信**。
3. **决策 + 计划层**：`ncclGroupEndInternal` → `groupLaunch`
   （[group.cc:L605](../src/group.cc#L605)）→ `ncclGetAlgoInfo`
   （[enqueue.cc:L2195](../src/enqueue.cc#L2195)）从代价表里选出
   `(算法, 协议, nChannels, nWarps)` → `scheduleCollTasksToPlan`
   （[enqueue.cc:L601](../src/enqueue.cc#L601)）把 task 编成 `ncclDevWorkColl` →
   `ncclLaunchKernel`（[enqueue.cc:L1784](../src/enqueue.cc#L1784)）以
   `grid = nChannels` 启动 kernel。
4. **设备层**：`ncclKernelMain`（[common.h:L364](../src/device/common.h#L364)）
   把 `blockIdx` 映射成 `channelId`、加载 comm/channel/work 到 shmem，分派到
   `RunWorkColl<AllReduce, RING, SIMPLE>` → `runRing`
   （[all_reduce.h:L36](../src/device/all_reduce.h#L36)）跑 `2(N-1)` 步，
   每步调 `Primitives` 在 NVLink 上直接 load/store 对端显存。

**加分点**：主动强调"`ncclAllReduce` 是异步的，返回时一个字节都没搬"，以及
"所有贵的事情（拓扑探测、通道搜索、建连、性能标定）都在 `ncclCommInitRank` 里做完了"。

→ 详见 [00-overview](./00-overview-allreduce-lifecycle.md)

---

### Q2 channel 到底是什么？为什么需要多个？

**一句话**：channel 是一条**独立的逻辑通信流水线**（一组 ring/tree 邻居关系 +
独立的 FIFO buffer + 独立的 head/tail 计数器），**一对一映射到一个 CUDA block**。

证据就一行（[enqueue.cc:L1789](../src/enqueue.cc#L1789)）：

```c
  dim3 grid  = {(unsigned)nChannels, 1, 1};
```

**为什么要多个**：

1. **单个 SM 打不满 NVLink**。一个 block 只在一个 SM 上跑，它能发起的
   outstanding load/store 有限；要吃满几百 GB/s 必须多个 SM 同时发。
2. **多条物理 link 需要多个逻辑流并行占用**。拓扑搜索会把不同 channel 映射到不同
   的 NVLink，单 channel 只能用其中一条。
3. **计算与通信重叠**：多 channel 让某个 channel 等对端时，其他 channel 仍在搬数据。

**反向也要会答**：channel 不是越多越好 —— block 太多会和计算 kernel 抢 SM，
并且每个 channel 都要独立的 buffer（显存开销 × nChannels）。所以 NCCL 会根据数据量
**主动减少** channel（[enqueue.cc:L2153](../src/enqueue.cc#L2153)）：

```c
    while (nBytes < nc * nt * threadThreshold) { if (nc >= 2) nc--; else break; }
```

**可验证**：`NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1` 跑 128MB，busbw 会明显下降。

→ 详见 [03-channel-ring-tree](./03-channel-ring-tree.md)、[13-bandwidth-saturation](./13-bandwidth-saturation.md)

---

### Q3 Ring AllReduce 为什么是"最优"的？为什么不是 2N 步而是 2(N-1) 步？

**最优性**：见 §1.3 公式（1），每 rank 流量 `2(N-1)/N × S`，与 N 无关，逼近理论下界
（AllReduce 的下界是 `2(N-1)/N × S`）。

**为什么 2(N-1) 而不是 2N**：

- ReduceScatter 需要 `N-1` 步：一个 chunk 从"起点 rank"走到"归属 rank"最多 `N-1` 跳，
  走满就累加完了全部 N 个 rank 的贡献（起点自己那份是本地加的）。
- AllGather 同理 `N-1` 步。
- **最后一步用 `directRecv` 只收不发**（[all_reduce.h:L143](../src/device/all_reduce.h#L143)），
  因为这个 chunk 的下一站正是它的归属 rank，那里早就有结果了，再转发纯属浪费带宽。

源码里的 5 段结构（[all_reduce.h:L90-L143](../src/device/all_reduce.h#L90)）值得背：

| 步 | 原语 | 干什么 |
|---|---|---|
| 0 | `directSend` | 流水线注水，只发不收 |
| 1..N-2 | `directRecvReduceDirectSend` | 收部分和 + 加本地 + 转发 |
| N-1 | `directRecvReduceCopyDirectSend`（`postOp=true`） | 最后一次累加 + 写 recvbuff + 开启 AllGather |
| N..2N-3 | `directRecvCopyDirectSend` | 收最终结果 + 落地 + 转发 |
| 2N-2 | `directRecv` | 只收不发，收尾 |

**加分点**：`postOp=true` 只出现在第 N-1 步 —— `Avg` 这种要"除以 N"的后处理必须在
累加完全部 rank 后**只做一次**。

→ 详见 [08-device-kernel-allreduce](./08-device-kernel-allreduce.md)

---

### Q4 Ring 和 Tree 什么时候各用哪个？谁决定的？

**决定者**：`ncclGetAlgoInfo` → `topoGetAlgoInfo`，遍历 `[算法][协议]` 代价表取最小
（[enqueue.cc:L2083-L2094](../src/enqueue.cc#L2083)）。

**判断依据**：`t = latency + S/bandwidth`

- **小消息**：latency 主导。Tree 的跳数是 `O(log N)`，Ring 是 `O(N)`，
  所以 N 大时 Tree 明显更快；单机 2 卡时 N 太小，差别不大。
- **大消息**：`S/bw` 主导。Ring 的每 rank 流量最优，Tree 会打折
  （还有 `treeCorrectionFactor`，[tuning.cc:L640](../src/graph/tuning.cc#L640)），
  所以选 Ring。

**Tree 为什么带宽会打折**：树的内部节点要同时和父 + 多个子通信，链路负载不均衡；
NCCL 用 **double binary tree** 缓解 —— 构造两棵互补的树，让每个节点在一棵树里是叶子、
在另一棵里是内部节点，两棵树各跑一半数据，从而**同时打满上行和下行**。

→ 详见 [04-algo-protocol-tuning](./04-algo-protocol-tuning.md)、[03-channel-ring-tree](./03-channel-ring-tree.md)

---

### Q5 Simple / LL / LL128 三个协议的本质区别？

**本质区别在"接收方怎么知道数据到了"**：

| 协议 | 同步方式 | 有效带宽 | 延迟 | 适用 |
|---|---|---|---|---|
| **Simple** | 独立的 `head/tail` 计数器（需要额外的同步往返 + fence） | ~100% | 高 | 大消息 |
| **LL** | flag **内联**在数据里：8B = 4B data + 4B flag，接收方 spin 检查 flag | **50%** | 极低 | 小消息 |
| **LL128** | 每 128B 里放 8B flag：120B data + 8B flag | **93.75%** | 低 | 中等消息 |

**LL 为什么快**：数据和"数据已就绪"这个信号在**同一次写**里到达，接收方不需要先读
tail、再读数据（省掉一次往返）。代价是一半带宽用来传 flag。

**LL128 为什么可行**：128B 是 NVLink/NVSwitch 的**原子写粒度** —— 一次 128B 写不会被
拆分或重排，所以"flag 可见 ⇒ 同 128B 内的 120B 数据也可见"这个推断是硬件保证的。
用 8/128 的开销换到了 LL 的免同步特性。**LL128 依赖硬件保证，所以 NCCL 会做可用性检测，
不满足就退回 Simple/LL。**

**Simple 的补偿**：Simple 用"额外一个 warp 专职同步"来摊薄同步成本
（[enqueue.cc:L2171](../src/enqueue.cc#L2171) 的 `nt += WARP_SIZE`），
其余 warp 全力搬数据，因此大消息下同步开销占比可忽略。

→ 详见 [10-primitives-ll-ll128](./10-primitives-ll-ll128.md)、[09-primitives-simple](./09-primitives-simple.md)

---

### Q6 两张 GPU 之间的字节到底怎么过去的？走 `cudaMemcpy` 吗？

**不走**。NVLink P2P 场景下：

1. init 阶段 `ncclTransportP2pSetup`（[init.cc:L1660](../src/init.cc#L1660)）
   通过 `cudaDeviceEnablePeerAccess` + IPC 句柄，把**对端的 buffer 映射进本地地址空间**
   （[transport/p2p.cc](../src/transport/p2p.cc)）。
2. 运行期 kernel 里，`Primitives` 直接用普通的 `ld/st` 指令（128-bit 向量化）
   读写那个映射过来的指针 —— 对 SM 来说和访问本地显存**没有语法差别**，
   地址翻译和跨链路搬运由硬件完成。

**为什么不用 memcpy/copy engine**：

- 省掉一次 launch 与 CE 同步开销；
- 归约必须在 GPU 上做，数据反正要过 SM，直接 `ld` 对端 → reduce → `st` 下一跳，
  **一趟就完成"搬 + 算"**，而 memcpy 方案要"搬进来"再"算一遍"，多一次显存往返。

**退化路径**：没有 NVLink / P2P 不可用时退到 `shm` transport
（[transport/shm.cc](../src/transport/shm.cc)），走 `/dev/shm` + PCIe，**带宽腰斩**。
面试里要主动提"怎么确认没退化"：`NCCL_DEBUG=INFO` 看连接日志里是不是 P2P/NVLink。

→ 详见 [06-transport-p2p-shm](./06-transport-p2p-shm.md)

---

### Q7 kernel 怎么保证不会读到对端还没写完的数据？

**三层机制**：

1. **单调递增的 step 计数器 + 环形 FIFO**。每个连接有 `head`（消费者进度）和
   `tail`（生产者进度），都是**只增不减的 64 位计数**，slot 号 = `step % NCCL_STEPS`。
   用单调计数而不是环形指针比较，**天然避免 ABA / 回绕歧义**。
2. **流控条件**：NCCL 用**一行代码同时表达了收和发两个方向**的等待
   （[prims_simple.h:L149-L150](../src/device/prims_simple.h#L149)）：

   ```c
        while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
          connStepCache = loadStepValue(connStepPtr);
        }
   ```

   - **收方向**（`isSendNotRecv=false`）：等 `对端已生产的 step ≥ 我要读的 step`；
   - **发方向**（`isSendNotRecv=true`）：加上 `NCCL_STEPS` 的余量，等
     `对端已消费的 step + 8 ≥ 我要写的 step`，即**最多允许 8 个 slot 在途**。

   `connStepCache` 是关键优化：远端 step 是一次跨设备访存，很贵，
   所以先用寄存器里的缓存值判断，只在不满足时才真的去 `loadStepValue` 重读。
   slot 地址就是 `connEltsFifo + (step % NCCL_STEPS) * connStepSize`
   （[L184](../src/device/prims_simple.h#L184)），而
   `stepSize = buffSizes[SIMPLE] / NCCL_STEPS`（[L623](../src/device/prims_simple.h#L623)）。

3. **内存序**：跨设备可见性靠 `volatile` / `ld.volatile.global` 访存绕过缓存，
   写数据与写 step 之间插 `__threadfence_system`，保证"step 可见 ⇒ 数据可见"。

**block 内部**还有一层：`barrier()`（`asm barrier.sync`，比 `__syncthreads` 更细粒度，
只同步参与的线程组），配合 `ROLE_WAIT_RECV / ROLE_POST_SEND` 之类的角色位掩码 ——
**少量线程专职做 flag 同步，其余线程全力搬数据**。

→ 详见 [09-primitives-simple](./09-primitives-simple.md)、[07-proxy-progress-engine](./07-proxy-progress-engine.md)

---

### Q8 Proxy 线程是干什么的？单机 NVLink 需要它吗？

**存在原因**：GPU kernel 不能调用 CPU 侧 API（`ibv_post_send`、socket 等）。
网络传输必须有一个 CPU 后台线程代为推进 —— 这就是 proxy
（[proxy.cc](../src/proxy.cc)）。

**工作方式**：enqueue 阶段生成 `ncclProxyOp` 挂到 plan 上，launch 后
`ncclLaunchKernelAfter_NoCuda`（[enqueue.cc:L1884](../src/enqueue.cc#L1884)）
把它提交出去；progress 线程轮询状态机，通过和 kernel **共享的 head/tail** 与设备侧握手。

**单机 NVLink 的关键结论（高频陷阱题）**：数据面**完全在 GPU 内闭环**，
proxy 不参与字节搬运。这正是单机 NVLink 延迟低的原因之一 —— **少一次 CPU-GPU 往返**。
面试时不要说"NCCL 都靠 proxy 搬数据"，那是网络路径。

→ 详见 [07-proxy-progress-engine](./07-proxy-progress-engine.md)

---

### Q9 带宽没打满，你怎么排查？

**按"从上到下"的层级顺序答，每层给一个具体动作**：

| 层 | 怀疑什么 | 怎么查 |
|---|---|---|
| 拓扑 | 退化到 PCIe/SHM 而不是 NVLink | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` 看连接类型 |
| 算法/协议 | 选了 Tree 或 LL 跑大消息 | 日志看实际选中的 algo/proto；用 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 对照 |
| channel 数 | 太少（SM 不够发满链路）或太多（抢 SM） | `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` 扫一遍 |
| 消息大小 | 消息太小，被并行度收缩逻辑降级了 | 加大 `-b/-e`，或用 group 合并多个小 collective |
| 对齐 | buffer 不是 16B 对齐 → 向量化访存退化 | 检查 `cudaMalloc` 偏移，尤其是"从大 buffer 里切一段"的情况 |
| 零拷贝 | 没注册用户 buffer，多一次 staging copy | 看是否走 direct/registered 路径 |
| 竞争 | 通信 kernel 和计算 kernel 抢 SM | 用 nsys 看 SM 占用与 kernel 重叠 |
| 测量 | 用了 algbw 而不是 busbw，或没预热 | 确认公式与 warm-up |

**加分点**：主动说"我会先用 `NCCL_MIN_NCHANNELS=1` 做一个**故意变慢的对照实验**，
确认 channel 是不是当前瓶颈"，体现你会做**受控变量实验**而不是猜。

→ 详见 [13-bandwidth-saturation](./13-bandwidth-saturation.md)

---

### Q10 小消息（几 KB）延迟怎么降？为什么还是有几微秒？

**能降的手段**：

1. **协议换 LL**：免同步往返，`hwLatency[NVLINK][Ring][LL] = 0.6µs` vs
   `Simple = 3.4µs`（[tuning.cc:L176](../src/graph/tuning.cc#L176)）。
2. **算法换 Tree**：跳数从 `O(N)` 降到 `O(log N)`。
3. **减少 launch 次数**：用 `ncclGroupStart/End` 把多个小 collective 合并进**一个
   plan、一次 kernel launch**（[group.cc](../src/group.cc)、[enqueue.cc](../src/enqueue.cc)）。
4. **CUDA Graph capture**：把 launch 开销降到近零（[misc/strongstream.cc](../src/misc/strongstream.cc)），
   这是小消息场景收益最大的单项优化。
5. **降并行度**：数据量小时主动减 channel / 减线程
   （[enqueue.cc:L2153-L2168](../src/enqueue.cc#L2153)），避免启动一堆空转 block。
6. **绕过 CPU**：单机 NVLink 纯 device-side 同步，省一次 CPU-GPU 往返。

**为什么还有几微秒（这才是区分度所在）**：

- **`baseLatency` 本身就是 6.6~8.4µs 量级**（[tuning.cc:L168](../src/graph/tuning.cc#L168)），
  这部分是 kernel launch + shmem 加载 + block 调度上 SM 的固定开销；
- kernel launch 到真正跑起来有调度延迟，CUDA Graph 只能省掉 CPU 侧提交，
  **省不掉 GPU 侧的 block 调度**；
- 跨设备可见性需要 `__threadfence_system` 级别的 fence，本身有代价；
- spin 等待有轮询粒度。

所以业界的进一步手段是**persistent kernel / device-side API**（把通信 kernel 常驻，
用户从设备侧直接发起），本仓库中对称内存 / device kernel 相关路径已被裁剪为桩。

→ 详见 [14-latency-optimization](./14-latency-optimization.md)

---

### Q11 为什么 NCCL 要给每个 (算子×算法×协议×算子类型×数据类型) 生成一个独立 kernel？

**做法**：[device/generate.py](../src/device/generate.py) 在编译期生成全部实例，
运行期用 `funcId` 索引 `ncclDevFuncTable`
（[common.h:L430](../src/device/common.h#L430)）。

**为什么值得**：

1. **零运行期分支**：算法、协议、Fan in/out、是否 direct 全部变成模板常量，
   编译器能把不走的路径整块删掉。
2. **常量传播**：`chunkSize`、`SlicePerChunk`、`Unroll` 是编译期常量，
   循环能完全展开，索引计算能折叠。
3. **寄存器分配最优**：每个特化 kernel 的寄存器压力独立优化，
   避免"为了兼容最复杂路径而整体降 occupancy"。
4. `ncclKernelMain` 里还有 `SpecializedFnId` 快路径
   （[common.h:L427](../src/device/common.h#L427)）：命中时连函数指针间接调用都没有。

**代价**：编译时间和库体积（这也是 NCCL 编译很慢的原因），以及 icache 压力。
面试时主动提这个 trade-off 会显得你真的读过。

→ 详见 [08-device-kernel-allreduce](./08-device-kernel-allreduce.md)

---

### Q12 `blockIdx` 是怎么映射到 channel 的？为什么不能直接 `channelId = blockIdx.x`？

因为一个 plan **可能只用部分 channel，且编号不连续**（`plan->channelMask` 是位图）。
`gridDim.x = countOneBits(channelMask)` 是稠密的，需要反查"mask 里第 `blockIdx.x` 个置位"。

[common.h:L378-L381](../src/device/common.h#L378)：

```c
  if (tid < MAXCHANNELS && (args->channelMask & (1ull << tid))) {
    int n = __popcll(args->channelMask & ((1ull << tid) - 1));
    if (blockIdx.x == n) ncclShmem.channelId = tid;
  }
```

**巧妙之处**：这是 `select-nth-set-bit` 问题。PTX 有 `fns` 指令但很慢，
NCCL 利用"所有线程查的是同一个 bitmask"的性质，让**每个线程负责一个 bit 位**，
用 `__popcll` 数出自己前面有几个置位，谁的序号等于 `blockIdx.x` 谁就是答案 ——
**把串行查找变成一次并行 popcount**。

→ 详见 [08-device-kernel-allreduce](./08-device-kernel-allreduce.md)

---

### Q13 为什么一定要 16 字节对齐？

**因为设备侧所有搬运都走 128-bit（16B）向量化 load/store**
（[device/op128.h](../src/device/op128.h) 的 `ld.volatile.global.v2.u64` 之类内联 PTX）。

- 对齐时：1 条指令搬 16B，内存事务完美合并，能吃满 LSU 与链路吞吐。
- 不对齐时：退化成更小粒度（8B/4B）甚至标量循环，**指令数翻数倍，带宽显著下降**。

所以连"尾部数据不足一轮"时缩小 chunk，也要保持对齐
（[all_reduce.h:L72](../src/device/all_reduce.h#L72)）：

```c
    if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16 / sizeof(T));
```

**面试延伸**：这也解释了为什么 LL128 选 128B 而不是别的数字 ——
既是 NVLink 原子写粒度，也刚好是一个 warp 做 16B/线程 × 8 线程的自然边界。

→ 详见 [11-reduce-and-vectorization](./11-reduce-and-vectorization.md)

---

### Q14 `ncclGroupStart/End` 到底优化了什么？

**三件事**：

1. **合并 launch**：group 内多个 collective 进同一个 plan，
   `ncclLaunchKernel` 只调一次 → N 次 launch 开销变 1 次。
2. **合并同步**：一次 kernel 内部完成多个算子，省掉算子之间的 stream 同步。
3. **让 NCCL 有全局视野**：知道总共有多少任务，才能合理分配 channel 与线程。

**实现关键**：所有 collective API 内部都有**隐式 group**
（[enqueue.cc:L3222](../src/enqueue.cc#L3222) 的 `ncclGroupStartInternal`），
只有 group 深度回到 1 时 `ncclGroupEndInternal` 才真正触发 launch。
所以"单独调用"和"group 内调用"走的是同一套代码，只是触发时机不同 —— **设计非常干净**。

**注意点**：同一 group 内的 comm 必须全部 CUDA-graph-capture 或全部不 capture
（[group.cc:L337-L343](../src/group.cc#L337)），否则 comm 会被永久标记为不可用。

→ 详见 [05-enqueue-plan-launch](./05-enqueue-plan-launch.md)

---

### Q15 用户 buffer 注册（user buffer registration）省了什么？

**不注册的路径**：`sendbuff` → 拷进本地通信 buffer（staging）→ 对端 kernel 从
staging 读 → 写对端 recvbuff。中间的 staging 意味着**额外一次显存写 + 一次显存读**。

**注册后**：对端 buffer 的 IPC 指针直接可用，kernel 里的 `directSend/directRecv`
直接读写**对端的用户 buffer**，跳过 staging（这就是原语名字里 `direct` 的含义）。

**收益**：省掉一次显存往返 → 大消息下显存带宽压力显著下降；同时少一份 buffer 显存占用。

**条件**：需要显式注册（或在 CUDA Graph capture 时自动注册）、需要满足对齐与
可映射性要求。不满足就自动退回 staging 路径。

→ 详见 [12-memory-and-registration](./12-memory-and-registration.md)

---

### Q16 NCCL 的性能模型是怎么标定的？为什么不动态测量？

**标定方式**：`ncclTopoTuneModel`（[init.cc:L1673](../src/init.cc#L1673) →
[tuning.cc](../src/graph/tuning.cc)）在 init 时把**硬编码的经验常量表**
（`baseLatencies`、`hwLatencies`、`llMaxBws`、`perChMaxRingLL128Bws`）
按当前拓扑（NVLink/PCI/NET）、架构（Volta/Ampere/Hopper/Blackwell）、
节点数索引出来，算成 `comm->latencies[][]` 和 `comm->bandwidths[][]`。

**为什么不动态测量**：

1. 动态测量要真跑，会引入初始化延迟（训练作业启动时间敏感）；
2. 测量结果受当时负载干扰，不稳定，可能导致同一作业不同 rank 选出不同算法 → **死锁**；
3. 硬件种类有限、行为可预测，查表足够准。

**代价与逃生口**：换新硬件时表可能不准 → 提供 `NCCL_ALGO` / `NCCL_PROTO` /
`NCCL_MIN_NCHANNELS` 等环境变量强制覆盖，以及 **tuner plugin** 接口
（[enqueue.cc:L2212-L2221](../src/enqueue.cc#L2212)）让厂商注入自己的模型。

**这题的加分点**：指出"**所有 rank 必须选出相同的算法/协议**，否则会挂"——
这是查表（确定性）而非测量（非确定性）的**根本原因**。

→ 详见 [04-algo-protocol-tuning](./04-algo-protocol-tuning.md)

---

## 3. 反问环节可以问的问题

聊完技术，这几个问题能显示你真的理解了这套系统：

- 你们在多机场景下 Tree 和 Ring 的实际切换阈值和 NCCL 默认模型差多少？有自己的 tuner plugin 吗？
- 通信 kernel 和计算 kernel 抢 SM 的问题怎么处理的？有用 SM 隔离或 CTA policy 吗？
- 小消息场景有没有上 device-side API / persistent kernel？
- 用户 buffer 注册在你们的训练框架里覆盖率如何？没覆盖的部分瓶颈在哪？

---

## 4. 复习清单（考前 30 分钟）

- [ ] 能不看文档画出 [00 总览](./00-overview-allreduce-lifecycle.md) 的四阶段图
- [ ] 能推导 `2(N-1)/N` 和 `busbw = algbw × 2(N-1)/N`
- [ ] 能说出 `gridDim.x == nChannels` 并解释它的含义
- [ ] 能背出 Ring 的 5 段原语序列，并解释最后一步为什么只收不发
- [ ] 能说清 Simple / LL / LL128 的 flag 机制与有效带宽（100% / 50% / 93.75%）
- [ ] 能说出 `NCCL_STEPS=8` 是流水线深度，以及 head/tail 为什么用单调计数
- [ ] 能解释单机 NVLink 下 proxy 不参与数据面
- [ ] 能列出至少 6 条"打满带宽"的杠杆（[13](./13-bandwidth-saturation.md)）
- [ ] 能解释为什么 NCCL 用查表而不是动态测量（一致性 → 否则死锁）
- [ ] 能说出为什么必须 16B 对齐
