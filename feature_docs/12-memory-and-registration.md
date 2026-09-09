# 12 · 内存模型与用户 Buffer 注册（为什么能"少拷一次"）

> 本文回答两个常被忽略、但面试必问的问题：
> **(1) NCCL 的通信 buffer 从哪来、多大、怎么切成 step？**
> **(2) 用户自己的 GPU buffer 能不能被 NCCL 直接读写？能省掉哪一次拷贝？**
>
> 上游是 [05 章](./05-enqueue-plan-launch.md)（enqueue 决定 size/通道数）和
> [08 章](./08-device-kernel-allreduce.md)（kernel 决定谁去哪个 buffer 收发）；
> 本层是"数据到底落在哪块显存、以及走不走直写"的落地细节。
> 配套：[11 章](./11-reduce-and-vectorization.md) 讲 load/store 怎么向量化，本层讲 load/store 的**目标地址从哪来**；
> [09 章](./09-primitives-simple.md)/[10 章](./10-primitives-ll-ll128.md) 的 FIFO step 流水正是建在本层的通信 buffer 之上。
>
> 所有行号均已逐条核对本仓库源码。

---

## 本文覆盖的源文件

| 文件 | 在本文中的作用 |
| --- | --- |
| [src/init.cc](../src/init.cc) | `DEFAULT_BUFFSIZE`(4MiB)、`computeBuffSizes` 决定每协议 buffer 大小 |
| [src/device/prims_simple.h](../src/device/prims_simple.h) | `stepSize = buffSizes[SIMPLE]/NCCL_STEPS/sizeof(T)`，FIFO 槽位切分 |
| [src/include/device.h](../src/include/device.h) | `ncclDevWorkColl.regUsed/direct` 位；`NCCL_STEPS` 等常量 |
| [src/register/register.cc](../src/register/register.cc) | `ncclCommRegister`：把用户 buffer 登记进 `regCache` → 后续走直写 |
| [src/transport/p2p.cc](../src/transport/p2p.cc) | P2P/IPC 连接建立，peer 显存映射成 `directBuff` 的基础 |

---

## 主题 1：NCCL 自己的通信 buffer——"中间站"为什么存在

### ① 解决什么问题（场景）

GPU 之间不能直接"知道对方显存地址随便写"——需要先把对端 buffer 通过 IPC 映射成本地可访问的指针，
并且需要一个**所有 rank 都按相同节奏读写、带 step 序号做同步**的收发区。
如果每个 collective 都让用户 buffer 直接互写，会遇到两个问题：
- 用户 buffer 的地址/生命周期/NUMA 属性五花八门，没法保证所有 rank 都 IPC 可映射；
- 没有"中间 buffer"的话，发送节奏只能靠用户 buffer 上的 flag 同步，难以做流水线。

所以 NCCL 在 comm 初始化时**统一分配一块通信 buffer**（per channel、per 协议各一份），
所有收发都先落到这块 buffer，由 prng 原语用 step 序号做带步同步的流水。

### ② 一句话本质

`comm->buffSizes[NCCL_NUM_PROTOCOLS]` 是每种协议一块通信 buffer 的大小（Simple=4MiB）；
这块 buffer 在**每个 channel** 上各分配一份，再按 `NCCL_STEPS`(=8) 切成 8 个等大的"槽（step）"，
每个槽由 `stepSize` 描述容量——这就是 [09/10 章](./09-primitives-simple.md) 里"FIFO step 流水线"的物理载体。

### ③ 代码链路

1. [`init.cc:827`](../src/init.cc#L827) `DEFAULT_BUFFSIZE = (1 << 22)` 即 **4 MiB**（Simple 协议默认）
2. [`init.cc:824-838`](../src/init.cc#L824) `DEFAULT_LL_BUFFSIZE` / `DEFAULT_LL128_BUFFSIZE` 另两种协议各自默认大小
3. [`init.cc:836-842`](../src/init.cc#L836) `computeBuffSizes`：可被环境变量 `NCCL_BUFFSIZE`/`NCCL_LL_BUFFSIZE`/`NCCL_LL128_BUFFSIZE` 覆盖
4. [`init.cc:618`](../src/init.cc#L618) 把 `buffSizes[p]` 拷到设备侧 `tmpCommAndChans.comm.buffSizes[p]`（device 可见）
5. [`prims_simple.h:623`](../src/device/prims_simple.h#L623) `stepSize = buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T)`
6. [`prims_simple.h:184/192/197`](../src/device/prims_simple.h#L184) 收发时按 `step % NCCL_STEPS` 索引槽位：`connEltsFifo + (step % NCCL_STEPS) * connStepSize`

### ④ 关键代码逐行解读

`computeBuffSizes`（[`init.cc:836-842`](../src/init.cc#L836)）：

```c
static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = {ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize()};
  int defaults[NCCL_NUM_PROTOCOLS] = {DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE};
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];  // 默认 4MiB（Simple）
  }
  // LL/LL128 的默认大小由硬件常量推导（每线程线数 × 线程数 × 步数 × 单元大小）
  ...
}
```

- **三种协议各一块 buffer**：`NCCL_PROTO_LL`(默认 `NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(ncclLLFifoLine)`)、
  `NCCL_PROTO_LL128`(类似，但按 `uint64_t` 计)、`NCCL_PROTO_SIMPLE`(4MiB)。
  为什么要分开？因为 LL/LL128 的"每步携带 flag"布局与 Simple 的纯数据布局不同，不能共用一块 buffer。
- **环境变量可覆盖**：`NCCL_BUFFSIZE` 调大 → 单步能搬更多数据 → 减少 step 数 → 降低同步开销；
  调小 → 省显存。4MiB 是"在显存占用与步数间取平衡"的默认甜点。

`stepSize` 的切分（[`prims_simple.h:623`](../src/device/prims_simple.h#L623)）：

```c
stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T) : stepSize_)
```

- `NCCL_STEPS = 8`，所以 Simple 的 4MiB 被切成 **8 个 512 KiB 的槽**。
- 为什么是 8 个？环形/树形算法要求"发送方写完第 k 步、接收方读第 k 步"的流水：
  8 个槽让发送方可以**超前写 8 步而不覆盖还没被读走的旧数据**，从而把"等对端确认"的空窗藏起来（详见 [09 章](./09-primitives-simple.md) 的 FIFO 流水）。
- 收发的物理索引就是 `step % NCCL_STEPS`（[`prims_simple.h:184`](../src/device/prims_simple.h#L184) 等），
  配合 `connFifo[step % NCCL_STEPS].size` 做"本步写了多少"的发布信号。

### ⑤ 面试考点

**Q：NCCL 的通信 buffer 多大？为什么默认 4MiB？**
A：Simple 协议默认 4MiB（`DEFAULT_BUFFSIZE`，可被 `NCCL_BUFFSIZE` 改）。4MiB 切成 8 步 = 每步 512KiB，
既能让单步搬足够多数据压满链路突发窗口，又不会因 buffer 过大浪费显存；8 步的流水深度足以掩盖跨卡同步延迟。

**Q：`NCCL_STEPS` 是谁在用？**
A：所有协议的收发原语都用 `step % NCCL_STEPS` 在通信 buffer 上循环取槽，实现"写第 k 步 / 读第 k 步"的带步同步流水。
步数越多，可掩盖的同步延迟越大，但 buffer 占用与索引位宽也越大，8 是权衡值。

---

## 主题 2：用户 Buffer 注册——把"中间站"那一次拷贝省掉

### ① 解决什么问题（场景）

默认路径里，一次 AllReduce 的数据流是：
**用户 buffer → 拷进 NCCL 通信 buffer → 跨卡收发（在通信 buffer 之间）→ 拷回用户 buffer**。
中间那两次"用户 buffer ↔ 通信 buffer"的拷贝是纯开销。
如果用户能保证自己的 buffer 生命周期稳定、且地址可被对端 GPU 直接访问（IPC/GDR），
**NCCL 就可以跳过中间站，直接把归约结果写到用户的 buffer（或直接从用户 buffer 读）**。

### ② 一句话本质

`ncclCommRegister` 把用户 GPU buffer 登记进 `comm->regCache`，标记为"可被对端直接访问"；
之后该 comm 上的 collective 在生成 `ncclDevWorkColl` 时打上 `regUsed=1` 位，
设备端 `Primitives::process()` 检测到 `regUsed`/`ipcRegFlag` 后，把 load/store 目标从"通信 buffer"改成
**`directBuff`（用户 buffer 的 IPC 映射地址）**，走 `DirectWrite/DirectRead` 直写，省掉中间那两次拷贝。

### ③ 代码链路

1. [`register/register.cc:162`](../src/register/register.cc#L162) `ncclCommRegister`：登记用户 buffer 进 `comm->regCache`
2. [`register/register.cc:36-41`](../src/register/register.cc#L36) 按页对齐、按 `cuMem`/`cuPointerGetAttribute` 判定 host/device 段
3. [`device.h:299`](../src/include/device.h#L299) `ncclDevWorkColl` 的 `regUsed:1` 位（由 enqueue 依据注册表填）
4. [`prims_simple.h:78`](../src/device/prims_simple.h#L78) `T* directBuff`：直连缓冲区（对端可直接读写的用户显存地址）
5. [`prims_simple.h:383`](../src/device/prims_simple.h#L383) `if (Direct && fn.work->regUsed)` → 把 `ptrs[index]` 指向 `directBuff`
6. [`prims_simple.h:546-554`](../src/device/prims_simple.h#L546) `loadRecvConn`：注册/IPC 场景下置 `DirectWrite/DirectRead` 标志

### ④ 关键代码逐行解读

`ncclDevWorkColl` 的注册位（[`device.h:299`](../src/include/device.h#L299)）：

```c
struct alignas(16) ncclDevWorkColl {
  ...
  uint32_t redOpArgIsPtr:1, regUsed:1, netRegUsed:1, oneNode:1, direct:2, isOneRPN:1;
  ...
};
```

- `regUsed` 是 **1 个 bit** 的标志：用户对该次 collective 的输入/输出 buffer 做了注册，则置 1。
  它和 `direct:2`（算法层的 direct 模式，如 Ring 的 direct 变体）是两个不同的概念——
  `direct` 是"算法是否让某条连接直连对端"，`regUsed` 是"用户 buffer 是否已注册可被直写"。

设备端如何据此切换目标地址（[`prims_simple.h:383-397`](../src/device/prims_simple.h#L383)）：

```c
} else if (Direct && fn.work->regUsed) {
  if (flags & DirectWrite) {
    ptrs[index] = directBuff;          // 直接写到用户的 buffer（对端视角）
  } else if (flags & DirectRead) {
    ptrs[index] = directBuff;          // 直接从用户 buffer 读
  }
  ...
}
```

- 普通（非注册）路径下，`ptrs[index]` 是 `connEltsFifo + (step % NCCL_STEPS) * connStepSize`，即通信 buffer 的某个槽。
- 注册路径下，`ptrs[index] = directBuff`——**归约/拷贝直接发生在用户的 buffer 上**。
  结合 [11 章](./11-reduce-and-vectorization.md) 的 `reduceCopyPacks`，这意味着 `ld/st.global.v2.u64` 直接打在用户显存地址上，
  中间的"用户 buffer ↔ 通信 buffer"两次拷贝被彻底省掉。

`loadRecvConn` 里如何决定 `DirectWrite/DirectRead`（[`prims_simple.h:546-554`](../src/device/prims_simple.h#L546)）：

```c
if (Direct) {
  if (ipcRegFlag) {                       // 用户已注册 → IPC 映射可达
    ...
    flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
  } else if (connIndex == 1 && direct) {
    flags |= DirectRead;                  // 算法层 direct 模式（如 tree 的直连拉取）
  } else {
    flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
  }
  ...
}
```

- `ipcRegFlag` 来自 `p2pWork->recvIpcReg/sendIpcReg`（[`prims_simple.h:682-685`](../src/device/prims_simple.h#L682)），
  由主机侧依据注册表设置；它和 `regUsed` 共同决定走直写还是直读。
- 注意：**mini-nccl 的 P2P 连接本来就是 NVLink+IPC 直连**（见 [06 章](./06-transport-p2p-shm.md)），
  但"对端 GPU 的通信 buffer"与"用户的 buffer"仍是两块不同的显存；注册打通的是后者，让数据不必先落到 NCCL 的中间 buffer。

### ⑤ 面试考点

**Q：注册用户 buffer 到底省了什么？**
A：省掉"用户 buffer ↔ NCCL 通信 buffer"的两次中转拷贝。注册后设备端 `Primitives` 的 load/store 直接打在用户的 IPC 映射地址（`directBuff`）上，归约原地完成。

**Q：`regUsed` 和 `direct` 是一回事吗？**
A：不是。`direct`（2 bit）是算法层的 direct 变体（比如 Ring/Trees 某些连接直连对端通信 buffer）；`regUsed`（1 bit）是"用户 buffer 已注册、可被直写"的标记。两者可以叠加：既走算法直连、又走用户 buffer 直写，省得最彻底。

**Q：注册有没有代价/限制？**
A：有。注册本身要把 buffer 钉页/Pin、建立 IPC 映射、维护引用计数（`ncclRegCache`，[`register.cc:36`](../src/register/register.cc#L36)），适合"同一块大 buffer 反复做很多次 collective"的场景（如训练循环里的梯度 AllReduce）；一次性小 buffer 注册反而得不偿失。另外 buffer 生命周期必须覆盖所有相关 collective，否则对端会访问到已释放显存。

---

## 主题 3：两种 buffer 的关系（一张对照表收尾）

| 维度 | NCCL 通信 buffer（中间站） | 用户 buffer（注册后直写） |
| --- | --- | --- |
| 来源 | `computeBuffSizes` 在 comm init 分配（4MiB/协议/channel） | 用户自己 `cudaMalloc` 的显存 |
| 生命周期 | 跟随 comm | 跟随用户，注册期需覆盖 collective |
| 访问方式（默认） | 所有收发都落在它的 step 槽里 | 数据先拷进/拷出它 |
| 注册后 | 可完全绕过 | `directBuff` 直写，归约原地完成 |
| 关键代码 | [`init.cc:836`](../src/init.cc#L836)、[`prims_simple.h:623`](../src/device/prims_simple.h#L623) | [`register.cc:162`](../src/register/register.cc#L162)、[`prims_simple.h:383`](../src/device/prims_simple.h#L383) |

**和整条链路的呼应**：
- [05 章](./05-enqueue-plan-launch.md) 的 size/channel 决策，决定了"一次 collective 要切多少 step、用哪块 buffer"；
- 本层提供"buffer 从哪来、多大、怎么切槽"以及"注册后能否直写"的落地；
- [08/09/10/11 章](./08-device-kernel-allreduce.md) 在这个 buffer 之上做算法、原语、向量化归约。
- 注册省掉的那两次拷贝，正是 [13 章](./13-bandwidth-saturation.md) 里"小消息延迟"和"大消息带宽"都能再进一步的关键开关之一。

---

## 4. 面试速记卡

- **通信 buffer**：`computeBuffSizes` 给每协议分一块（Simple=4MiB），每 channel 一份，按 `NCCL_STEPS=8` 切 8 槽，`stepSize = 4MiB/8/sizeof(T)`。
- **为什么 8 槽**：发送可超前写 8 步不覆盖未读数据，用流水深度掩盖跨卡同步延迟。
- **注册省拷贝**：`ncclCommRegister` → work 带 `regUsed=1` → 设备端 `directBuff` 直写用户显存，跳过"用户↔通信 buffer"两次中转。
- **`regUsed` ≠ `direct`**：前者是用户 buffer 已注册；后者是算法层直连变体。可叠加。
- **代价**：注册要 Pin/IPC/引用计数，仅适合反复使用的大 buffer。
