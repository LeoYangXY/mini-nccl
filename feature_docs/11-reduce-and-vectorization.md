# 11 · 归约与向量化访存（GPU 上真正"算 + 搬"的那一层）

> 这是 mini-nccl 里 **"SM 到底怎么把字节读进来、做求和、再写出去"** 的那一层。
> 上游是 [08 章](./08-device-kernel-allreduce.md) 的设备端 AllReduce kernel（它决定"数据从哪来、到哪去"），
> 本层是它的"搬运 + 计算"原语。
> 通用归约/拷贝代码在 [`common_kernel.h`](../src/device/common_kernel.h)，
> 128-bit 打包访存在 [`op128.h`](../src/device/op128.h)，
> 求和算子在 [`reduce_kernel.h`](../src/device/reduce_kernel.h)。
>
> 所有行号均已逐条核对本仓库源码。

---

## 本文覆盖的源文件

| 文件 | 在本文中的作用 |
| --- | --- |
| [src/device/op128.h](../src/device/op128.h) | `BytePack<N>` 字节打包 union、`ld/st_global<16>` 128-bit 访存、内存序 fence |
| [src/device/common_kernel.h](../src/device/common_kernel.h) | `reduceCopyPacks` / `reduceCopy`：多源多目标一次遍历的 load→reduce→store |
| [src/device/reduce_kernel.h](../src/device/reduce_kernel.h) | `FuncSum`/`FuncProd` 等归约算子、`Apply_Reduce` 特化、`SPECIALIZE_REDUCE` 宏 |
| [src/device/common.h](../src/device/common.h) | `COLL_UNROLL` 宏（编译期/运行期展开因子） |
| [src/include/device.h](../src/include/device.h) | `WARP_SIZE`、协议线程数等常量 |

---

## 主题 1：为什么需要"打包"——128-bit 向量化访存

### ① 解决什么问题（场景）

GPU 的"现实"是：**内存带宽极大，但每个 `ld.global`/`st.global` 指令默认只搬 4 字节（一个 `int`/`float`）**，
而且指令发射有固定的发射间隔、warp 内 32 线程的访存要能被合并（coalesce）才能打满一条 128-byte 的 L1/L2 段。

对 AllReduce 来说，每个元素都要被 load（从对端/自己 buffer）→ reduce（求和）→ store（写到对端/自己 buffer）。
如果按 4 字节一个一个搬，指令数直接 ×4，发射端口和延迟都会被浪费，链路带宽根本喂不满。
**所以 NCCL 的核心访存优化就是：一次 load/store 尽可能多的字节，并让这 32 次访存对齐合并。**

### ② 一句话本质

`BytePack<N>` 把 N 个字节拼成一个 `union`，让编译器能一次 `(ld/st).global.v4.u32`（128-bit，16 字节）把数据整块搬进寄存器文件；
归约函数则在寄存器内对这个 `BytePack` 整体做类型重解释 + SIMD 式加法，**访存与计算都按"块"进行**。

### ③ 代码链路

1. [`op128.h:96-144`](../src/device/op128.h#L96) `BytePack<N>` 模板 union 定义
2. [`op128.h:131`](../src/device/op128.h#L131) `BytePack<16>` 为 `union alignas(16) BytePack<16>`，强制 16 字节对齐
3. [`op128.h` 的 `ld_global<16>` / `st_global<16>`](../src/device/op128.h) 128-bit 访存内建封装
4. [`op128.h:400`](../src/device/op128.h#L400) `fence_acq_rel_sys()` 跨卡内存序 fence
5. [`common_kernel.h:48`](../src/device/common_kernel.h#L48) `BytePerHunk = Unroll * WARP_SIZE * BytePerPack` 一次迭代搬运量

### ④ 关键代码逐行解读

`BytePack` 的定义骨架（[`op128.h:96-144`](../src/device/op128.h#L96)）：

```c
template<int N> union BytePack {};          // 通用模板（仅声明）
union BytePack<1> { uint8_t u8; int8_t i8; /* ... */ };
union BytePack<2> { uint16_t u16; half h; /* ... */ };
union BytePack<4> { uint32_t u32; float f; /* ... */ };
union BytePack<8> { uint64_t u64; double d; /* ... */ };
union alignas(16) BytePack<16> {            // L131：16 字节版本必须 16B 对齐
  uint64_t u64[2];                          // 用两个 64-bit 寄存器表示一个 128-bit 包
  uint32_t u32[4];
  // ...
};
```

- `BytePack<N>` 是 **字节级 union**：同一片内存可以同时被当作 `float`、`half2`、`uint64_t[2]` 来读。
  这让"从显存搬字节"和"在寄存器里做类型相关的加法"能共用同一份寄存器数据，不需要额外的 `reinterpret_cast` 指令。
- `BytePack<16>` 上的 `alignas(16)` 很关键：**128-bit load/store 要求地址 16 字节对齐**。
  若不对齐，要么触发 CUDA trap 报错，要么被硬件拆成多次窄访存（带宽直接腰斩）。
  NCCL 通信 buffer 的分配（见 [12 章](./12-memory-and-registration.md)）和 `stepSize` 切分都保证 16B 对齐。
- 真正发出 128-bit 访存的是 `ld_global<16>`/`st_global<16>`：它们编译成
  `ld.global.v2.u64` / `st.global.v2.u64`（一次搬 128 位），配合 `fence_acq_rel_sys()` 保证跨 SM/跨卡的可见性顺序。

**为什么这能打满带宽（定量）**：
`reduceCopyPacks` 里 `BytePerHunk = Unroll * WARP_SIZE * BytePerPack`（[`common_kernel.h:48`](../src/device/common_kernel.h#L48)）。
- 取 `BytePerPack = 16`（128-bit）、`Unroll = 8`（sm_90 下 `COLL_UNROLL` 实际取值，见 [`common.h:25`](../src/device/common.h#L25)）、`WARP_SIZE = 32`：
  **一个 warp 一次迭代就搬 `8 × 32 × 16 = 4096` 字节（4 KiB）**。
- 32 个线程每个发一条 `ld.global.v2.u64`，正好拼成 32×16 = 512 字节/warp-step；
  8 次 unroll 就是 4 KiB；多个 warp 叠加，单 block 一次就把一条 NVLink 的突发段喂满。
- 关键：**一次 load 16 字节 vs 一次 load 4 字节，指令数降为 1/4**，发射端口不再成为瓶颈。

### ⑤ 面试考点

**Q：NCCL 怎么做到"向量化"访问的？为什么是 16 字节而不是 8 或 32？**
A：用 `BytePack<16>` 把 16 字节拼成 `u64[2]`，配合 `ld/st.global.v2.u64` 一次搬 128-bit。
16 字节是 GPU 对 `.v2.u64` 支持且能与 128-byte L2 段（= 8 个 16B 包）对齐的"甜点"；
32 字节（`.v4.u64`）寄存器压力大、且一次 32B × 32 线程 = 1KiB 已远超大多 cache 行收益，反而降低并发度。

**Q：不对齐会怎样？NCCL 怎么保证对齐？**
A：128-bit 访存要求 16B 对齐，不对齐会 trap 或降速。NCCL 靠两点保证：
通信 buffer 由 `computeBuffSizes` 按 16B 对齐分配；`stepSize` 切分时是 `buffSize/NCCL_STEPS` 且元素数对齐到 pack 边界。

---

## 主题 2：`reduceCopyPacks` —— 多源多目标一次遍历完成归约

### ① 解决什么问题（场景）

AllReduce 的 Ring/Trees 里，一个 warp 经常要 **从一个源（自己的输入）读、再从另一个源（对端收到的部分结果）读，
把两者 reduce，再写到输出（自己输出 + 可能还要发给下游）**。
朴素做法是：先读 A→算，再读 B→算，再写 C。这样访存被拆成多趟、无法流水。

### ② 一句话本质

`reduceCopyPacks` 把 **多个源（MinSrcs/MaxSrcs）和多个目标（MinDsts/MaxDsts）的 load/reduce/store
合并进同一段 unroll 循环**：每个 unroll 步内"把所有源 load 进来 → reduce 进 `acc` → 把所有目标 store 出去"，
让访存与计算重叠，且多源多目标共享同一次遍历。

### ③ 代码链路

1. [`common_kernel.h:38-49`](../src/device/common_kernel.h#L38) 模板签名与 `BytePerHunk` 计算
2. [`common_kernel.h:85-191`](../src/device/common_kernel.h#L85) 主 unroll 循环：load 多源 → reduce → store 多目标
3. [`common_kernel.h:206-258`](../src/device/common_kernel.h#L206) `reduceCopy` 外层：按 `T` 的 size 选 `BytePerPack` 与 `Unroll`

### ④ 关键代码逐行解读

主循环骨架（节选自 [`common_kernel.h:85-173`](../src/device/common_kernel.h#L85)）：

```c
while (Unroll == 1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
  BytePack<BytePerPack> acc[Unroll];
  // —— (1) 第一个源（通常是"自己的输入"或 multimem 加载）做初始值 ——
  for (int u = 0; u < Unroll; u++) {
    if (MultimemSrcs > 0)
      acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]); // 走 multimem 聚合
    else
      acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);               // 普通 128-bit load
    minSrcs[0] += WARP_SIZE * BytePerPack;
  }
  // —— (2) 其余源：load 进来后逐元素 reduce 进 acc ——
  for (int s = 1; s < MinSrcs; s++) {
    BytePack<BytePerPack> tmp[Unroll];
    for (int u = 0; u < Unroll; u++) {
      tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
      minSrcs[s] += WARP_SIZE * BytePerPack;
    }
    for (int u = 0; u < Unroll; u++)
      acc[u] = applyReduce<RedFn, BytePerPack>(redFn, acc[u], tmp[u]);     // 放进 acc
  }
  // —— (3) 可选 postOp（如 Avg 的除法） ——
  if (postOp) for (int u = 0; u < Unroll; u++) acc[u] = applyPostOp(redFn, acc[u]);
  // —— (4) 写到所有目标（自己的输出 + 可能发给下游） ——
  for (int d = 0; d < MinDsts; d++) {
    for (int u = 0; u < Unroll; u++) {
      st_global<BytePerPack>(minDsts[d], acc[u]);
      minDsts[d] += WARP_SIZE * BytePerPack;
    }
  }
  nWarps = nThreads / WARP_SIZE;
  // 跨 warp 推进：每个 warp 各搬了一块 BytePerHunk，整体前进
  for (int s = 0; s < MinSrcs; s++) minSrcs[s] += (nWarps - 1) * BytePerHunk;
  for (int d = 0; d < MinDsts; d++) minDsts[d] += (nWarps - 1) * BytePerHunk;
}
```

- **load→reduce→store 三段都在同一个 unroll 里**：编译器可以把 `ld` 的延迟用后续 `reduce`/`st` 掩盖（指令级并行），
  访存延迟不再"空等"。
- `acc[u]` 是寄存器里的归约累加器；`MinSrcs` 个源都 reduce 到它，最后 `MinDsts` 个目标直接 store 同一份 `acc`——
  **一份数据只在寄存器里算一次，就被写到多个目的地**，省掉了"先写到中间 buffer 再读出来转发"的额外往返。
- 外层 `reduceCopy`（[`common_kernel.h:206-258`](../src/device/common_kernel.h#L206)）按元素宽度选 pack：
  `T` 是 `float` 时 `BytePerPack = sizeof(float) = 4`，但可叠到 16；对 half 会直接走 `Unroll * (16/sizeof(T)) / 2` 去尽量凑满 16B。

### ⑤ 面试考点

**Q：为什么 reduce 和 copy 要放在同一个 kernel 原语里，而不是"先 reduce 完再 copy"？**
A：AllReduce 的环形/树形每一步都是"收到一部分→和本地合并→再发下一部分"。合并（reduce）与转发（copy）必须交织，
否则要多一次完整 buffer 读写。NCCL 用 `reduceCopyPacks` 在寄存器内一次性完成"多源 reduce + 多目标 store"，
既省显存往返，又让 load 延迟被 reduce/store 掩盖。

**Q：`Unroll` 是什么？为什么需要它？**
A：`Unroll` 是编译期展开的迭代次数（sm_90 实际运行时取 8，见 `COLL_UNROLL`）。展开能让：
(1) 更多独立的 `ld/st` 指令同时驻留发射队列，提升访存并行度；(2) 减少循环分支开销；
(3) 一个 warp 一次迭代就搬 `Unroll×32×BytePerPack` 字节（8×32×16=4KiB），瞬间填满链路的突发窗口。

---

## 主题 3：归约算子本身——`FuncSum` 怎么"一条指令算两个"

### ① 解决什么问题（场景）

AllReduce 的核心是"求和"。但 `float` 是 4 字节、`half` 是 2 字节——如果按 `float` 一个一个加，
half 这种窄类型不仅带宽利用率低，而且 GPU 的 half 加法是成对（`half2`）才有峰值吞吐的。

### ② 一句话本质

NCCL 把归约做成 **`BytePack` 上的 trait（`Apply_Reduce`）**，按元素类型特化：
`half`/`__nv_bfloat16` 用 `half2`/`__nv_bfloat162` 打包，一条 `__hadd2` 同时算两个元素，
**算术吞吐翻倍，且和上面的 16 字节向量化访存天然对齐**（16B = 八个 half = 四个 half2）。

### ③ 代码链路

1. [`reduce_kernel.h:54`](../src/device/reduce_kernel.h#L54) `struct FuncSum` 求和算子
2. [`reduce_kernel.h:312-368`](../src/device/reduce_kernel.h#L312) `Apply_Reduce` 递归特化（按 `EltPerPack` 二分）
3. [`reduce_kernel.h:433-468`](../src/device/reduce_kernel.h#L433) `SPECIALIZE_REDUCE` 宏：`half2=__hadd2`、`__nv_bfloat162=__hadd2`

### ④ 关键代码逐行解读

`SPECIALIZE_REDUCE` 对 half 的特化（[`reduce_kernel.h:433-437`](../src/device/reduce_kernel.h#L433)）：

```c
SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd(x, y))      // 注意：宏展开后 EltPerPack=2
```

- 这里 `EltPerPack=2` 表示**一次归约处理 2 个 half**；底层调 `__hadd2(x, y)`（`x/y` 是 `half2`，
  即两个 half 打包在一个 32-bit 寄存器里），**一条指令完成两个元素的求和**。
- `__nv_bfloat16` 走完全对称的逻辑（[`reduce_kernel.h:460`](../src/device/reduce_kernel.h#L460)：`__nv_bfloat162` + `__hadd2`）。
- `Apply_Reduce<Fn, EltPerPack>` 是递归模板（[`reduce_kernel.h:327-331`](../src/device/reduce_kernel.h#L327)）：
  当 `EltPerPack>1` 时把 half 数组拆成 `half[0]`/`half[1]` 两份各调 `EltPerPack/2` 的子特化，
  最终落到 `EltPerPack=1` 的 `FuncSum<T>::reduce`。这样 `BytePack<16>`（=8 个 half）会被自动展开成 4 次 `__hadd2`。

**和主题 1/2 的呼应**：`reduceCopyPacks` 以 `BytePack<16>`（16 字节）为单位 load/store，
而 half 的 `Apply_Reduce` 以 2 字节（half）为单位 reduce——**16B 正好容纳 8 个 half = 4 个 half2**，
向量化访存与向量化算术严丝合缝，没有任何"拆包"浪费。

### ⑤ 面试考点

**Q：NCCL 处理 half/bf16 的求和有什么特别？**
A：不是按单个 half 加，而是用 `half2`/`__nv_bfloat162` 打包 + `__hadd2`，一条指令算两个元素，
算术吞吐翻倍；同时 16 字节的向量化访存正好对应 8 个 half，访存与计算的对齐是"免费"的。

**Q：`ncclAvg` 怎么实现？**
A：先按 `FuncSum` 求和（`FuncSumPostDiv` 在 [`reduce_kernel.h:815`](../src/device/reduce_kernel.h#L815) 直接复用 `FuncSum` 的 reduce），
最后在 `postOp` 阶段除以 rank 数（`Apply_PostOp<FuncSumPostDiv>`），所以平均是"先求和再除一次"，避免每步都除带来的精度损失。

---

## 主题 4：本层在整条链路里的位置（收尾）

| 层次 | 职责 | 本章对应的"钩子" |
| --- | --- | --- |
| [08 章](./08-device-kernel-allreduce.md) 算法层 | 决定环形/树形、每个 block 负责哪块数据、调谁收发 | 把"源/目标指针 + 字节数"交给本层 |
| **本章（11）归约/访存层** | 真正 `ld/st.global.v2.u64` + `FuncSum` 求和 | `reduceCopyPacks` / `BytePack` / `FuncSum` |
| [09/10 章](./09-primitives-simple.md) 原语层 | 决定数据走"内部 buffer 的 step FIFO"还是"注册 buffer 直写" | 提供 `directBuff`/`regUsed` 给本层当目标指针 |

**三个优化点的叠加效果**：
1. 向量化访存（16B/指令）→ 指令数 ↓4×
2. unroll + 多源多目标一次遍历 → 访存延迟被计算掩盖，显存往返 ↓
3. half2/bf16x2 打包求和 → 算术吞吐 ×2

三者共同把"每个元素的 load→reduce→store"成本压到接近硬件理论下限，这正是 [13 章](./13-bandwidth-saturation.md)
"能打满带宽"的底层原因之一。

---

## 6. 面试速记卡

- **向量化**：`BytePack<16>` + `ld/st.global.v2.u64`，一次 16 字节；需 16B 对齐，靠 buffer 分配与 `stepSize` 切分保证。
- **一次搬多少**：`Unroll(8) × WARP_SIZE(32) × BytePerPack(16) = 4KiB`/warp/iter（sm_90）。
- **减少往返**：`reduceCopyPacks` 在同一 unroll 内完成"多源 load→寄存器 reduce→多目标 store"。
- **算术翻倍**：half/bf16 用 `half2`+`__hadd2`，一条指令算两元素；与 16B 访存天然对齐。
- **Avg**：先 `FuncSum` 求和、最后 `postOp` 除一次，保精度。
