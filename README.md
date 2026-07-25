# mini-nccl（独立精简仓库）

> ## ⭐ 使用 AI Agent 辅助，从 **NVIDIA NCCL 2.30.7** 抽取出的**单机多卡最小通信库**，仅保留 **AllReduce 全链路**，用于**学习**
>
> 本仓库**完全自包含、不依赖原 `/root/nccl`**：从源码 `make` 即可产出 `libnccl.so`，
> nccl-tests 链接它后功能与性能都和原版 NCCL 一致（128MB 档 ~281 GB/s bus bandwidth）。

---

## 一、目录结构与各文件夹职责

### 仓库根
| 文件 | 作用 |
|---|---|
| `Makefile` | 顶层入口。`make lib` 编动态库，`make staticlib` 编静态库 |
| `makefiles/common.mk` | 编译选项、显卡架构(`NVCC_GENCODE`)、优化/警告开关 |
| `makefiles/version.mk` | 版本号（2.30.7） |
| `setup_mini_nccl.sh` | 一键环境配置 + 编译 + 跑通验证脚本 |
| `LICENSE.txt` | NVIDIA 许可证 |
| `src/` | 全部源码（见下） |
| `tests/` | 仓库自带的 all_reduce 验证（取自 nccl-tests，已裁剪为只编译 `all_reduce_perf`，默认链接本仓库 `build/lib`） |
| `build/` | 编译产物（动态库 / 静态库 / 目标文件），由 `make` 生成 |

### `src/` —— 全部源码

**顶层 `.cc`（主机侧核心，AllReduce 链路直接经过）**
| 文件 | 作用 |
|---|---|
| `bootstrap.cc` | **建联**：socket 建环、各 rank 交换 IP/网卡/busId，是所有通信的起点 |
| `init.cc` | comm 初始化，把 bootstrap→graph→transport→kernel 注册串起来 |
| `group.cc` | group 语义：多算子批处理与同步 |
| `enqueue.cc` | **入队与计划(plan)构建**：决定算法(Ring/Tree)、协议(Simple/LL/LL128)、通道数，发起 kernel |
| `channel.cc` | channel（通道）抽象 |
| `collectives.cc` | 集合算子分发入口 |
| `proxy.cc` | **代理线程进度引擎**：网络 / 异步传输的后台推进线程 |
| `transport.cc` | 传输层注册与连接管理 |
| `debug.cc` | 日志输出（`NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS`） |
| `allocator.cc` / `mem_manager.cc` | 设备内存分配与管理 |
| `dev_runtime.cc` / `dev_runtime_segments.cc` | 设备运行时、kernel 启动胶水层 |
| `sym_kernels.cc` 等 | 对称内存 kernel（已禁用，仅桩） |
| `ce_coll.cc` / `mnnvl.cc` / `enhcompat.cc` | 高级特性（已弱化/禁用/兼容层） |

**`src/` 子目录**
| 目录 | 作用 | 能否删 |
|---|---|---|
| `graph/` | **拓扑检测与算法选择**：拓扑探测(topo)、路径(paths)、建环(rings)、建树(trees)、通道搜索(search)、调优(tuning)、XML(connect/xml) | ❌ 核心 |
| `transport/` | 传输层：`p2p.cc`(NVLink P2P，实际走的)、`shm.cc`(共享内存兜底)、`net*.cc`(多机网络，单节点未用)、`nvls.cc`(禁用)、`profiler.cc`(桩) | ⚠️ p2p/shm 必留 |
| `device/` | **设备端 kernel**：`all_reduce.h`(RING/TREE 算法) + `prims_simple.h`/`prims_ll.h`/`prims_ll128.h`(通信原语) | ❌ 核心 |
| `misc/` | 基础设施：socket、CUDA/NVML/GDR 封装、参数校验、共享内存、strongstream(CUDA graph)、utils | ❌ 必留 |
| `include/` | 195 个内部头文件，全模块接口声明，互相交叉引用 | ❌ 碰不得 |
| `os/` | OS 抽象层：线程/事件/IPC socket/桩(linux_*.cc) | ❌ 必留 |
| `param/` | `NCCL_*` 环境变量注册表与 C API | ❌ 必留 |
| `register/` | 内存注册：通用(register)、集合(coll_reg 用户 buffer)、sendrecv | ❌ 必留 |
| `scheduler/` | 集合调度器(AllGatherV/Symmetric)。AllReduce 不直接用，但运行时经函数指针表引用，**删了会 `undefined symbol`** | ❌ 必留 |
| `devcomm/` | device-comm 跨版本兼容结构体(2.29.02/2.29.07/2.30.00) | ❌ 被 dev_runtime 引用 |
| `nccl_device/` | 设备侧辅助：`core.cc`、`ll_a2a.cc`(LL 全交换)、`lsa_barrier.cc`(对齐屏障) | ❌ LL 协议需要 |

---

## 二、你主要该关注哪里（AllReduce 全链路）

你的理解**大方向对**——建联 + 拓扑检测 + kernel 算法 + 通信原语，确实是核心。
但还差**两块**才能真正看懂"数据怎么在两卡间流动"：

```
bootstrap.cc ──建联──► graph/ ──拓扑/算法选择──► enqueue.cc ──计划/入队──┐
                                                                         │
transport/p2p.cc ──GPU 直连搬字节──► proxy.cc ──后台进度──► device/ ──────┘
                                              all_reduce.h(算法)
                                              prims_*(通信原语)
```

1. **建联** `bootstrap.cc`：多进程怎么互相认识、交换连接信息
2. **拓扑检测** `graph/`（topo/rings/trees/tuning）：怎么决定走 Ring 还是 Tree、怎么分通道
3. **入队与计划** `enqueue.cc` / `group.cc`：一次 all_reduce 怎么被拆成 plan、怎么排队下发
4. **传输层** `transport/p2p.cc`（+ `shm.cc` 兜底）：字节实际怎么在两卡间搬（GPU 直连 IPC）—— **你漏的①**
5. **代理进度** `proxy.cc`：异步/网络传输的后台推进线程 —— **你漏的②**
6. **设备算法 + 原语** `device/all_reduce.h`（RING/TREE）+ `device/prims_simple.h`/`prims_ll.h`/`prims_ll128.h`：你提到的"算法+原语"，kernel 内部怎么发/收/归约

> 即：你漏了 **传输层(transport)** 和 **代理/入队(proxy/enqueue)**。否则只知道"算什么"，不知道"数据怎么过去"。

---

## 三、推荐阅读顺序

> `bootstrap.cc` → `graph/`（topo、rings、trees、tuning）→ `enqueue.cc`
> → `transport/p2p.cc` → `proxy.cc` → `device/all_reduce.h` → `device/prims_simple.h`

---

## 四、构建与验证

```bash
# 1) 一键（环境 + 编译 + 跑通）
cd mini-nccl
bash setup_mini_nccl.sh

# 2) 或手动编译库（适配 H20 = sm_90）
make -j$(nproc) lib CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# 3) 仓库自带 tests 验证（无需外部 nccl-tests）
make test
# 等价于下面两条:
#   make -C tests   # 编译 tests/build/all_reduce_perf (默认链接本仓库 build/lib)
#   LD_LIBRARY_PATH=build/lib ./tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
# 期望: Out of bounds values : 0 OK, 128MB 档 busbw ≈ 281 GB/s
```

调试验证技巧：
```bash
# 看建联/拓扑/算法选择过程
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL ./build/all_reduce_perf -b 64M -e 64M -g 2
```
