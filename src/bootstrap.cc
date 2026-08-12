/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * bootstrap.cc —— NCCL 引导/建连控制面（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 本文件在 AllReduce 全链路中的定位：
 *   它是通信域 ncclComm 建立的第一步（“引导”阶段），在用户还没发起任何
 *   ncclAllReduce 之前，先把所有参与通信的进程/GPU 用一张基于 socket 的控制面
 *   网络连起来——这张网络不经过 NCCL 自己的 GPU 传输，只用来交换拓扑与连接信息。
 *
 * 主要职责：
 *   1) bootstrapGetUniqueId  : 生成一个唯一 handle（magic + 监听地址），多进程靠它
 *                              汇聚到同一个 root 节点。
 *   2) bootstrapInit         : 核心建连流程——建立 ring 环网、收集每个 rank 的
 *                              peerP2pAddresses / peerProxyAddresses / UDS 等地址信息，
 *                              初始化 proxy 与 RAS。后续 ring/tree 拓扑与 GPU 直连都依赖它。
 *   3) bootstrapAllGather    : 把所有节点的地址/信息收集到每个 rank（ring 算法实现）。
 *   4) bootstrapSend/Recv    : 控制面的点对点数据交换（含 unexpected 连接队列）。
 *   5) bootstrapBarrier/Broadcast : 进程同步与广播（dissemination 算法）。
 *
 * 关键概念：
 *   - root 线程：每个进程派生一个 bootstrapRoot 线程，负责接收所有 rank 的建连请求、
 *     收集上报信息、回发 next-peer 邻接关系。
 *   - ring 环网：所有 rank 连成一条环，bootstrap 用“先发长度再发数据”的 socket 协议
 *     完成环形 AllGather，这是收集拓扑信息的高效方式。
 *   - 多 root：支持按 NCCL 配置把 nranks 分成多个 root 组（firstRankFromRoot 等映射）。
 * ============================================================================
 */

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "proxy.h"
#include "param.h"
#include "ras.h"
#include <mutex>
#include "os.h"
#include <thread>
#include <chrono>

#define BOOTSTRAP_N_CHECK_ABORT 10000
#define BOOTSTRAP_TAG_CONNECT (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)
#define BOOTSTRAP_TAG_GROW_BOUNDARY (0x1 << 27)

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND 1
#define BOOTSTRAP_INIT_TIME_RECV 2
#define BOOTSTRAP_INIT_TIME_RING 3
#define BOOTSTRAP_INIT_TIME_TOTAL 4
#define BOOTSTRAP_INIT_TIME_DELAY 5
#define BOOTSTRAP_INIT_TIME_N 6
#define BOOTSTRAP_INIT_ROOT_WAIT 0
#define BOOTSTRAP_INIT_ROOT_SEND 1
#define BOOTSTRAP_INIT_ROOT_RECV 2
#define BOOTSTRAP_INIT_ROOT_N 3
#define BOOTSTRAP_PROF_OPEN(time) \
  do { \
    time = clockNano(); \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do { \
    time = clockNano() - time; \
  } while (0)

/* ============================================================================
 * 多根节点(multi-root)的 rank 分配辅助函数
 * ----------------------------------------------------------------------------
 * 背景：bootstrap 阶段所有 rank 都要向“根(root)”上报自己的连接信息。如果规模很大
 *   (成千上万个 rank)，单个 root 会成为瓶颈甚至被连接风暴打挂。因此 NCCL 支持把
 *   rank 分给 nRoots 个根来分担压力。
 *
 * 分配规则(尽可能均匀)：设有效 rank 数 N = nRanks - offset，根数为 R，
 *   则 rpr = N/R (每个根的基础份额)，rmr = N%R (余数)。
 *   前 rmr 个根各带 rpr+1 个 rank，其余根各带 rpr 个 rank。
 *   例如 N=10, R=3 => 分配为 [4, 3, 3]。
 *
 * offset 的作用：用于 comm “扩容(grow)”场景——前 offset 个 rank 是已存在的老成员，
 *   不参与本次分配，只对 offset 之后的新 rank 做划分。
 * ============================================================================ */

// 周期化取模：把下标 i 折算到 [0, n) 区间(先加 n 再取模，兼容负数输入)
#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))

// 返回归属于某个 根 的第一个(最小的) rank 编号，要求 根 >= 0。
// 注意：若 根 >= nRoots，本函数不做周期化处理(不会自动折回)。
static int firstRankFromRoot(int root, int n_ranks, int nRoots, int offset) {
  if (root == -1) return 0;
  // 只把 n_ranks - 偏移 这部分 rank 分配给各个 根
  n_ranks -= offset;
  // 根 * rpr 是前面各根按基础份额占用的数量；
  // 最小值(根, rmr) 是前面的根中“多拿了 1 个”的那些根的个数，需要额外累加。
  return offset + root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}

// 反向查询：给定 rank，返回它归属哪个 根，要求 rank >= 0。
// 注意：若 rank >= nRanks，本函数不做周期化处理。
static int rootIdFromRank(int rank, int nRanks, int nRoots, int offset) {
  // 小于 偏移 的 rank 属于“老成员”，没有 根(返回 -1)；偏移 之后的才参与分配
  if (nRoots == 0 || rank < offset) return -1;
  nRanks -= offset;
  rank -= offset;
  int rmr = nRanks % nRoots; // rank mod root：余数，即“多带一个 rank”的根的个数
  int rpr = nRanks / nRoots; // rank per root：每个根的基础份额
  int D = rmr * (rpr + 1);   // 前 rmr 个“大根”一共覆盖的 rank 数量，即分界点
  // 分界点之前：每个根带 rpr+1 个，直接整除即可定位
  if (rank < D) return rank / (rpr + 1);
  // 分界点之后：每个根带 rpr 个，先减去分界点再整除，最后补上前面 rmr 个根的偏移
  else return (rank - D) / rpr + rmr;
}

// 返回某个 根 名下挂了多少个 rank(即它的“孩子”数量)，根 会被周期化处理。
static int nRankFromRoot(int root, int nRanks, int nRoots, int offset) {
  if (root == -1) return 0;
  nRanks -= offset;
  int ir = BOOTSTRAP_PID(root, nRoots);
  int rmr = nRanks % nRoots; // rank mod root：余数
  int rpr = nRanks / nRoots; // rank per root：基础份额
  // 前 rmr 个根多分到 1 个 rank
  return rpr + ((ir < rmr) ? 1 : 0);
}

// 返回某个 rank 在其所属 根 内部的“局部编号”(从 0 开始计数)。
// 根 会被周期化处理，rank 则不会。
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  // 根 为 -1 时不存在分组概念，局部编号就等于 rank 本身
  if (root == -1) return rank;
  int ir = BOOTSTRAP_PID(root, nRoots);
  // 局部编号 = 全局 rank - 该 根 名下的首个 rank
  return rank - firstRankFromRoot(ir, nRanks, nRoots, offset);
}

// 判断给定 rank 是否是其所属 根 名下的第一个 rank(局部编号为 0)。
// 该 rank 在建环过程中承担特殊职责(作为本组的起点)。
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots, offset));
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
static std::mutex bootstrapNetMutex;

NCCL_PARAM(BootstrapNetEnable, "OOB_NET_ENABLE", 0);

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;
      if (env) {
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t {
  findSubnetIf = -1,
  dontCareIf = -2
};

// 检查 中止 函数
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && COMPILER_ATOMIC_LOAD(flag, std::memory_order_acquire)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// 发送/接收 函数
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag,
                             void** sendReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag,
                             void** recvReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle,
                                void* recvComm, void* recvData, int recvSize, void* recvDataHandle, int tag,
                                volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// 基于 套接字 的辅助收发函数：先发送长度，再发送消息体(解决 TCP 流式协议的分包问题)
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0) NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0) NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize,
                                   struct ncclSocket* recvSock, void* recvData, int recvSize) {
  int senderRecvSize;
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange 大小 then asynchronously exchange 数据 入 发送->接收->发送->接收 order
  int senderRecvSize1, senderRecvSize2;
  NCCLCHECK(ncclSocketSendRecv(ops[0].sock, &ops[0].size, sizeof(int), ops[1].sock, &senderRecvSize1, sizeof(int)));
  NCCLCHECK(ncclSocketSendRecv(ops[2].sock, &ops[2].size, sizeof(int), ops[3].sock, &senderRecvSize2, sizeof(int)));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size,
         ops[3].size);
    return ncclInternalError;
  }
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

// 环上邻居的连接信息：两种形态二选一
//   - addr   : 走普通 TCP 套接字 时，就是对端的 IP+端口
//   - 句柄 : 走网络插件(OOB 网络)时，是插件自定义的不透明句柄
union ringConnectInfo {
  union ncclSocketAddress addr;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

// 各 rank 向 根 “报到(检查 入)”时发送的信息包。
// 根 收集齐所有 rank 的这份信息后，就能把它们串成一个环。
struct extInfo {
  int rank;                                  // 发起报到的进程的 rank 编号
  int nranks;                                // 通信域内 rank 总数
  int iroot;                                 // 本 rank 归属的 root 索引
  int nroots;                                // root 的总数量
  int offset;                                // rank 分配偏移量(comm 扩容场景下为老成员数)
  union ncclSocketAddress listenRootAddress; // 本 rank 用于接收 root 回信的监听地址
  union ringConnectInfo connectInfo;         // 本 rank 提供给“环上前驱”用来连接自己的信息
};
#define NET_HANDLE(h, rank) ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

// 根 向某个 rank 回送一条“你的后继邻居的连接信息”。
// 每次都是新建短连接 -> 发送 -> 立即关闭，避免 根 长期持有大量 套接字 句柄。
static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  // magic 是本次通信域的随机校验码，用于防止不同 通信域/不同任务的连接串扰
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}

/* ============================================================================
 * bootstrapRoot —— root 端后台线程：整个 NCCL 建链流程的“交换机”
 * ----------------------------------------------------------------------------
 * 它是 mini-nccl 建链的起点。运行在创建 uniqueId 的那个进程里(通常是 rank 0)。
 *
 * 职责：接收所有 rank 的报到信息，并把它们**首尾相接串成一个环**，
 *       告诉每个 rank “你的下一个邻居是谁、怎么连它”。
 *
 * 为什么只建环而不是全连接？
 *   因为环是后续一切通信的引导拓扑：有了环，各 rank 就能通过 allgather
 *   把所有人的详细信息(busId/网卡/IPC 句柄等)扩散给全体，进而建立任意连接。
 *   这样 root 只需处理 O(nranks) 次交互，而不是 O(nranks^2)。
 *
 * 核心算法(在线串环)：
 *   rank 报到时会带上“别人怎么连我”的 connectInfo，以及“我在哪等回信”的监听地址。
 *   root 维护两张表：rankInfo[](谁的连接信息) 和 rankAddressesRoot[](谁在等回信)。
 *   由于各 rank 报到顺序是乱的，root 采用“**能配对就立刻发，配不上就先存**”的策略：
 *     - 若我的前驱已经在等了 -> 立刻把我的 connectInfo 发给前驱
 *     - 若我的后继已经报到过 -> 立刻把后继的 connectInfo 发给我
 *   这样无需等全部报到完就能边收边配对，显著降低建链延迟。
 * ============================================================================ */
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0, offset = 0;
  struct extInfo info;
  union ringConnectInfo* rankInfo = NULL;            // 暂存表①：各 rank 的连接信息(供其前驱使用)
  union ncclSocketAddress* rankAddressesRoot = NULL; // 暂存表②：各 rank 等待 root 回信的地址
  // 准备全零对象，用于判断表项“是否已被填充”(全零 = 该 rank 还没报到)
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));
  ncclOsSetFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* 主循环：逐个接受各 rank 的报到连接，直到收满 nrecv 个为止 */
  do {
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);   // 阻塞等待一个 rank 连进来
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out); // 收下它的 extInfo 报到包
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);                 // 信息拿到即可关闭，不长期占用连接

    if (c == 0) {
      // 第一个报到的 rank 决定了本次的整体规模参数(后续 rank 必须与之一致，否则报错)
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      nranks = info.nranks;
      iroot = info.iroot;
      nroots = info.nroots;
      offset = info.offset;
      // n2send：本 根 名下真正“归自己管”的 rank 数量，也是最终要回信的数量
      n2send = nRankFromRoot(iroot, nranks, nroots, offset);
      // nrecv：需要接收的报到数量。多根场景下要多收 1 个——
      // 即“下一个 根 名下的首个 rank”，因为环要跨 根 闭合，本 根 的最后一个
      // rank 的后继属于下一个 根，必须拿到它的连接信息才能把环接上。
      // 偏移 > 0(通信域 扩容场景)同理需要切换到多根逻辑。
      nrecv = n2send + ((offset > 0 || nroots > 1) ? 1 : 0);
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot || offset != info.offset) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d, offset %d "
           "vs %d",
           nranks, info.nranks, nroots, info.nroots, iroot, info.iroot, offset, info.offset);
      goto out;
    }

    localId = localIdFromRoot(info.rank, iroot, nranks, nroots, offset);
    if (localId < 0 || localId >= nrecv) {
      WARN("Bootstrap Root : localId %d is out of range", localId);
      goto out;
    }
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    /* --- 配对方向①：把“我的连接信息”交给我的前驱，让前驱能连上我 --- */
    // 计算前驱的局部编号：
    //   - 单 根 时环是闭合的，localId=0 的前驱要绕回最后一个，因此用周期化取模；
    //   - 多 根 时 localId=0 的前驱属于上一个 根 管辖，不归我管，所以直接减 1
    //     得到 -1，随后被 prev >= 0 的条件过滤掉。
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    // 条件：前驱合法(在我管辖范围内)且它已经报到并在等待回信 -> 立即把我的信息发给它
    if (prev >= 0 && prev < n2send &&
        memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      // 前驱还没来报到 -> 先把我的连接信息存起来，等前驱来了再发给它
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }

    /* --- 配对方向②：把“我的后继的连接信息”发给我，让我能连上后继 --- */
    // 后继一定用周期化取模：因为多收的那 1 个信息正好补上跨 根 的环闭合点。
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    // 条件：我在本 根 管辖范围内 [0, n2send)，且后继的信息已经存在表里 -> 立即回信给我
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      // 后继还没报到 -> 把我的等待地址记下来，等后继报到时再由方向①触发发送
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // 发送 剩余的 信息 到 ranks 谁 haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // 此处需 发送 信息 仅 to my 自身的 本地 处理
  for (int r = 0; r < n2send; ++r) {
    // 使用 nrecv to periodize: 若 1 根, 我们会 发送 第一个 one to 最后一个 one,
    // 若 >1 roots 我们会 发送 额外的 one 我们已有 received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)",
        timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo) free(rankInfo);
  if (rankAddressesRoot) free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  std::thread thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  thread = std::thread(bootstrapRoot, args);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  thread.detach();
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // 如果传入了 通信域(扩容操作)，则不应再设置 NCCL_COMM_ID 环境变量
    if (comm) {
      WARN("ncclCommGetUniqueId should not be called when NCCL_COMM_ID is set");
      return ncclInvalidUsage;
    }
    // 正常 初始化: 使用 NCCL_COMM_ID from environment
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
  } else {
    if (comm) {
      // 通信域->childCount 将会 increment 入 ncclCommGrow 对所有 existing ranks, 使用 +1 here
      handle->magic = hashCombine(comm->magic, comm->childCount + 1);
    } else {
      NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    }
    handle->nRanks = comm ? comm->nRanks : 0;
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  if (!parent || !handle) {
    WARN("bcastGrowHandle: parent comm and handle must be provided");
    return ncclInvalidArgument;
  }

  // 父通信域只有单个 rank 时它已持有 句柄，无需再广播
  if (parent->nRanks == 1) return ncclSuccess;
  if (isRoot) {
    NCCLCHECK(bootstrapSend(parent->bootstrap, 0, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
    NCCLCHECK(bootstrapSend(parent->bootstrap, parent->nRanks - 1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  } else {
    NCCLCHECK(bootstrapRecv(parent->bootstrap, -1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    struct {
      struct ncclSocket recv;
      struct ncclSocket send;
    } socket;
  };
};
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct ncclSocket socket; // socket to be used for the ring
  };
};

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;
  ncclNet_t* net;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};
#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// 辅助 函数
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket,
                                       union ncclSocketAddress* addr, ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}
#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;
  if (devOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // 遍历所有设备，返回第一个匹配的
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // 检查 against 用户 specified HCAs/端口
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // 至此，所有普通物理设备都已完成初始化
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          if (!searchNot) {
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          } else {
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "",
                 userIfEnv);
          }
          return ncclInvalidArgument;
        }
      } else {
        // 默认选择 0 号设备
        devOOB = 0;
      }
      // 打印所选设备的信息
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;
  return ncclSuccess;
}

static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen,
                                   char peerHandle[NCCL_NET_HANDLE_MAXSIZE], void** sendComm,
                                   ncclNetDeviceHandle_t** sendDevHandle, void** recvComm,
                                   ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm) NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm) NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}
static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket,
                                      struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic,
                                      volatile uint32_t* abortFlag) {
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  NCCLCHECK(ncclSocketInit(recvSocket));
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}
static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss, union ncclSocketAddress* peerProxy,
                                uint64_t* peerUDS, struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // 打包
  if (peerAddresss) memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy) memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS) memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks) memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // 全收集
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // 解包
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss) memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy) memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS) memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks) memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return ncclSuccess;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = {0};
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;

  // 设置 magic: for grow existing ranks, 接收 from coordinator; 否则 使用 句柄 magic.
  // 这是 一致的 带有 magic 已创建 入 ncclCommGetUniqueId.
  if (handles != NULL) {
    // 状态 并且 通信域 magic 设为 第一个 magic ID
    comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;
  } else if (parent != NULL) {
    comm->magic = state->magic = hashCombine(parent->magic, parent->childCount);
  } else {
    WARN("bootstrapInit: handles and parent are NULL");
    return ncclSystemError;
  }

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // 填充信息结构体
  info.nranks = nranks;
  info.nroots = nHandles;
  // 获取环形连接信息
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (ncclParamBootstrapNetEnable()) {
    // 创建网络接口，供其它 rank 联系我(用于 全收集)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                 &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // 创建 套接字，供环上邻居联系我
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr,
                                 ncclSocketTypeBootstrap));
  }
  // 创建 套接字 for 根 to contact me 使用 the 根's magic
  // For grow 操作, 偏移 is 父->nRanks - 1 (最后 existing rank joins the 根)
  // For 正常 初始化, 偏移 is 0
  int offset = 0;
  if (comm->isGrow) {
    if (parent != NULL) {
      offset = parent->nRanks - 1;
    } else {
      if (handles != NULL) {
        offset = BOOTSTRAP_HANDLE(handles, 0)->nRanks - 1;
      } else {
        WARN("bootstrapInit: handles and parent are NULL");
        return ncclSystemError;
      }
    }
  }
  int curr_root = rootIdFromRank(rank, nranks, nHandles, offset);
  if (curr_root >= 0) {
    NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot,
                                 &info.listenRootAddress, ncclSocketTypeBootstrap));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // 错开各 rank 的连接时刻，避免瞬间大量连接把 根 压垮
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles, offset);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // 套接字 场景下的消息速率(微秒)
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles, offset) / msg_rate;
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    std::this_thread::sleep_for(std::chrono::microseconds(musec));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // 把我的监听 套接字 信息发送给 根
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // 把联系方式发送给我所属的 根
  info.rank = rank;
  info.iroot = curr_root;
  info.offset = offset;
  if (curr_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  if (parent && comm->isGrow && rank != 0) {
    // Grow: ranks 1 to N-1 使用 the 父 bootstrap to 发送 连接 information to 上一个 rank
    NCCLCHECK(bootstrapSend(parent->bootstrap, rank - 1, 0, &info.connectInfo, sizeof(info.connectInfo)));
  }
  // 如有需要, 发送 连接 信息 to 上一个 根
  // commGrow with more than = 1 rank 在 ... 中 父 通信域 is a 特殊的 情形 of multiroot
  if (((comm->isGrow && parent && (parent->nRanks > 1)) || nHandles > 1) &&
      isFirstFromRoot(rank, curr_root, nranks, nHandles, offset)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles, offset);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // 仅 发送 若 根 is 合法的, existing rank N-1 will 使用 the bootstrapSend 仅 上方
    if (prev_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // 获取 信息 on my "下一个" rank 在 ... 中 bootstrap 环 from 根
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  if (curr_root >= 0) {
    NCCLCHECK(ncclSocketInit(&sock));
    NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
    NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
    NCCLCHECK(ncclSocketClose(&sock));
    NCCLCHECK(ncclSocketClose(&listenSockRoot));
  }
  if (parent && comm->isGrow && rank != parent->nRanks - 1) {
    // Grow: ranks 0 to N-2 使用 the 父 bootstrap to 接收 连接 information to 下一个 rank.
    // 这是 一致的 带有 bootstrapSend 上方.
    NCCLCHECK(bootstrapRecv(parent->bootstrap, rank + 1, 0, &nextPeer, sizeof(nextPeer)));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // accept 并且 connect the 环 网络
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                             state->abortFlag));
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // 全收集 所有 listen handlers
  // 若失败, 那些 resources 将会 释放'd 当 calling bootstrapDestroy, 所以 我们可以 返回 immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                   ncclSocketTypeProxy),
                result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // 创建 a 套接字 for others to reach 出 (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                                   ncclSocketTypeBootstrap),
                result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // 初始化 RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS 线程 will 取 ownership 之后 ncclRasAddRanks succeeds.
    NCCLCHECKGOTO(ncclCalloc(&rasRanks, nranks), result, fail);
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = ncclOsGetPid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks + rank) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should 仍 participate 在 ... 中 ringAllInfo 下方 as the 对等端 将会 waiting for us.
      // 仅 确保 那个 the 地址 is clearly 非法的...
      memset(rasRanks + rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                            state->peerProxyAddressesUDS, rasRanks),
                result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // 创建 service 代理 并且 获取 UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result,
                fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    } else {
      rasRanks = nullptr;
    }
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)",
       timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9, timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9, timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9, timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  free(rasRanks);
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  union ringConnectInfo info;
  union ringConnectInfo nextPeer;
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // 创建 a 句柄 为了 others to reach 出 to me
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // 创建 套接字，供环上邻居联系我
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }
  // 创建 a 套接字 for others to reach 出 (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                               ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    }
  }

  // 获取 addr from 下一个 rank 使用 the 父's 连接
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)),
                ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer,
                              sizeof(union ringConnectInfo)),
                ret, fail);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                                 state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->shareResources) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // 创建 service 代理 并且 获取 UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                     ncclSocketTypeProxy),
                  ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                              state->peerProxyAddressesUDS, NULL),
                  ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE",
        comm, parent, rank, nranks, color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;
  int tag;
};
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){state->rank, tag};
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap,
                               state->abortFlag),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}
// Bootstrap 发送/接收 函数
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // 新的未预期消息
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // 入队
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock,
                                      int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    // 对等端 < 0 means wildcard (accept from 任意 对等端)
    if ((peer < 0 || elem->peer == peer) && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// 我们可以't 知道 谁 we'll 接收 from, 所以 需要 接收 everything at 一旦
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected 连接 第一
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new 连接
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    // Match: tag must match, 并且 对等端 must match (对等端 < 0 means wildcard)
    if (ack.tag == tag && (peer < 0 || ack.rank == peer)) return ncclSuccess;
    // 无 match: 队列 for later 并且 尝试 下一个 连接
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
// 我们可以't 知道 谁 we'll 接收 from, 所以 需要 接收 everything at 一旦
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/ true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data,
                                     int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle,
                              tag, abortFlag),
                  res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // 执行 不 失败 若发生 错误, 尝试 deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock, struct ncclSocket* prevSock, int rank, int nranks,
                                        char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);
  int totalSteps = nranks / 2;
  TRACE(NCCL_BOOTSTRAP, "bidirectional bootstrap: totalSteps=%d", totalSteps);
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    // N ranks requires (N-1)/2 步骤 为了 双精度 环  算法.
    // 若 N is 甚至, 最后一个 步骤 is requires a 单个 发送/接收
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    // Ring0: 环 from 前一个 to 下一个
    int sendSliceRing0 = (rank - step + nranks) % nranks;      // Send this slice to next neighbor
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;  // Receive this slice from prev neighbor
    // Ring1: 环 from 下一个 to 前一个
    int sendSliceRing1 = (rank + step) % nranks;               // Send this slice to prev neighbor
    int recvSliceRing1 = (rank + step + 1) % nranks;           // Receive this slice from next neighbor
    if (isFinalUnidirectional) {
      // 最终的 unidirectional 步骤, 仅 Ring0 用于
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size,
                                   size),
                    res, exit);
    } else {
      // Bidirectional 步骤: Ring0 并且 Ring1 用于 simultaneously
      // clang-格式 off
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock, data + sendSliceRing0 * size, size, 0},  // Ring0: send to next
        {NCCL_SOCKET_RECV, prevSock, data + recvSliceRing0 * size, size, 0},  // Ring0: recv from prev
        {NCCL_SOCKET_SEND, prevSock, data + sendSliceRing1 * size, size, 0},  // Ring1: send to prev
        {NCCL_SOCKET_RECV, nextSock, data + recvSliceRing1 * size, size, 0}   // Ring1: recv from next
      };
      // clang-格式 on
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank,
                                   nranks, (char*)allData, size, state->abortFlag),
                  res, exit);
  } else {
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks,
                                      (char*)allData, size),
                  res, exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9,
        (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1) return ncclSuccess;
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];
  int nextRank = ranks[(rank + 1) % nranks];
  // intraNode bootstrap 已完成 defacto 使用 the 套接字-based 实现
  struct ncclSocket recvSocket, sendSocket;
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));

  NCCLCHECK(ncclSocketClose(&sendSocket));
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// [IntraNode] 入-place 广播
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                          int size) {
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root) {
        NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
      }
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData,
                            size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size,
        time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // close unexpected 并且 返回 an 错误 若 we are 不 aborting 并且 仍 操作 在 ... 中 pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  // close the p2p 套接字
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // 代理 things are 释放'd elsewhere
  free(state->peerP2pAddresses);
  free(state);
  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // 当 aborting 需要 close the 代理 here (maybe?)
  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
