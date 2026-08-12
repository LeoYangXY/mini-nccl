/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "checks.h"
#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "shmutils.h"
#include "p2p.h"
#include "transport.h"
#include "mem_manager.h"
#include <assert.h>
#include "shm.h"
#include "register_inline.h"

/* ============================================================================
 * transport/p2p.cc —— GPU 点对点直连传输（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 在 AllReduce 全链路中的定位：device kernel(all_reduce.h)在 GPU 上执行规约时，
 * 需要把数据写到“相邻 rank 的 buffer”或从那里读回来。本文件提供这种
 * “GPU 间直接收发”的传输能力，是 ring/tree 算法相邻 rank 之间 send/recv 的底层支撑。
 *
 * 关键能力：
 *   - 4 种 P2P 类型（p2pType）：DIRECT(同进程直接指针)、INTERMEDIATE、IPC(跨进程
 *     CUDA IPC 显存句柄)、CUMEM(CUDA 虚拟内存映射)。单机多卡大多走 IPC/CUMEM。
 *   - p2pSendSetup/p2pRecvSetup : 建立连接前的信息交换（交换 IPC 句柄/地址）。
 *   - p2pSendConnect/p2pRecvConnect : 完成连接，把对端显存映射成本地可访问的指针。
 *   - p2pSendProxyProgress/p2pRecvProxyProgress : proxy 线程驱动的实际数据搬运进度
 *     函数（Simple 协议下由它把数据推到对端 / 从对端拉回）。
 *   - 末尾 ncclTransport p2pTransport : 把上面这些函数注册为名为 "P2P" 的传输层。
 * ============================================================================
 */

enum p2pType {
  P2P_DIRECT,
  P2P_INTERMEDIATE,
  P2P_IPC,
  P2P_CUMEM
};

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct ncclP2pRequest {
  size_t size;
  int refcount;
  int peerRank;
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // 已使用 by CE memcpy
  ncclShmIpcDesc_t desc;
};
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};
struct p2pShmProxyInfo {
  // 代理 线程与接收端 GPU 之间的共享内存
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;

  // 发送方的中转步骤
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // 接收方 缓冲区
  char* recvFifo;

  // 已使用 by CE memcpy progress 仅
  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[NCCL_STEPS];
};
static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "P2P Connect info is too large");

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  int sendMemSameProc;
  void* recvMemIpc;
  int recvMemSameProc;
  // CE memcpy 支持
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;
};

// 是否支持 cuMem(CUDA 虚拟内存管理)API
struct p2pCuMemProxyInfo {
  struct ncclP2pBuff p2pBuff;
};

#include <sys/types.h>

NCCL_PARAM(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0);
#define NCCL_P2P_MAX_PHYSICAL_SEGMENTS 8192

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (!CUDASUCCESS(cudaGetDeviceCount(&ndev))) return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (!CUDASUCCESS(cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i))) return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was 不 已找到 入 our locally visible CUDA 设备
  return -1;
}

// CE memcpy 支持
NCCL_PARAM(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);
static int useMemcpy = 0;
static void initCeOperation();

extern int64_t ncclParamMNNVLEnable();

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1,
                           struct ncclPeerInfo* info2) {
  initCeOperation();

  // 检查 拓扑 / p2p 层级.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank, NULL));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) {
    if (useMemcpy) *ret = 0;
    return ncclSuccess;
  }

  // 检查是否改用网络传输(网络)效果更好
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) {
    *ret = 0;
    return ncclSuccess;
  }

  // hostHash 是主机的唯一标识。只要两个 rank 不在同一台机器上，
  // 就不可能走 GPU 直连(P2P 只在单机内有效)，此时直接返回让上层改用网络传输。
  if (info1->hostHash != comm->peerInfo[comm->rank].hostHash || info1->hostHash != info2->hostHash) {
    // 只要任意一端不是本机的，就无需继续判断了。
    return ncclSuccess;
  }

  // 把对端的 busId(PCIe 总线地址，全局唯一)转换成本进程可见的 cudaDev 序号。
  // 之所以需要转换，是因为 CUDA_VISIBLE_DEVICES 会让同一块物理卡在不同进程中
  // 拥有不同的设备序号，只有 busId 是稳定不变的。
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
    // 转换失败，说明该卡在本进程的可见设备列表之外
#if CUDART_VERSION >= 10010
    // CUDA 10.1 及以后版本，即使设备对本进程不可见，也仍然可以通过 IPC 走 P2P，
    // 因此这里保持 *ret 的原值(乐观地认为可用)。
    return ncclSuccess;
#else
    // 旧版 CUDA：对端设备在本进程中不可见，就无法与之通信。
    *ret = 0;
    return ncclSuccess;
#endif
  }

  // 询问 CUDA 运行时：这两张卡之间到底能不能做 P2P 访问
  int p2p;
  if (cudaDev1 == cudaDev2) {
    // 两个 rank 落在同一张物理卡上(多 rank 共享一个 GPU 的场景)。
    // 此时必然可以互访，但 cudaDeviceCanAccessPeer 对“自己访问自己”会返回不允许，
    // 所以这里必须特判，直接置为可用。
    p2p = 1;
  } else if (!CUDASUCCESS(cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2))) {
    INFO(NCCL_INIT | NCCL_P2P, "peer query failed between dev %d(=%lx) and dev %d(=%lx)", cudaDev1, info1->busId,
         cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }

  // 下面这段“传统 IPC 探测”在开启 NCCL_CUMEM_ENABLE=1 时必然失败，
  // 因为那种模式走的是 cuMem 新接口而非 cudaIpc 旧接口，所以要用 !ncclCuMemEnable() 排除掉。
  if (p2p != 0 && !ncclCuMemEnable()) {
    // 用 静态 变量缓存探测结果：这个检测需要真实分配显存并申请 IPC 句柄，
    // 开销较大，而同一进程内结果恒定，因此只做一次。
    static int legacyIPC = -1;
    if (legacyIPC >= 0) {
      *ret = legacyIPC;
      return ncclSuccess;
    }
    // 探测传统 cudaIpc 是否真的可用(这是针对 WSL 环境的规避手段 WAR：
    // WSL 下 cudaDeviceCanAccessPeer 会返回可用，但实际申请 IPC 句柄时会失败，
    // 因此必须实际试一次才能确定)。
    char* dummy;
    cudaIpcMemHandle_t ipc;
    NCCLCHECK(ncclCudaMalloc(&dummy, CUDA_IPC_MIN, comm->memManager, ncclMemOffload));
    if (!CUDASUCCESS(cudaIpcGetMemHandle(&ipc, dummy))) {
      INFO(NCCL_INIT | NCCL_P2P, "Legacy IPC not supported");
      *ret = 0;
    }
    NCCLCHECK(ncclCudaFree(dummy, comm->memManager));
    legacyIPC = *ret;
    return ncclSuccess;
  }

  if (p2p == 0) {
    INFO(NCCL_INIT | NCCL_P2P, "Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)", cudaDev1, info1->busId,
         cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC) \
  do { \
    unsigned long* devIpc = (unsigned long*)(DEVIPC); \
    TRACE(P2P, "IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P, "IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// 是否支持 cuMem(CUDA 虚拟内存管理)API
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc* ipcDesc, void** ptr, int peerRank,
                                            struct ncclMemManager* manager, ncclMemType_t memtype) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    CUmemAllocationHandleType type = ncclCuMemHandleType;

    // 是否支持 cuMem(CUDA 虚拟内存管理)API
    CUmemGenericAllocationHandle handle;
    NCCLCHECK(ncclCuMemAlloc(ptr, &handle, type, size, manager, memtype));
    if (manager != nullptr && peerRank >= 0 && memtype != ncclMemPersist) {
      NCCLCHECK(ncclDynMemMarkExportToPeer(manager, *ptr, peerRank));
    }
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // 返回原生 cuMem 句柄，供后续通过 UDS(Unix 域套接字)导出/导入使用
      memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle));
    } else {
      CUCHECK(cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0));
    }
    if (refcount) {
      memcpy(&ipcDesc->memHandle, &handle, sizeof(handle));
      for (int r = 0; r < refcount; ++r) CUCHECK(cuMemRetainAllocationHandle(&handle, *ptr));
    }
#else
    return ncclInternalError;
#endif
  } else {
    // 分配一块 CUDA 显存，并为其生成 IPC 句柄(供其它进程映射)
    NCCLCHECK(ncclCudaCalloc((char**)ptr, size, manager));
    cudaError_t res = cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr);
    if (res != cudaSuccess) {
      WARN("cudaIpcGetMemHandle failed : %s", cudaGetErrorString(res));
      ncclCudaFree(*ptr, manager);
      CUDACHECK(res);
    }
  }
  INFO_LOC(NCCL_P2P | NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc);

  return ncclSuccess;
}

ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc* ipcDesc) {
  return ncclSuccess;
}

ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm* comm, int peer, size_t size, ncclIpcDesc* ipcDesc,
                                          void** devMemPtr, void* ownerPtr, ncclMemType_t memType) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    // 是否支持 cuMem(CUDA 虚拟内存管理)API
    CUdeviceptr dptr = 0;
    CUmemAllocationHandleType type = ncclCuMemHandleType;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc* cuDesc = &ipcDesc->cuDesc;
    CUmemAllocationProp prop = {};
    size_t granularity = 0;

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = type;
    prop.location.id = comm->cudaDev;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, granularity);

    // 导入远端内存描述符，并把它映射到本地 GPU 的地址空间
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd 支持
      int fd = -1;
      // 把 cuMem 句柄发送给远端，由其转换为文件描述符(fd)
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, peer, &cuDesc->data, &fd));
      INFO(NCCL_P2P, "UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&cuDesc->data, fd, peer);
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void*)(uintptr_t)fd, type));
      SYSCHECK(close(fd), "close");
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&handle, cuDesc, type));
    }
    CUCHECK(cuMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    CUCHECK(cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0));

    TRACE(NCCL_P2P, "Imported shareable buffer size %zu handle 0x%llx dptr %p", size, handle, (void*)dptr);

    // 授权本地 GPU 访问该内存
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = comm->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, size, &accessDesc, 1));
    TRACE(NCCL_P2P, "Set Access for %p size %zu on dev %d", (void*)dptr, size, accessDesc.location.id);

    *devMemPtr = (void*)dptr;

    // 跟踪已导入的缓冲区
    NCCLCHECK(ncclMemTrackImportFromPeer(comm->memManager, (void*)dptr, size, handle, type, memType, peer,
                                         comm->peerInfo[peer].cudaDev, ownerPtr));
#else
    return ncclInternalError;
#endif
  } else {
    // 传统 CUDA IPC 路径
    CUDACHECK(cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess));
  }

  INFO_LOC(NCCL_P2P, "Imported shareable buffer device %d size %zu ptr %p", comm->cudaDev, size, *devMemPtr);

  return ncclSuccess;
}

/*
 * P2P 有两种数据搬运方向，性能差异很大：
 *   - Write(写)：发送方主动把数据写入接收方显存。这是默认方式，
 *                写操作可以“发射后不管(fire-and-forget)”，不需要等待返回，延迟低。
 *   - Read(读) ：接收方主动去发送方显存里把数据读回来。读操作必须等待数据返回，
 *                但在某些拓扑(如 Ampere + NVLink)上能获得更高的带宽利用率。
 * 该参数设为非 0 时强制使用 Read 模式；-2 表示“不覆盖，交由拓扑自动决定”。
 */
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);
// 设为 1 可禁用 P2P 直连指针(direct 指针)优化，强制走中转缓冲区，一般仅用于排查问题
NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

// 判断两个 rank 是否处于同一个进程内：主机相同 且 进程号相同。
// 同进程意味着共享同一个虚拟地址空间，可以省去 IPC 句柄导入的开销。
#define P2P_SAME_PID(MYINFO, PEERINFO) \
  ((MYINFO->hostHash == PEERINFO->hostHash) && (MYINFO->pidHash == PEERINFO->pidHash))

// 决定这一对 rank 之间该用 读取 还是 写入 模式，以及是否需要经由中转 rank。
static ncclResult_t p2pGetInfo(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read,
                               int* intermediateRank) {
  int p2p;
  // 查询拓扑信息：如果两张 GPU 是 Ampere 及以上架构、且通过 NVLink 直连，
  // 则默认启用 P2P 读取 模式(该组合下读比写更能压满 NVLink 带宽)。
  // intermediateRank 是输出参数：当两卡无法直连时，返回一个可作为中转的 rank。
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, &p2p, read, intermediateRank, NULL));

  // 用户通过环境变量显式指定时(非默认值 -2)，覆盖掉上面拓扑的自动判断结果
  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, struct ncclPeerInfo* myInfo,
                           struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {
  if (P2P_SAME_PID(myInfo, peerInfo)) {
    // 情况一：双方在同一进程内。地址空间共享，对端指针可以直接使用，
    // 只需向 CUDA 声明“允许访问对方显存”即可，无需做 IPC 句柄的导出/导入。
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // 同一进程但是不同 GPU：需要显式开启 对等端 access 权限
      // (走传统 CUDA IPC 路径)
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // 已经开启过则视为成功；这里调用 cudaGetLastError 是为了把这个
        // “非致命错误”从 CUDA 的错误状态中清除掉，避免影响后续调用的错误判断。
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s", peerInfo->cudaDev, peerInfo->busId, err,
             cudaGetErrorString(err));
        return ncclInternalError;
      }
      if (ncclCuMemEnable()) {
        // 对于同进程内的 rank，仍需要映射对端的 memHandle 以增加其引用计数。
        // 否则一旦对端异常退出并释放了该缓冲区，本 rank 继续访问就会触发非法内存访问。
        NCCLCHECK(ncclCuMemAllocAddr(devMem, &p2pBuff->ipcDesc.memHandle, p2pBuff->size));
        CUCHECK(cuMemRelease(p2pBuff->ipcDesc.memHandle));
        *ipcPtr = *devMem;

        // 登记为“从对端导入的内存”，纳入动态内存管理进行跟踪
        // 传 句柄=0：因为上面已经释放过一次引用，挂起流程不应重复释放
        NCCLCHECK(ncclMemTrackImportFromPeer(comm->memManager, *devMem, p2pBuff->size, 0, ncclCuMemHandleType,
                                             ncclMemOffload, peerInfo->rank, peerInfo->cudaDev, p2pBuff->directPtr));
      } else {
        *devMem = p2pBuff->directPtr;
        *ipcPtr = NULL;
      }
    } else {
      *devMem = p2pBuff->directPtr;
      *ipcPtr = NULL;
    }
  } else {
    // 不同 PID
    // 把 p2pBuff->directPtr 作为 ownerPtr 传入，供恢复(restore)阶段做 P2P 句柄交换
    NCCLCHECK(ncclP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size, &p2pBuff->ipcDesc, devMem,
                                           p2pBuff->directPtr, ncclMemOffload));
    *ipcPtr = *devMem;
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
// 发送端连接建立前的“设置”：与对端交换拓扑/地址信息，确定用哪种 p2pType
// (DIRECT/IPC/CUMEM)，并准备对端显存的 IPC 句柄。全规约 的 环/树 每个
// 发送 通道都会先走这里拿到连接所需元信息。
ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo,
                          struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send,
                          int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank));
  if (useMemcpy) useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // CollNet 场景：散播-规约 阶段(连接 1)用写模式，广播-收集 阶段(连接 0)用读模式
  if (graph && connIndex == 1) info->read = 0;
  const char* useReadStr = info->read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // P2P 读模式下，SIMPLE 协议的缓冲区被追加在 ncclSendMem 结构体的末尾
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) {
      resources->type = P2P_DIRECT;
      INFO(NCCL_INIT | NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/direct pointer%s", channelId, connIndex,
           myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr);
    } else {
      // 是否支持 cuMem(CUDA 虚拟内存管理)API
      if (ncclCuMemEnable()) {
        resources->type = P2P_CUMEM;
        const char* MNNVL = comm->MNNVL ? "MNNVL" : "CUMEM";
        INFO(NCCL_INIT | NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/%s%s%s", channelId, connIndex,
             myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, MNNVL, useReadStr,
             useMemcpy ? "/CE" : "");
      } else {
        // 传统 CUDA IPC 路径
        resources->type = P2P_IPC;
        INFO(NCCL_INIT | NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/IPC%s%s", channelId, connIndex,
             myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : "");
      }
    }
    send->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
    INFO(NCCL_INIT | NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/indirect/%d[%d]%s", channelId, connIndex,
         myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, intermediateRank,
         comm->peerInfo[intermediateRank].nvmlDev, useReadStr);
  }

  memset(&req, '\0', sizeof(req));
  req.size = sendSize;
  req.refcount = 0;
  req.peerRank = peerInfo->rank;  // Track which peer will import this buffer
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) &&
      (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) {
    req.refcount++;
  }
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) {
    req.refcount++;
  }
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, info->rank, &send->proxyConn));
  if (useMemcpy) {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo,
                                    sizeof(struct p2pShmProxyInfo)));
    memcpy(&info->desc, &resources->proxyInfo.desc, sizeof(ncclShmIpcDesc_t));
  } else {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest),
                                    &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo + info->rank, &info->p2pBuff,
                     (void**)&resources->sendDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  }

  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo,
                          struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv,
                          int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // CollNet 场景：散播-规约 阶段(连接 1)用写模式，广播-收集 阶段(连接 0)用读模式
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // P2P 读模式下，SIMPLE 协议的缓冲区被追加在 ncclSendMem 结构体的末尾
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
  }
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) {
      resources->type = P2P_DIRECT;
    } else {
      if (ncclCuMemEnable()) {
        // 是否支持 cuMem(CUDA 虚拟内存管理)API
        resources->type = P2P_CUMEM;
        TRACE(NCCL_INIT | NCCL_P2P, "Ring %02d : %d[%d] <- %d[%d] via P2P/CUMEM", channelId, myInfo->rank,
              myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
      } else {
        // 传统 CUDA IPC 路径
        resources->type = P2P_IPC;
      }
    }
    recv->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
  }

  memset(&req, '\0', sizeof(req));
  req.size = recvSize;
  req.refcount = 0;
  req.peerRank = peerInfo->rank;  // Track which peer will import this buffer
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) &&
      (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) {
    req.refcount++;
  }
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) {
    req.refcount++;
  }
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 0, info->rank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest),
                                  &info->p2pBuff, sizeof(struct ncclP2pBuff)));

  NCCLCHECK(p2pMap(comm, &recv->proxyConn, myInfo, comm->peerInfo + info->rank, &info->p2pBuff,
                   (void**)&resources->recvDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  return ncclSuccess;
}

/* Connect/Send to this peer */
// 发送端“connect”：使用 设置 阶段拿到的连接信息，把对端显存映射成本地可写指针，
// 填充连接结构(结构体 connect)。之后 设备 内核 就可以通过该指针把数据直接写到
// 相邻 rank 的 缓冲区（跨进程走 CUDA IPC / CUMEM）。
static ncclResult_t p2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank,
                                   struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  struct ncclRecvMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo + rank, comm->peerInfo + info->rank, &info->p2pBuff,
                   (void**)&remDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));

  char* buff = (char*)(remDevMem + 1);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      if (resources->sendDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      send->conn.buffs[p] = (char*)(resources->sendDevMem + 1);
    } else {
      send->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

  if (useMemcpy) {
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail;
    send->conn.connFifo = resources->proxyInfo.ceRecvMem->connFifo;
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head;
    // 把 SIMPLE 缓冲区交给 代理，并用本地缓冲区替换它
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE],
                                    sizeof(void*), NULL, 0));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff;
  } else {
    send->conn.tail = &remDevMem->tail;
    send->conn.head = &resources->sendDevMem->head;
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
    send->conn.redOpArgExchange = resources->sendDevMem->redOpArgExchange;
  }
  // 必须设置 proxyConn 的 proxyProgress 属性，才能在入队时正确校验
  send->proxyConn.proxyProgress = p2pTransport.send.proxyProgress;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank,
                            struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  struct ncclSendMem* remDevMem = NULL;

  if (useMemcpy) {
    // 挂接到对端的共享内存(SHM)段
    NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->shm,
                                           (void**)&resources->devShm, &resources->desc));

    recv->conn.tail = &resources->devShm->recvMem.tail;
    recv->conn.head = &resources->devShm->sendMem.head;
  } else {
    NCCLCHECK(p2pMap(comm, &recv->proxyConn, comm->peerInfo + rank, comm->peerInfo + info->rank, &info->p2pBuff,
                     (void**)&remDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));

    struct ncclRecvMem* devMem = resources->recvDevMem;
    recv->conn.tail = &devMem->tail;
    recv->conn.head = &remDevMem->head;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
    recv->conn.redOpArgExchange = remDevMem->redOpArgExchange;
  }
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

  char* buff = (char*)(resources->recvDevMem + 1);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      if (remDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = (char*)(remDevMem + 1);
    } else {
      recv->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }
  return ncclSuccess;
}

ncclResult_t p2pSendFree(struct ncclComm* comm, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // 是否支持 cuMem(CUDA 虚拟内存管理)API
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc, comm->memManager));
        } else {
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc, comm->memManager));
        }
      }

      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc, comm->memManager));
        } else {
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc, comm->memManager));
        }
      }
    } else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
    }
    free(resources);
  }
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // 是否支持 cuMem(CUDA 虚拟内存管理)API
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc, comm->memManager));
        } else {
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc, comm->memManager));
        }
      }

      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc, comm->memManager));
        } else {
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc, comm->memManager));
        }
      }
    } else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
      if (useMemcpy) {
        NCCLCHECK(ncclShmIpcClose(&resources->desc));
      }
    }
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t p2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                      void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (useMemcpy) {
    // CE memcpy 支持
    struct p2pShmProxyInfo* proxyInfo;
    size_t shmSize;

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;

    NCCLCHECK(ncclCudaCalloc(&proxyInfo->ceDevBuff, proxyState->buffSizes[NCCL_PROTO_SIMPLE], proxyState->memManager));

    // 创建一个共享内存段，供对端挂接
    shmSize = sizeof(struct ncclSendMem) + sizeof(struct ncclRecvMem);
    NCCLCHECK(ncclShmAllocateShareableBuffer(shmSize, false, &proxyInfo->desc, (void**)&proxyInfo->shm,
                                             (void**)&proxyInfo->devShm));

    NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1));
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo));
  } else {
    struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
    if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
    int size = req->size;
    if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr, req->peerRank,
                                             proxyState->memManager, ncclMemOffload));
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
      // 是否支持 cuMem(CUDA 虚拟内存管理)API
      struct p2pCuMemProxyInfo* proxyInfo;
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
      connection->transportResources = proxyInfo;
    } else {
      connection->transportResources = p2pBuff->directPtr;
    }
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                      void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
  if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
  int size = req->size;
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
  NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr, req->peerRank,
                                           proxyState->memManager, ncclMemOffload));
  p2pBuff->size = size;
  if (ncclCuMemEnable()) {
    // 是否支持 cuMem(CUDA 虚拟内存管理)API
    struct p2pCuMemProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    connection->transportResources = proxyInfo;
  } else {
    connection->transportResources = p2pBuff->directPtr;
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                        void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;

  if (reqSize != sizeof(void*)) return ncclInternalError;
  proxyInfo->recvFifo = *((char**)reqBuff);

  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i = 0; i < NCCL_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events + i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  return ncclSuccess;
}

static ncclResult_t p2pDeregisterMemHandle(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                           struct ncclIpcImpInfo* ipcInfo) {
  if (ipcInfo->legacyIpcCap) {
    CUDACHECK(cudaIpcCloseMemHandle((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)));
  } else {
    if (connection->sameProcess) {
      NCCLCHECK(ncclCuMemFreeAddr((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset), nullptr,
                                  ipcInfo->numSegments));
    } else {
      NCCLCHECK(ncclCudaFree((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset), nullptr, ipcInfo->numSegments));
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
    struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
    p2pDeregisterMemHandle(connection, proxyState, (struct ncclIpcImpInfo*)memHandle->handle);
    free(memHandle->handle);
    free(memHandle);
  }

  // CE memcpy 支持
  if (useMemcpy) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      NCCLCHECK(ncclCudaHostFree(proxyInfo->ceRecvMem));
      NCCLCHECK(ncclCudaFree(proxyInfo->ceDevBuff, proxyState->memManager));
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i = 0; i < NCCL_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
      // 是否支持 cuMem(CUDA 虚拟内存管理)API
      struct p2pCuMemProxyInfo* proxyInfo = (struct p2pCuMemProxyInfo*)connection->transportResources;
      if (proxyInfo) {
        struct ncclP2pBuff* p2pBuff = &proxyInfo->p2pBuff;
        ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
        ncclCudaFree(p2pBuff->directPtr, proxyState->memManager);
        free(proxyInfo);
      }
    } else {
      // 执行 不 检查 返回 代码 as CUDA may have 已经 shut down
      ncclCudaFree(connection->transportResources, proxyState->memManager);
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
    struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
    p2pDeregisterMemHandle(connection, proxyState, (struct ncclIpcImpInfo*)memHandle->handle);
    free(memHandle->handle);
    free(memHandle);
  }

  if (ncclCuMemEnable()) {
    struct p2pCuMemProxyInfo* proxyInfo = (struct p2pCuMemProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      struct ncclP2pBuff* p2pBuff = &proxyInfo->p2pBuff;
      ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
      ncclCudaFree(p2pBuff->directPtr, proxyState->memManager);
      free(proxyInfo);
    }
  } else {
    // 执行 不 检查 返回 代码 as CUDA may have 已经 shut down
    ncclCudaFree(connection->transportResources, proxyState->memManager);
  }
  return ncclSuccess;
}

// CE memcpy 支持
// 代理 线程驱动的“发送进度”函数：Simple 协议下，由 代理 线程把本 rank 的数据
// 推送到相邻 rank 的 缓冲区（或 LL/LL128 协议下做带 标志 的同步搬运）。它会被
// ncclProxyProgress 在进度循环里反复调用，直到本次 发送 的 所有 字节 完成。
static ncclResult_t p2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
      // 向上取整到 sliceSteps 的整数倍
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE) {
        // 仅 Simple 使用 cudaMemcpy
        resources->step = sub->base + sub->nsteps;
        args->done++;
        continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // 检查 GPU has sent everything
        if ((*recvTail > sub->base + sub->transmitted)) {
          int size = connFifo[buffSlot].size;
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo + buffSlot * stepSize,
                                    resources->ceDevBuff + buffSlot * stepSize, size, cudaMemcpyDeviceToDevice,
                                    resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base + sub->done) % NCCL_STEPS;
        cudaError_t res = CUDACLEARERROR(cudaEventQuery(resources->events[buffSlot]));
        if (res != cudaErrorNotReady) CUDACHECK(res);
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          // 通知 SHM
          resources->shm->recvMem.tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t ipcHandleMultiSegmentRegistration(CUdeviceptr userBuff, size_t userBuffSize, ncclComm* comm,
                                               struct ncclProxyConnector* proxyConn, size_t* totalMappedBufferSize,
                                               int* numSegments, struct p2pIpcExpInfo** ipcInfos) {
  ncclResult_t ret = ncclSuccess;
  *totalMappedBufferSize = 0;
  *numSegments = 0;
  CUdeviceptr userBuffStart = userBuff;
  CUdeviceptr userBuffEnd = userBuffStart + userBuffSize;
  CUdeviceptr mappedPtrEnd = userBuffStart;
  CUdeviceptr tmpBase;
  size_t tmpBaseSize;
  CUmemGenericAllocationHandle* segmentHandles = nullptr;
  int* expFds = nullptr;
  int* impFds = nullptr;
  int capacity = 2;
  // 该代码路径下至少需要两个段
  NCCLCHECK(ncclCalloc(ipcInfos, capacity));
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    NCCLCHECK(ncclCalloc(&expFds, capacity));
    for (int idx = 0; idx < capacity; idx++) {
      expFds[idx] = -1;
    }
  }
  NCCLCHECK(ncclCalloc(&segmentHandles, capacity));

  while (mappedPtrEnd < userBuffEnd) {
    int segment = *numSegments;
    if (segment == capacity) {
      capacity = capacity * 2;
      NCCLCHECKGOTO(ncclRealloc(ipcInfos, segment, capacity), ret, fail);
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        NCCLCHECKGOTO(ncclRealloc(&expFds, segment, capacity), ret, fail);
        for (int idx = segment; idx < capacity; idx++) {
          expFds[idx] = -1;
        }
      }
      NCCLCHECKGOTO(ncclRealloc(&segmentHandles, segment, capacity), ret, fail);
    }
    struct p2pIpcExpInfo* ipcInfo = *ipcInfos + segment;
    CUCHECKGOTO(cuMemGetAddressRange(&tmpBase, &tmpBaseSize, mappedPtrEnd), ret, fail);
    ipcInfo->size = tmpBaseSize;
    CUCHECKGOTO(cuMemRetainAllocationHandle(&segmentHandles[segment], (void*)tmpBase), ret, fail);
    // 在此处递增 numSegments，以便某个段导出失败时，已持有的句柄仍能被正确释放
    *numSegments = *numSegments + 1;
    if (*numSegments > NCCL_P2P_MAX_PHYSICAL_SEGMENTS) {
      INFO(NCCL_REG,
           "Number of segments exceeded maximum number of supported physical segments for p2p (%d). Skipping "
           "multi-segment registration.",
           NCCL_P2P_MAX_PHYSICAL_SEGMENTS);
      ret = ncclInternalError;
      goto fail;
    }
    if (proxyConn->sameProcess) {
      memcpy(&ipcInfo->ipcDesc.memHandle, &segmentHandles[segment], sizeof(CUmemGenericAllocationHandle));
    } else {
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        CUCHECKGOTO(cuMemExportToShareableHandle(&expFds[segment], segmentHandles[segment], ncclCuMemHandleType, 0),
                    ret, fail);
      } else {
        CUCHECKGOTO(cuMemExportToShareableHandle(&ipcInfo->ipcDesc.cuDesc.handle, segmentHandles[segment],
                                                 ncclCuMemHandleType, 0),
                    ret, fail);
      }
    }

    *totalMappedBufferSize += tmpBaseSize;
    mappedPtrEnd = tmpBase + tmpBaseSize;
  }

  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    NCCLCHECKGOTO(ncclCalloc(&impFds, *numSegments), ret, fail);
    NCCLCHECKGOTO(ncclProxyClientBatchQueryFdBlocking(comm, proxyConn, expFds, impFds, *numSegments), ret, fail);
  }
  for (int segment = 0; segment < *numSegments; segment++) {
    struct p2pIpcExpInfo* ipcInfo = *ipcInfos + segment;
    if (!proxyConn->sameProcess) {
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        ipcInfo->impFd = impFds[segment];
        close(expFds[segment]);
        expFds[segment] = -1;
      }
    }
    CUCHECKGOTO(cuMemRelease(segmentHandles[segment]), ret, fail);
  }

  TRACE(NCCL_REG, "Populated multi-segment IPC infos. totalMappedBufferSize : %zu, Num segments : %d",
        *totalMappedBufferSize, *numSegments);

exit:
  free(expFds);
  free(impFds);
  free(segmentHandles);
  return ret;
fail:
  if (ipcInfos && *ipcInfos) {
    free(*ipcInfos);
    *ipcInfos = nullptr;
  }
  for (int segment = 0; segment < *numSegments; segment++) {
    if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      if (expFds[segment] != -1) {
        close(expFds[segment]);
        expFds[segment] = -1;
      }
    }
    CUCHECKIGNORE(cuMemRelease(segmentHandles[segment]));
  }
  goto exit;
}

static ncclResult_t ipcRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers,
                                      ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag,
                                      uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc) {
  ncclResult_t ret = ncclSuccess;
  struct p2pIpcExpInfo* ipcInfo = nullptr;
  struct ncclIpcRegInfo* newInfo = NULL;
  uintptr_t* peerRmtAddrs = NULL;
  int legacyIpcCap = 0;
  size_t baseSize = 0;
  size_t totalMappedSize = 0;
  void* baseAddr = NULL;
  bool needUpdate = false;
  // 跨 clique(可直连分组)的 P2P：用 peerRank 作为下标，用 nRanks 作为数组长度
  int ipcIndexSize = comm->p2pCrossClique ? comm->nRanks : comm->localRanks;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (isLegacyIpc) *isLegacyIpc = false;
  if (regRecord) {
    // 该缓冲区由用户注册，我们需要开始注册它或复用已有注册
    int peerIndex = -1;

    // 分配 或者 resize ipcInfos 数组 如有需要
    if (regRecord->ipcInfos == NULL || regRecord->ipcInfosSize < ipcIndexSize) {
      NCCLCHECKGOTO(ncclRealloc(&regRecord->ipcInfos, regRecord->ipcInfosSize, ipcIndexSize), ret, fail);
      regRecord->ipcInfosSize = ipcIndexSize;
    }

    for (int p = 0; p < nPeers; p++) {
      int peerRank = peerRanks[p];
      if (peerRank < 0 || peerRank >= comm->nRanks) {
        WARN("rank %d invalid IPC peerRank %d nRanks %d", comm->rank, peerRank, comm->nRanks);
        ret = ncclInternalError;
        goto fail;
      }
      // 跨 clique 的 P2P 直接使用 peerRank，避免不同 clique 之间 localRank 冲突
      peerIndex = comm->p2pCrossClique ? peerRank : comm->rankToLocalRank[peerRank];
      if (peerIndex < 0 || peerIndex >= ipcIndexSize) {
        WARN("rank %d invalid IPC peerIndex %d for peerRank %d ipcIndexSize %d ipcInfosSize %d p2pCrossClique %d",
             comm->rank, peerIndex, peerRank, ipcIndexSize, regRecord->ipcInfosSize, comm->p2pCrossClique);
        ret = ncclInternalError;
        goto fail;
      }
      if (regRecord->ipcInfos[peerIndex]) {
        // 该对端的 IPC 信息已存在，无需重复注册，直接复用即可
        *regBufFlag = 1;
        if (isLegacyIpc) *isLegacyIpc = regRecord->ipcInfos[peerIndex]->impInfo.legacyIpcCap;
        INFO(NCCL_REG,
             "rank %d - IPC reuse buffer %p size %zu (baseAddr %p size %zu numSegments %d) to peer %d regAddr %p",
             comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, regRecord->endAddr - regRecord->begAddr,
             regRecord->ipcInfos[peerIndex]->impInfo.numSegments, peerRank,
             regRecord->ipcInfos[peerIndex]->impInfo.rmtRegAddr);
      } else {
        // 使用 peerLocalRank 注册缓冲区
        struct ncclProxyConnector* proxyConn = NULL;
        int numSegments = 1;
        bool multiSegment = false;

        if (baseAddr == NULL) {
          CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail);
          CUCHECKGOTO(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                                            (CUdeviceptr)baseAddr),
                      ret, fail);
        }

        if ((uint64_t)baseAddr + baseSize < (uint64_t)userbuff + buffSize) multiSegment = true;
        if (multiSegment && (!ncclCuMemEnable() || !ncclParamMultiSegmentRegister())) goto exit;

        if (!multiSegment) {
          NCCLCHECKGOTO(ncclCalloc(&ipcInfo, 1), ret, fail);
          totalMappedSize = baseSize;
        }

        if (comm->gproxyConn[peerRank].initialized == false) {
          NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, peerRank, &comm->gproxyConn[peerRank]), ret, fail);
        }
        proxyConn = &comm->gproxyConn[peerRank];

        // 获取该缓冲区的内存句柄。它可能是通过 cudaMalloc 分配的，那样我们会
        // 拿到传统 CUDA 内存句柄；也可能是通过 cuMem* 系列接口分配的。
        if (ncclCuMemEnable()) {
          if (multiSegment) {
            NCCLCHECKGOTO(ipcHandleMultiSegmentRegistration((CUdeviceptr)userbuff, buffSize, comm, proxyConn,
                                                            &totalMappedSize, &numSegments, &ipcInfo),
                          ret, fail);
          } else {
            CUmemGenericAllocationHandle handle;
            if (CUPFN(cuMemRetainAllocationHandle(&handle, baseAddr)) != CUDA_SUCCESS) {
              // 若 cuMem* export 失败, 重试 legacy export
              if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
              CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo->ipcDesc.devIpc, baseAddr), ret, fail);
              ipcInfo->legacyIpcCap = true;
              if (isLegacyIpc) *isLegacyIpc = true;
            } else {
              ipcInfo->legacyIpcCap = false;
              if (isLegacyIpc) *isLegacyIpc = false;
              // 通过 cuMem* 接口导出为文件描述符或 fabric 句柄
              if (proxyConn->sameProcess) {
                memcpy(&ipcInfo->ipcDesc.memHandle, &handle, sizeof(CUmemGenericAllocationHandle));
              } else {
                if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
                  int expFd = -1;
                  CUCHECKGOTO(cuMemExportToShareableHandle(&expFd, handle, ncclCuMemHandleType, 0), ret, fail);
                  NCCLCHECKGOTO(ncclProxyClientQueryFdBlocking(comm, proxyConn, expFd, &ipcInfo->impFd), ret, fail);
                  SYSCHECKGOTO(close(expFd), "close", ret, fail);
                } else {
                  // 允许此处静默失败：用户缓冲区确实存在无法注册的情况，属于正常回退
                  if (CUPFN(cuMemExportToShareableHandle(&ipcInfo->ipcDesc.cuDesc.handle, handle, ncclCuMemHandleType,
                                                         0)) != CUDA_SUCCESS) {
                    CUCHECKGOTO(cuMemRelease(handle), ret, fail);
                    goto fail;
                  }
                }
              }
              CUCHECKGOTO(cuMemRelease(handle), ret, fail);
            }
          }
        } else if (legacyIpcCap) {
          // 旧版导出
          if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
          CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo->ipcDesc.devIpc, baseAddr), ret, fail);
          ipcInfo->legacyIpcCap = true;
          if (isLegacyIpc) *isLegacyIpc = true;
        } else {
          // nothing works, 仅 返回
          goto fail;
        }

        void* rmtRegAddr = NULL;
        if (!multiSegment) {
          ipcInfo->size = totalMappedSize;
        }
        ipcInfo->offset = regRecord->begAddr - (uintptr_t)baseAddr;
        // 此时 ipcInfo 已包含全部必要的注册信息。接下来在 代理 侧注册缓冲区，
        // 并把远端的注册地址取回来。
        if (proxyConn) {
          INFO(NCCL_REG,
               "rank %d - IPC registering buffer %p size %zu (baseAddr %p totalSize %zu numSegments %d) to peer %d",
               comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, totalMappedSize, numSegments, peerRank);
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgRegister, ipcInfo,
                                              sizeof(p2pIpcExpInfo) * numSegments, &rmtRegAddr, sizeof(void*)),
                        ret, fail);
        }
        if (rmtRegAddr) {
          NCCLCHECKGOTO(ncclCalloc(&newInfo, 1), ret, fail);
          assert(regRecord->ipcInfos[peerIndex] == NULL);
          regRecord->state |= IPC_REG_COMPLETE;
          newInfo->peerRank = peerRank;
          newInfo->baseAddr = baseAddr;
          newInfo->impInfo.rmtRegAddr = rmtRegAddr;
          newInfo->impInfo.offset = ipcInfo->offset;
          newInfo->impInfo.legacyIpcCap = ipcInfo->legacyIpcCap;
          newInfo->impInfo.numSegments = numSegments;
          newInfo->ipcProxyconn = proxyConn;
          regRecord->ipcInfos[peerIndex] = newInfo;
          if (regRecord->regIpcAddrs.hostPeerRmtAddrs == NULL) {
            NCCLCHECKGOTO(ncclCalloc(&regRecord->regIpcAddrs.hostPeerRmtAddrs, ipcIndexSize), ret, fail);
          }
          regRecord->regIpcAddrs.hostPeerRmtAddrs[peerIndex] = (uintptr_t)rmtRegAddr;
          needUpdate = true;
          *regBufFlag = 1;
          INFO(NCCL_REG,
               "rank %d - IPC register buffer %p size %zu (baseAddr %p totalSize %zu numSegments %d) to peer %d "
               "regAddr %p offsetOut %ld",
               comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, totalMappedSize, numSegments, peerRank,
               rmtRegAddr, (uintptr_t)userbuff - regRecord->begAddr);
        }
      }
    }

    if (*regBufFlag) {
      if (needUpdate) {
        cudaStream_t hostStream, deviceStream;
        NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
                                              &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream),
                      ret, fail);
        NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
                                              &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream),
                      ret, fail);
        if (regRecord->regIpcAddrs.devPeerRmtAddrs == NULL) {
          NCCLCHECKGOTO(ncclCudaCallocAsync(&regRecord->regIpcAddrs.devPeerRmtAddrs, ipcIndexSize, hostStream,
                                            comm->memManager),
                        ret, fail);
        }
        NCCLCHECKGOTO(ncclCudaMemcpyAsync(regRecord->regIpcAddrs.devPeerRmtAddrs,
                                          regRecord->regIpcAddrs.hostPeerRmtAddrs, ipcIndexSize, hostStream),
                      ret, fail);
        NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), ret, fail);
        NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
                                              &comm->sharedRes->hostStream, /*concurrent=*/false),
                      ret, fail);
        NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
                                              &comm->sharedRes->deviceStream, /*concurrent=*/false),
                      ret, fail);
      }
      if (type == NCCL_IPC_COLLECTIVE) {
        // for 集合, 已注册 远端 缓冲区 are copied to dev 内存 for future 参考
        peerRmtAddrs = regRecord->regIpcAddrs.devPeerRmtAddrs;
      } else {
        assert(nPeers == 1);
        // p2p always 返回 远端 addr here 自 远端 缓冲区 addr is passed 入 ncclDevWorkP2p 结构体
        peerRmtAddrs = (uintptr_t*)regRecord->regIpcAddrs.hostPeerRmtAddrs[peerIndex];
      }
      *offsetOut = (uintptr_t)userbuff - regRecord->begAddr;
      *peerRmtAddrsOut = peerRmtAddrs;
    }
  }
exit:
  free(ipcInfo);
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (newInfo) free(newInfo);
  INFO(NCCL_REG, "rank %d failed to IPC register userbuff %p buffSize %ld nPeers %d isLegacyIpc %d type %s", comm->rank,
       userbuff, buffSize, nPeers, isLegacyIpc ? *isLegacyIpc : -1,
       ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? "POSIX_FD" : "FABRIC");
  goto exit;
}

ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                        int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                        uintptr_t** peerRmtAddrsOut) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg* regRecord = NULL;
  bool isValid = false;
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    if (isValid) {
      NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag,
                                      offsetOut, peerRmtAddrsOut, NULL),
                    ret, fail);
    }
  }

exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm* comm;
  struct ncclReg* reg;
};

static ncclResult_t cleanupIpc(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclIpcCleanupCallback* obj = (struct ncclIpcCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                        int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                        uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  void* baseAddr;
  size_t baseSize;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue =
    reinterpret_cast<struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>*>(cleanupQueuePtr);
  bool isLegacyIpc = false;
  struct ncclReg* regRecord = NULL;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclCuMemGetAddressRange((CUdeviceptr)userbuff, buffSize, (CUdeviceptr*)&baseAddr, &baseSize, NULL),
                  ret, fail);
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, baseAddr, baseSize, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut,
                                    peerRmtAddrsOut, &isLegacyIpc),
                  ret, fail);
    if (*regBufFlag) {
      struct ncclIpcCleanupCallback* record;
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
      record->base.fn = cleanupIpc;
      record->comm = comm;
      record->reg = regRecord;
      if (isLegacyIpc) {
        ncclIntruQueueEnqueue(&comm->legacyRegCleanupQueue, (struct ncclCommCallback*)record);
      } else {
        ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
        if (nCleanupQueueElts) *nCleanupQueueElts += 1;
      }
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }

exit:
  // coverity[leaked_storage:假] => normally, addrsRecord is added 到 cleanupQueue
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo) {
  NCCLCHECK(ncclProxyCallBlocking(comm, regInfo->ipcProxyconn, ncclProxyMsgDeregister, &regInfo->impInfo,
                                  sizeof(struct ncclIpcImpInfo), NULL, 0));
  INFO(NCCL_REG, "rank %d - IPC deregistered buffer %p peer %d ipc remote buffer %p", comm->rank, regInfo->baseAddr,
       regInfo->peerRank, regInfo->impInfo.rmtRegAddr);
  return ncclSuccess;
}

static ncclResult_t p2pProxyRegister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                     void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff;
  void* regAddr = NULL;
  ncclResult_t ret = ncclSuccess;
  assert(reqSize % sizeof(struct p2pIpcExpInfo) == 0);
  int numSegments = reqSize / sizeof(struct p2pIpcExpInfo);
  bool* mapped = nullptr;
  bool* imported = nullptr;
  CUmemGenericAllocationHandle* segmentHandles = nullptr;
  size_t totalSize = 0;
  NCCLCHECKGOTO(ncclCalloc(&segmentHandles, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&mapped, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&imported, numSegments), ret, fail);
  for (int segment = 0; segment < numSegments; segment++) {
    totalSize += ipcExpInfo[segment].size;
  }
  assert(sizeof(void*) == respSize);

  INFO(NCCL_REG,
       "Proxy rank %d register reqBuff %p size %zu offset %ld legacyIpcCap %d sameProcess %d, totalSize : %zu",
       proxyState->tpRank, reqBuff, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap,
       connection->sameProcess, totalSize);

  // 请求 对等端 passes 所有 necessary 缓冲区 信息 to import. The 代理 线程 would 寄存器
  // 该缓冲区 locally 并且 返回 寄存器 addr 后
  if (ipcExpInfo->legacyIpcCap) {
    // 旧版导入
    CUDACHECKGOTO(cudaIpcOpenMemHandle(&regAddr, ipcExpInfo->ipcDesc.devIpc, cudaIpcMemLazyEnablePeerAccess), ret,
                  fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  } else {
    CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)&regAddr, totalSize, /* alignment */ 0, /* addr */ 0, /* flags */ 0),
                ret, fail);
    size_t offset = 0;
    for (int segment = 0; segment < numSegments; segment++) {
      // cuMem 导入
      if (connection->sameProcess) {
        // 若 代理 is 相同 处理 as 请求 对等端, we 仅 需要 映射 the 句柄.
        memcpy(&segmentHandles[segment], &ipcExpInfo[segment].ipcDesc.memHandle, sizeof(CUmemGenericAllocationHandle));
      } else {
        if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
          CUCHECKGOTO(cuMemImportFromShareableHandle(&segmentHandles[segment],
                                                     (void*)(uintptr_t)ipcExpInfo[segment].impFd, ncclCuMemHandleType),
                      ret, fail);
          SYSCHECKGOTO(close(ipcExpInfo[segment].impFd), "close", ret, fail);
        } else {
          CUCHECKGOTO(cuMemImportFromShareableHandle(&segmentHandles[segment],
                                                     (void*)&ipcExpInfo[segment].ipcDesc.cuDesc.handle,
                                                     ncclCuMemHandleType),
                      ret, fail);
        }
      }
      imported[segment] = true;
      CUCHECKGOTO(cuMemMap((CUdeviceptr)regAddr + offset, ipcExpInfo[segment].size, /* offset */ 0,
                           segmentHandles[segment], /* flags */ 0),
                  ret, fail);
      offset += ipcExpInfo[segment].size;
      mapped[segment] = true;
    }
    // 授权本地 GPU 访问该内存
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = proxyState->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)regAddr, totalSize, &accessDesc, 1), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo[0].offset);
  }
  INFO(NCCL_REG, "Proxy rank %d register success regAddr %p size %ld offset %ld legacyIpcCap %d sameProcess %d",
       proxyState->tpRank, regAddr, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap,
       connection->sameProcess);

exit:
  if (regAddr) {
    struct proxyMemHandle* memHandle;
    struct ncclIpcImpInfo* ipcInfo;
    NCCLCHECK(ncclCalloc(&memHandle, 1));
    NCCLCHECK(ncclCalloc(&ipcInfo, 1));
    ipcInfo->rmtRegAddr = regAddr;
    ipcInfo->offset = ipcExpInfo[0].offset;
    ipcInfo->legacyIpcCap = ipcExpInfo->legacyIpcCap;
    ipcInfo->numSegments = numSegments;
    memHandle->handle = (void*)ipcInfo;
    ncclIntruQueueEnqueue(&connection->proxyMemHandleQueue, memHandle);
  }
  memcpy(respBuff, (void*)&regAddr, sizeof(void*));
  *done = 1;
  free(mapped);
  free(imported);
  free(segmentHandles);
  return ret;
fail:
  if (!ipcExpInfo->legacyIpcCap) {
    size_t offset = 0;
    for (int segment = 0; segment < numSegments; segment++) {
      if (mapped[segment]) {
        CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)regAddr + offset, ipcExpInfo[segment].size));
      }
      if (imported[segment]) {
        CUCHECKIGNORE(cuMemRelease(segmentHandles[segment]));
      }
    }
    if (regAddr) CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)regAddr, totalSize));
  }
  regAddr = NULL;
  goto exit;
}

static bool p2pHandleCmp(struct proxyMemHandle* a, struct proxyMemHandle* b) {
  struct ncclIpcImpInfo* ipcInfoA = (struct ncclIpcImpInfo*)a->handle;
  struct ncclIpcImpInfo* ipcInfoB = (struct ncclIpcImpInfo*)b->handle;
  return ipcInfoA->rmtRegAddr == ipcInfoB->rmtRegAddr && ipcInfoA->offset == ipcInfoB->offset &&
         ipcInfoA->legacyIpcCap == ipcInfoB->legacyIpcCap && ipcInfoA->numSegments == ipcInfoB->numSegments;
}

static ncclResult_t p2pProxyDeregister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                       void* reqBuff, int reqSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcImpInfo* ipcInfo = (struct ncclIpcImpInfo*)reqBuff;
  assert(sizeof(struct ncclIpcImpInfo) == reqSize);

  struct proxyMemHandle memHandle = {};
  struct proxyMemHandle* deletedHandle;
  memHandle.handle = (void*)ipcInfo;
  deletedHandle = ncclIntruQueueDelete(&connection->proxyMemHandleQueue, &memHandle, p2pHandleCmp);
  free(deletedHandle->handle);
  free(deletedHandle);

  NCCLCHECKGOTO(p2pDeregisterMemHandle(connection, proxyState, ipcInfo), ret, fail);

exit:
  *done = 1;
  return ret;
fail:
  goto exit;
}

// 把 P2P 传输层注册到 NCCL：名字 "P2P"，并挂上上面实现的 设置/connect/progress
// 等函数表。enqueue/transport 层据此选择 "P2P" 作为本机 GPU 间的传输后端。
struct ncclTransport p2pTransport = {"P2P",
                                     p2pCanConnect,
                                     {p2pSendSetup, p2pSendConnect, p2pSendFree, NULL, p2pSendProxySetup, NULL,
                                      p2pSendProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister},
                                     {p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL, p2pRecvProxySetup, NULL,
                                      p2pRecvProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister}};

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

// 函数 to 检查 若 P2P is 使用 memcpy (for 注册 优化)
bool ncclP2pUsesMemcpy() {
  initCeOperation(); // Ensure initialization
  return useMemcpy != 0;
}
