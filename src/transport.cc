/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/transport.cc — 传输层(transport)注册与选择
 * ----------------------------------------------------------------------------
 * 维护 ncclTransports[] 全局数组（登记 P2P/NVLS/Net/CollNet 等各传输实现），并提供
 * 传输选择逻辑：根据 comm 配置与拓扑，为给定 channel 选出可用的 transport。是“算法
 * 图”与“具体建链(transport/*.cc)”之间的分发枢纽。
 */

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
#define ENABLE_TIMER 0
#include "timer.h"
#include "transport.h"

struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {
  &p2pTransport, &shmTransport, &netTransport, &collNetTransport,
  &profilerTransport // Not really used for transport, only to create proxy ops polling on profiler counters.
};

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect,
                                    int channelId, int peer, int connIndex, int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo + comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo + peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t = 0; t < NTRANSPORTS; t++) {
    struct ncclTransport* transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank,
       peerInfo->busId);
  return ncclSystemError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend,
                                     int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  uint64_t mask = 1ULL << channel->id;
  for (int i = 0; i < nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) {
      continue;
    }
    comm->connectRecv[peer] |= mask;
  }
  for (int i = 0; i < nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) {
      continue;
    }
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n = 0; n < ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i = 0; i < sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);

#include "os.h"

// 检测通信域内的 CUDA P2P 连通性(仅针对本地 rank)。
// *isAllDirectP2p 返回 1 表示：所有本地 rank 之间都具备 CUDA P2P 连通性，
// 且彼此距离不超过 NCCL_P2P_LEVEL 限定的范围。
// *directMode 返回 1 表示：存在任意两个本地 rank 由同一个进程管理。
// *isAllCudaP2p 返回 1 表示：所有本地 rank 之间都具备 CUDA P2P 连通性(不考虑距离限制)。
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode,
                                       bool* isAllCudaP2p) {
  bool ncclP2pFlag = true;
  bool directFlag = false;
  bool cudaP2pFlag = true;
  for (int i = 0; i < comm->localRanks; ++i) {
    for (int j = i + 1; j < comm->localRanks; ++j) {
      int ipeer = comm->localRankToRank[i];
      int jpeer = comm->localRankToRank[j];
      struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
      struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
      int canConnect = 0;
      int intermediateRank = -1;
      int cudaP2p = 0;
      NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, ipeerInfo->rank, jpeerInfo->rank, &canConnect, NULL,
                                 &intermediateRank, &cudaP2p));
      if (!canConnect || intermediateRank != -1) {
        ncclP2pFlag = false;
      }
      if (!cudaP2p) {
        cudaP2pFlag = false;
      }
      if (ipeerInfo->hostHash == jpeerInfo->hostHash && ipeerInfo->pidHash == jpeerInfo->pidHash) {
        directFlag = true;
      }
      if (!ncclP2pFlag && directFlag && !cudaP2pFlag) {
        break;
      }
    }
  }
  *isAllDirectP2p = ncclP2pFlag;
  *directMode = directFlag;
  *isAllCudaP2p = cudaP2pFlag;
  INFO(NCCL_INIT, "Check P2P Type isAllDirectP2p %d directMode %d isAllCudaP2p %d", *isAllDirectP2p, *directMode,
       *isAllCudaP2p);
  return ncclSuccess;
}

/*
 * ncclTransportP2pSetup —— 传输层建链总入口
 * ----------------------------------------------------------------------------
 * 在 AllReduce 全链路中的位置：bootstrap 建好引导环、graph 算好拓扑之后，
 * 由本函数真正为每一对需要通信的 (rank, channel) 建立底层连接。
 *
 * 建链是一个“三段式握手”过程：
 *   1) setup   : 本地准备资源(分配缓冲区、生成 IPC 句柄等)，产出 ncclConnect 信息
 *   2) exchange: 通过 bootstrap 把 ncclConnect 信息发给对端、并收下对端的
 *   3) connect : 用对端信息完成本地连接对象的最终装配
 * 必须分三段，是因为双方都要拿到对方的信息才能完成映射，无法一步到位。
 *
 * 为了避免一次性和所有 rank 握手造成的资源峰值(以及 socket 数爆炸)，
 * 这里采用**分轮次(round)**处理：每轮最多处理 maxPeers 个对端。
 */
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  // 建链过程中使用的 CUDA 流：P2P 预连接与 CUDA 图 场景都需要它
  ncclResult_t ret = ncclSuccess;
  struct ncclConnect** data; // 暂存握手用的 send/recv 连接信息结构体
  struct ncclConnect** recvData = NULL; // 指向 data 内部：某 channel 上接收连接对应的条目
  struct ncclConnect** sendData = NULL; // 指向 data 内部：某 channel 上发送连接对应的条目
  int done = 0;                                  // 已完成建链的对端数量
  int maxPeers = ncclParamConnectRoundMaxPeers(); // 每轮并发处理的最大对端数(控制资源峰值)

  struct timeval timeStart, timeLast;
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart; // struct copy
  bool timeReported = false;
  cudaStream_t hostStream, deviceStream;

  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                        /*concurrent=*/false, &hostStream),
                ret, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false, &deviceStream),
                ret, fail);
  // 首轮初始化：按“距离 i”依次与各对端配对握手
  for (int i = 1; i < comm->nRanks; i++) {
    // bootstrapTag 用于区分不同轮次、不同拓扑图的握手消息，避免消息串扰。
    // 低 8 位放 图 id，高位放距离 i，组合成唯一标签。
    int bootstrapTag = (i << 8) + (graph ? graph->id + 1 : 0);
    // 关键设计：接收方向取 rank-i，发送方向取 rank+i。
    // 这样在同一轮里，每个 rank 恰好有一个发送目标和一个接收来源，
    // 且全局形成一个完美配对(我发给谁，谁就正好在等我)，不会出现多打一的死锁。
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];  // 位图：与该对端在哪些 channel 上需要建收连接
    uint64_t sendMask = comm->connectSend[sendPeer];  // 位图：与该对端在哪些 channel 上需要建发连接

    /* data[p] 的内存布局说明：
     *   它保存了与某一对 (发送对端, 接收对端) 之间**所有** channel 的连接信息，
     *   按照实际连接的 recvChannels / sendChannels 数量紧凑打包：
     *     前 N 项 = recvData，存放各接收连接的信息
     *     后 M 项 = sendData，存放各发送连接的信息
     *   注意：不保证每个 data[p] 拥有相同数量的连接(不同对端可能用不同数量的 channel)，
     *   因此必须靠 recvChannels/sendChannels 计数来定位，不能按固定步长索引。
     */
    int p = i - (done + 1);
    if (recvMask || sendMask) {
      if (data[p] == NULL) NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
    }
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;
    TIME_START(0);
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (recvMask & (1ULL << c)) {
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p] + recvChannels++, c, recvPeer, connIndex, &type), ret,
                      fail);
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[p] = recvData[p] + recvChannels;
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (sendMask & (1ULL << c)) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p] + sendChannels++, c, sendPeer, connIndex, &type), ret,
                      fail);
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    if (sendPeer == recvPeer) {
      if (recvChannels + sendChannels) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p],
                                    sizeof(struct ncclConnect) * (recvChannels + sendChannels)),
                      ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p],
                                    sizeof(struct ncclConnect) * (recvChannels + sendChannels)),
                      ret, fail);
        sendData[p] = data[p];
        recvData[p] = data[p] + sendChannels;
      }
    } else {
      if (recvChannels) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p],
                                    sizeof(struct ncclConnect) * recvChannels),
                      ret, fail);
      }
      if (sendChannels) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p],
                                    sizeof(struct ncclConnect) * sendChannels),
                      ret, fail);
      }
      if (sendChannels) {
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p],
                                    sizeof(struct ncclConnect) * sendChannels),
                      ret, fail);
      }
      if (recvChannels) {
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p],
                                    sizeof(struct ncclConnect) * recvChannels),
                      ret, fail);
      }
    }
    TIME_STOP(2);

    if (i - done == maxPeers || i == comm->nRanks - 1) {
      // 循环直到与所有 rank 的所有 通道 都完成连接为止
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j = done + 1; j <= i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j - (done + 1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c = 0; c < MAXCHANNELS; c++) {
            TIME_START(3);
            if (sendMask & (1ULL << c)) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // 此 connector hasn't 已完成 连接 yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn),
                              ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex],
                                                &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice,
                                                hostStream),
                                ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }
            TIME_STOP(3);

            // 先处理接收方向的 通道
            TIME_START(4);
            if (recvMask & (1ULL << c)) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // 此 connector hasn't 已完成 连接 yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn),
                              ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] is a device memory access. */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex],
                                                &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice,
                                                hostStream),
                                ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
            TIME_STOP(4);
          }
        }
        if (ncclParamReportConnectProgress() && comm->rank == 0 && done > 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          if (((now.tv_sec - timeLast.tv_sec) * 1.0 + (now.tv_usec - timeLast.tv_usec) * 1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec) * 1.0 + (now.tv_usec - timeStart.tv_usec) * 1e-6;
            float remaining = elapsed * (comm->nRanks - done) / done;
            printf("%sP2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d                                       ",
                   timeReported ? "\r" : "", done * 100.0 / comm->nRanks, ((int)elapsed) / 60, ((int)elapsed) % 60,
                   ((int)remaining) / 60, ((int)remaining) % 60);
            fflush(stdout);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;
    }
  }

  {
    struct timeval now;
    gettimeofday(&now, NULL);
    float elapsed = (now.tv_sec - timeStart.tv_sec) * 1.0 + (now.tv_usec - timeStart.tv_usec) * 1e-6;
    if (elapsed > 1.0) {
      INFO(NCCL_PROFILE, "timings: rank %d nranks %d P2p connect done in %.2f", comm->rank, comm->nRanks, elapsed);
    }
    if (timeReported) {
      printf("\rP2p connect done in %d:%02d                                                                       \n",
             ((int)elapsed) / 60, ((int)elapsed) % 60);
      fflush(stdout);
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
      if (comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      }
      if (comm->connectSend[sendPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
      if (comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

  TIME_PRINT("P2P Setup/Connect");
exit:
  for (int i = 0; i < maxPeers; ++i) {
    if (data[i]) free(data[i]);
  }
  free(data);
  if (sendData) free(sendData);
  if (recvData) free(recvData);

  NCCLCHECK(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                    /*concurrent=*/false));
  return ret;
fail:
  goto exit;
}

extern struct ncclTransport collNetTransport;

// 所有 ranks must participate 入 collNetSetup 调用
// 这里刻意不用 NCCLCHECK 包裹：CollNet 初始化失败属于可接受情况，会自动回退到 P2P 网络
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel,
                               int masterRank, int masterPeer, int collNetGraphChannelId, int type,
                               ncclConnect* connect) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int isMaster = (rank == masterRank) ? 1 : 0;

  // 检查能否连接到 collnet，其根节点是第 nranks 个 rank(额外预留的那个)
  struct ncclPeerInfo *myInfo = comm->peerInfo + rank, *peerInfo = comm->peerInfo + nranks;
  peerInfo->rank = nranks;

  if (isMaster && type == collNetSend) {
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank,
          comm->node, nMasters, masterPeer);
  }

  // 选择
  struct ncclChannelPeer* root = channel->peers[nranks];
  // 连接器下标约定：0 表示接收，1 表示发送
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv + type : root->send + type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // 设置
  struct ncclConnect myConnect = {0};
  struct {
    int isMaster;
    ncclConnect connect;
  }* allConnects = NULL;
  ncclConnect* masterConnects = NULL;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId,
                                   type));
  }
  // prepare connect 句柄
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {
    // 接收 side: 全收集
    // 所有 ranks must participate
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), ret, cleanup);
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), ret, cleanup);
    // 合并
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects + c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        c++;
      }
    }
  } else {
    // 发送 side : 拷贝 入 connect 信息 received from 对等端 接收 master
    if (isMaster) memcpy(masterConnects + comm->node, connect, sizeof(struct ncclConnect));
  }
  // 连接
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, comm->node, conn), ret, cleanup);
    struct ncclDevChannelPeer* devRoot;
    CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*),
                             cudaMemcpyDeviceToHost),
                  ret, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), ret,
                  cleanup);
  }
  if (isMaster && type == collNetRecv) {
    memcpy(connect, masterConnects + comm->node, sizeof(struct ncclConnect));
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank,
          comm->node, nMasters, masterPeer);
  }
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return ret != ncclSuccess;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // 全收集 collNet 设置 results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                        allGatherFailures, sizeof(int)));
  for (int i = 0; i < comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // 释放 collNet resources
  for (int r = 0; r < comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels + r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b = 0; b < NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(comm, send));
          send->transportResources = NULL; // avoid double free
        }
        for (int b = 0; b < NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(comm, recv));
          recv->transportResources = NULL; // avoid double free
        }
      }
    }
  }
  return ncclSuccess;
}
