/*************************************************************************
 * mini-nccl: Linux stub implementations for symbols provided in the full
 * NCCL by plugin/, ras/, gin/, rma/ and nccl_device/gin_*.cc.
 *
 * This lean single-node build keeps only: bootstrap, topology detection,
 * P2P/SHM transports, ring/tree algorithms and simple/LL/LL128 protocols.
 * Networking plugins, RAS, GIN (GPU-initiated networking) and RMA are
 * stubbed out. Adapted from os/windows_stubs.cc of the original NCCL.
 *************************************************************************/

#include "nccl.h"
#include "comm.h"
#include "checks.h"
#include "net.h"
#include "ras.h"
#include "profiler.h"
#include "env.h"
#include "tuner.h"
#include "gin/gin_host.h"
#include "rma.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"
#include "device.h"
#include "transport.h"

#include <cstring>
#include <cstdlib>
#include <mutex>

/* Cached socket transport device counts (populated during ncclNetInit) */
static int miniNetPhysDev = 0;
static int miniNetVirtDev = -1; /* NCCL_UNDEF_DEV_COUNT */
static bool miniNetInitialized = false;
static std::mutex miniNetInitMutex;

/* --------------------------------------------------------------------------
 * RAS stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank) {
  (void)comm; (void)myRank;
  return ncclSuccess;
}

ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks) {
  (void)ranks; (void)nranks;
  return ncclSuccess;
}

ncclResult_t ncclRasCommFini(const struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Net plugin stubs: use built-in socket transport without plugin layer
 * -------------------------------------------------------------------------- */
ncclResult_t ncclNetInit(struct ncclComm* comm) {
  comm->ncclNet = &ncclNetSocket;
  comm->ncclCollNet = nullptr;
  comm->netPluginIndex = -1;

  // Initialize the socket transport to enumerate network interfaces.
  ncclNetCommConfig_t commConfig = {};
  commConfig.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  NCCLCHECK(comm->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ncclDebugLog, NULL));

  {
    std::lock_guard<std::mutex> lock(miniNetInitMutex);
    if (!miniNetInitialized) {
      int ndev = 0;
      NCCLCHECK(comm->ncclNet->devices(&ndev));
      miniNetPhysDev = ndev;
      miniNetInitialized = true;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->netContext = parent->netContext;
  comm->collNetContext = parent->collNetContext;
  comm->ncclNet = parent->ncclNet;
  comm->ncclCollNet = parent->ncclCollNet;
  comm->netPluginIndex = parent->netPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  if (comm->ncclNet && comm->netContext) NCCLCHECK(comm->ncclNet->finalize(comm->netContext));
  return ncclSuccess;
}

ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport) {
  (void)comm;
  *gdrSupport = 0;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Env plugin stubs
 * -------------------------------------------------------------------------- */
bool ncclEnvPluginInitialized(void) {
  return false;
}

ncclResult_t ncclEnvPluginInit(void) {
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void) {}

const char* ncclEnvPluginGetEnv(const char* name) {
  return std::getenv(name);
}

/* --------------------------------------------------------------------------
 * Net/CollNet dev count stubs (plugin provides these in full NCCL)
 * -------------------------------------------------------------------------- */
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)netPluginIndex;
  *nPhysDev = miniNetPhysDev;
  *nVirtDev = miniNetVirtDev;
  return ncclSuccess;
}

ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDev) {
  (void)netPluginIndex;
  miniNetVirtDev = nVirtDev;
  return ncclSuccess;
}

ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)netPluginIndex;
  *nPhysDev = 0;
  *nVirtDev = 0;
  return ncclSuccess;
}

ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDev) {
  (void)netPluginIndex; (void)nVirtDev;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * GIN stubs (GPU-initiated networking is multi-node only)
 * -------------------------------------------------------------------------- */
ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                             ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) {
  (void)comm; (void)team; (void)nBarriers; (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t ncclGinOutboxCreateRequirement(int nBlocks, int size_log2, ncclGinOutboxHandle* outHandle,
                                            ncclDevResourceRequirements_t* outReq) {
  (void)nBlocks; (void)size_log2; (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t ncclGinInboxA2ACreateRequirement(ncclTeam peers, int nBlocks, int size_log2,
                                              ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements_t* outReq) {
  (void)peers; (void)nBlocks; (void)size_log2; (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  (void)comm;
  *ginType = (ncclGinType_t)0; /* NCCL_GIN_TYPE_NONE */
  return ncclSuccess;
}

ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  (void)comm;
  *ginType = (ncclGinType_t)0; /* NCCL_GIN_TYPE_NONE */
  return ncclSuccess;
}

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm) {
  (void)comm; (void)reqs; (void)devComm;
  return ncclSuccess;
}

ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  (void)comm; (void)devComm;
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags, bool multiSegment,
                             int memType) {
  (void)comm; (void)address; (void)size; (void)ginHostWins; (void)ginDevWins;
  (void)winFlags; (void)multiSegment; (void)memType;
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  (void)comm; (void)ginHostWins;
  return ncclSuccess;
}

ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)ginPluginIndex;
  *nPhysDev = 0;
  *nVirtDev = 0;
  return ncclSuccess;
}

ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  (void)ginState;
  if (hasError) *hasError = false;
  return ncclSuccess;
}

ncclResult_t ncclGinInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  (void)comm; (void)parent;
  return ncclSuccess;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * RMA stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclRmaInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  (void)comm; (void)parent;
  return ncclSuccess;
}

ncclResult_t ncclRmaGetDevCount(int rmaPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)rmaPluginIndex;
  if (nPhysDev) *nPhysDev = 0;
  if (nVirtDev) *nVirtDev = -1;
  return ncclSuccess;
}

ncclResult_t ncclRmaFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Profiler plugin stubs and globals
 * -------------------------------------------------------------------------- */
thread_local ncclProfilerApiState_t ncclProfilerApiState = {0,       0,       ncclProfilerGroupApiStartStateReset,
                                                            nullptr, nullptr, nullptr};

int ncclProfilerEventMask = 0;

bool ncclProfilerPluginLoaded(void) {
  return false;
}

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info; (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info; (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopP2pApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info; (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCollApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  (void)plan; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  (void)op;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args) {
  (void)sub; (void)args;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args) {
  (void)sub; (void)args;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub; (void)args; (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub; (void)args; (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub; (void)args; (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  (void)profilerContext;
  if (eHandle) *eHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  (void)eHandle;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start) {
  (void)args; (void)s; (void)start;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop) {
  (void)args; (void)s; (void)stop;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  (void)sub; (void)args; (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId,
                                                   ncclProfilerEventState_t eState) {
  (void)sub; (void)args; (void)stepId; (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  (void)eHandle; (void)appended; (void)eState;
  return ncclSuccess;
}

bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op) {
  (void)comm; (void)op;
  return false;
}

/* CE profiler stubs */
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  (void)comm; (void)args; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  (void)comm; (void)args; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream,
                                          void** ceSyncHandle) {
  (void)comm; (void)args; (void)stream;
  if (ceSyncHandle) *ceSyncHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle, cudaStream_t stream) {
  (void)comm; (void)ceSyncHandle; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                           void** ceBatchHandle) {
  (void)comm; (void)args; (void)params; (void)stream;
  if (ceBatchHandle) *ceBatchHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream) {
  (void)comm; (void)ceBatchHandle; (void)stream;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * RMA proxy / CE / scheduler stubs (RMA is a multi-node feature)
 * -------------------------------------------------------------------------- */
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  (void)comm; (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  (void)comm; (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclRmaCeInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclRmaCeFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  (void)comm; (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
                                  void** rmaHostWins) {
  (void)comm; (void)address; (void)size; (void)rmaHostWins;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  (void)comm; (void)rmaHostWins;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyPutBuildOp(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx, int ctx,
                                    bool persistent, struct ncclDevrWindow* srcWin, size_t srcOff,
                                    struct ncclDevrWindow* peerWin, size_t peerOff, size_t size, int peer,
                                    ncclSignalMode_t signalMode, struct ncclRmaPutSignalOp* op) {
  (void)comm; (void)rmaProxyCtx; (void)ctx; (void)persistent; (void)srcWin; (void)srcOff;
  (void)peerWin; (void)peerOff; (void)size; (void)peer; (void)signalMode; (void)op;
  return ncclInternalError;
}

ncclResult_t ncclRmaProxyPutGroupBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                           struct ncclKernelPlan* plan, int nOps, struct ncclRmaPutSignalOp** ops,
                                           int ctx, struct ncclRmaProxyDesc* desc) {
  (void)comm; (void)rmaProxyCtx; (void)plan; (void)nOps; (void)ops; (void)ctx; (void)desc;
  return ncclInternalError;
}

ncclResult_t ncclRmaProxyWaitBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, int npeers, int** peers, int** nsignals,
                                       struct ncclRmaProxyDesc* desc) {
  (void)comm; (void)rmaProxyCtx; (void)plan; (void)npeers; (void)peers; (void)nsignals; (void)desc;
  return ncclInternalError;
}

int ncclRmaProxyPutGroupStartNumOps(bool persistent) {
  (void)persistent;
  return 0;
}

ncclResult_t ncclRmaProxyPutGroupStartParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  (void)desc; (void)params;
  return ncclSuccess;
}

int ncclRmaProxyPutGroupDoneNumOps(bool persistent) {
  (void)persistent;
  return 0;
}

ncclResult_t ncclRmaProxyPutGroupDoneParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  (void)desc; (void)params;
  return ncclSuccess;
}

int ncclRmaProxyWaitNumStreamOps(const struct ncclRmaProxyDesc* desc) {
  (void)desc;
  return 0;
}

ncclResult_t ncclRmaProxyWaitParams(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc,
                                    CUstreamBatchMemOpParams* params) {
  (void)rmaProxyCtx; (void)desc; (void)params;
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc** desc) {
  (void)rmaProxyCtx; (void)desc;
  return ncclInternalError;
}

ncclResult_t ncclRmaProxyDestroyDesc(struct ncclComm* comm, struct ncclRmaProxyDesc** desc) {
  (void)comm;
  if (desc) *desc = nullptr;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * NVLS stubs not covered by the disabled branch in transport/nvls.cc
 * -------------------------------------------------------------------------- */
ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclNvlsGroupCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
                                 unsigned long long* mcHandle, char* shareableHandle) {
  (void)comm; (void)prop; (void)rank; (void)nranks; (void)mcHandle; (void)shareableHandle;
  return ncclInternalError;
}

ncclResult_t ncclNvlsGroupConnect(struct ncclComm* comm, char* shareableHandle, int rank,
                                  unsigned long long* mcHandle) {
  (void)comm; (void)shareableHandle; (void)rank; (void)mcHandle;
  return ncclInternalError;
}

/* --------------------------------------------------------------------------
 * Tuner plugin stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}
