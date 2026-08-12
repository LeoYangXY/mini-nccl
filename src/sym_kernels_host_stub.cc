/*************************************************************************
 * mini-nccl: stub host table for symmetric kernels.
 *
 * The full NCCL generates this table from src/device/symmetric/. In this
 * lean build symmetric kernels are removed: ncclSymkAvailable() always
 * returns false (see sym_kernels.cc), so these entries are never used.
 * ncclInitKernelsForDevice() in enqueue.cc safely skips nullptr entries.
 *************************************************************************/

/*
 * src/sym_kernels_host_stub.cc — symmetric kernel 的 host 桩表（精简版）
 * ----------------------------------------------------------------------------
 * 本文件是“对称内存 kernel”的 host 侧桩表。完整版 NCCL 会从 src/device/symmetric/
 * 生成该表；mini-nccl 精简构建里对称 kernel 被移除：ncclSymkAvailable() 恒返回
 * false（见 sym_kernels.cc），因此这些表项永不被使用，enqueue.cc 的
 * ncclInitKernelsForDevice() 会安全地跳过 nullptr 项。
 */

#include "sym_kernels.h"
#include "debug.h"

extern int const ncclSymkKernelCount = 0;
void* ncclSymkKernelList[1] = {nullptr};
int ncclSymkKernelRequirements[1] = {0};
int ncclSymkKernelMaxDynamicSmem[1] = {0};

int ncclSymkGetKernelIndex(ncclSymkKernelId id, int red, ncclDataType_t ty) {
  (void)id; (void)red; (void)ty;
  WARN("mini-nccl: symmetric kernels are not built");
  return -1;
}
