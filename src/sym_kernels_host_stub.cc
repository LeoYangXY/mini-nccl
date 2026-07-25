/*************************************************************************
 * mini-nccl: stub host table for symmetric kernels.
 *
 * The full NCCL generates this table from src/device/symmetric/. In this
 * lean build symmetric kernels are removed: ncclSymkAvailable() always
 * returns false (see sym_kernels.cc), so these entries are never used.
 * ncclInitKernelsForDevice() in enqueue.cc safely skips nullptr entries.
 *************************************************************************/

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
