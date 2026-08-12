/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/misc/utils.cc — 通用工具实现
 * ----------------------------------------------------------------------------
 * 实现 include/utils.h 的通用工具：字符串/时间/数值解析、字节单位换算、bit 操作、
 * 链表/队列等，被各模块广泛引用，属于“杂项工具箱”。
 */

#include "utils.h"
#include "core.h"
#include "os.h"

#include "nvmlwrap.h"

#include <stdlib.h>
#include <mutex>

// 获取 当前的 计算 能力
int ncclCudaCompCap() {
  int cudaDev;
  if (!CUDASUCCESS(cudaGetDevice(&cudaDev))) return 0;
  int ccMajor, ccMinor;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev))) return 0;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev))) return 0;
  return ccMajor * 10 + ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (unsigned long)((id) >> 20), (unsigned long)((id & 0xff000) >> 12),
          (unsigned long)((id & 0xff0) >> 4), (unsigned long)(id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
  char hexStr[17];  // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else {
      break;
    }
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return ncclSuccess;
}

// 获取 an int64 from a PCI 路径. 例如, sys/类/pci0000:00/0000:00:02.0/0000:02:00.0/ will 返回 0x000002000.
ncclResult_t pciPathToInt64(char* path, int64_t* id) {
  char* str = path + strlen(path) - 1;
  // 移除末尾的 "/"
  if (*str == '/') str--;
  // 查找下一个 /
  while (*str != '/') str--;
  str++;
  NCCLCHECK(busIdToInt64(str, id));
  return ncclSuccess;
}

// 将 ... 转换 logical cudaDev 索引 到 NVML 设备 minor number
ncclResult_t getBusId(int cudaDev, int64_t* busId) {
  // On most systems, the PCI 总线 ID comes 后 as 在 ... 中 0000:00:00.0
  // 格式. 仍需 to 分配 proper space 以防 PCI 域 goes
  // 更高。
  char busIdStr[] = "00000000:00:00.0";
  CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

static uint64_t hostHashValue = 0;
/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the NCCL_HOSTID env var.
 */
#if defined(NCCL_OS_LINUX)

#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"

#elif defined(NCCL_OS_WINDOWS)

/* Get Windows MachineGuid - similar to boot_id on Linux */
static bool getWindowsMachineGuid(char* guid, size_t len) {
  HKEY hKey;
  LONG result = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0, KEY_READ, &hKey);
  if (result != ERROR_SUCCESS) {
    return false;
  }
  DWORD dataSize = (DWORD)len;
  DWORD dataType;
  result = RegQueryValueExA(hKey, "MachineGuid", NULL, &dataType, (LPBYTE)guid, &dataSize);
  RegCloseKey(hKey);
  if (result != ERROR_SUCCESS || dataType != REG_SZ) {
    return false;
  }
  return true;
}
#endif

static void getHostHashOnce() {
  char hostHash[1024];
  const char* hostId;

  // 回退为 the 满的 hostname 若是如此mething 失败
  (void)getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = ncclGetEnv("NCCL_HOSTID")) != NULL) {
    INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash) - 1);
    hostHash[sizeof(hostHash) - 1] = '\0';
  } else {
#if defined(NCCL_OS_LINUX)
    FILE* file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char* p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
        free(p);
      }
      fclose(file);
    }
#elif defined(NCCL_OS_WINDOWS)
    char machineGuid[256];
    if (getWindowsMachineGuid(machineGuid, sizeof(machineGuid))) {
      strncpy(hostHash + offset, machineGuid, sizeof(hostHash) - offset - 1);
    }
#endif
  }

  // 确保该 string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';

  TRACE(NCCL_INIT, "unique hostname '%s'", hostHash);

  hostHashValue = getHash(hostHash, strlen(hostHash));
}
uint64_t getHostHash(void) {
  static std::once_flag once;
  std::call_once(once, getHostHashOnce);
  return hostHashValue;
}

uint64_t hashCombine(uint64_t baseHash, uint64_t value) {
  uint64_t hacc[2] = {1, 1};
  eatHash(hacc, &baseHash);
  eatHash(hacc, &value);
  return digestHash(hacc);
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Linux: hash of $$ $(readlink /proc/self/ns/pid) (pid + PID namespace)
 * Windows: hash of PID only (no namespaces; PID is unique system-wide)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // 以 ... 开始 our pid ($$)
  sprintf(pname, "%ld", (long)ncclOsGetPid());
  int plen = strlen(pname);
#if defined(NCCL_OS_LINUX)
  int len = readlink("/proc/self/ns/pid", pname + plen, sizeof(pname) - 1 - plen);
  if (len < 0) len = 0;
  plen += len;
#endif
  pname[plen] = '\0';
  TRACE(NCCL_INIT, "unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  char* str = strdup(string);
  if (!str) return 0;

  int ifNum = 0;
  char* savePtr = NULL;
  char* entry = strtok_r(str, ",", &savePtr);
  while (entry != NULL && ifNum < maxList) {
    char* c = entry;
    char* tok = ncclOsStrSep(&c, ":");
    if (tok && tok[0] != '\0') {
      snprintf(ifList[ifNum].prefix, sizeof(ifList[ifNum].prefix), "%s", tok);
      // 端口, rail, 并且 plane will 默认 to -1 若 absent 或者 空的
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].port = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].rail = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].plane = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      ifNum++;
    }
    entry = strtok_r(NULL, ",", &savePtr);
  }
  free(str);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // 务必确保 包含 '\0' 在 ... 中 exact 情形
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact, int* ifId) {
  // 做个例外 为了 情形 何処 无 用户 列表 被定义为
  if (ifId) *ifId = -1;
  if (listSize == 0) return true;

  for (int i = 0; i < listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact) && matchPort(port, ifList[i].port)) {
      if (ifId) *ifId = i;
      return true;
    }
  }
  return false;
}

thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance;

void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  // `me->hunks` points 到 顶 的 栈 non-空的 hunks. Hunks 上方
  // 此 (reachable via `->上方`) are 空的.
  struct Hunk* top = me->topFrame.hunk;
  size_t mallocSize = 0;

  // 若 我们已有 lots of space 左 入 hunk 但 那个 wasn't enough then we'll
  // 分配 object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8 << 10) goto unhunked;

  // 若 我们已有 另一个 hunk (该 必须为 空的) waiting 上方 此 one 并且
  // the object fits then 使用 那个.
  if (top && top->above) {
    struct Hunk* top1 = top->above;
    uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align - 1) & -uintptr_t(align);
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk 代理 fits 入 当前 hunk then go 分配 as unhunked.
    size_t nextSize = (top ? top->size : 0) + (64 << 10);
    constexpr size_t maxAlign = 64;
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk) - 1) & -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end) goto unhunked;
    }

    // 此时 we must 需要 另一个 hunk, 二者之一 to fit the object
    // itself 或者 its Unhunk 代理.
    mallocSize = nextSize;
    INFO_LOC(NCCL_ALLOC_HOST, "memory stack hunk malloc(%llu)", (unsigned long long)mallocSize);
    struct Hunk* top1 = (struct Hunk*)malloc(mallocSize);
    if (top1 == nullptr) goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top) top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper = reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align - 1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void*>(uobj);
    }
  }

unhunked:
  { // We need to allocate the object out-of-band and put an Unhunk proxy in-band
    // to 保留 track of it.
    uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk) - 1) & -uintptr_t(alignof(Unhunk));
    Unhunk* proxy = reinterpret_cast<Unhunk*>(uproxy);
    me->topFrame.bumper = uproxy + sizeof(Unhunk);
    proxy->next = me->topFrame.unhunks;
    me->topFrame.unhunks = proxy;
    mallocSize = size;
    proxy->obj = malloc(mallocSize);
    INFO_LOC(NCCL_ALLOC_HOST, "memory stack non-hunk malloc(%llu)", (unsigned long long)mallocSize);
    if (proxy->obj == nullptr) goto malloc_exhausted;
    return proxy->obj;
  }

malloc_exhausted:
  WARN("Unrecoverable error detected: malloc(size=%llu) returned null.", (unsigned long long)mallocSize);
  abort();
}

void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // 释放 unhunks 第一 因为 两者 the frames 并且 unhunk 代理 lie with在 ... 中 hunks.
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // 释放 hunks
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk* h1 = h->above;
    free(h);
    h = h1;
  }
}

/* return concatenated string representing each set bit */
ncclResult_t ncclBitsToString(uint32_t bits, uint32_t mask, const char* (*toStr)(int), char* buf, size_t bufLen,
                              const char* wildcard) {
  if (!buf || !bufLen) return ncclInvalidArgument;

  bits &= mask;

  // 打印 wildcard 值 若 所有 位 设置
  if (wildcard && bits == mask) {
    snprintf(buf, bufLen, "%s", wildcard);
    return ncclSuccess;
  }

  // Add 每个 设置 位 to string
  int pos = 0;
  for (int i = 0; bits; i++, bits >>= 1) {
    if (bits & 1) {
      if (pos > 0) pos += snprintf(buf + pos, bufLen - pos, "|");
      pos += snprintf(buf + pos, bufLen - pos, "%s", toStr(i));
    }
  }

  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// Hash 函数 for 指针 类型 (shared by 地址 映射 实现)
// 使用 shadowpool's 算法
uint64_t ncclHashPointer(int hbits, void* key) {
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13;
  return (uint64_t)h >> (64 - hbits);
}

////////////////////////////////////////////////////////////////////////////////
// Intrusive 地址 映射 实现 (untyped core 函数)

// 辅助: 读取 key from object at 给定的 偏移
// Key 必须为 convertible to uintptr_t, 所以 we 读取 keySize 字节 并且 zero-extend
static inline uintptr_t readKey(void* object, int keySize, int keyFieldOffset) {
  void* keyPtr = (char*)object + keyFieldOffset;
  uintptr_t result = 0;
  memcpy(&result, keyPtr, keySize);
  return result;
}

// 辅助: 读取 下一个 指针 from object at 给定的 偏移
// 使用 memcpy to 避免 strict aliasing violations 当 actual 类型 is T*, 不 void*
static inline void* readNextPtr(void* object, int nextFieldOffset) {
  void* nextPtr = (char*)object + nextFieldOffset;
  void* result = nullptr;
  memcpy(&result, nextPtr, sizeof(void*));
  return result;
}

// 辅助: 写入 下一个 指针 to object at 给定的 偏移
// 使用 memcpy to 避免 strict aliasing violations 当 actual 类型 is T*, 不 void*
static inline void writeNextPtr(void* object, int nextFieldOffset, void* value) {
  void* nextPtr = (char*)object + nextFieldOffset;
  memcpy(nextPtr, &value, sizeof(void*));
}

ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key, void* object) {
  // Runtime 校验
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Lazy 初始化 - 创建 table on 第一 insert
  if (map->hbits == 0) {
    const int initHbits = 4;
    const size_t tableEntries = (size_t)1 << initHbits;
    map->hbits = initHbits;
    map->table = (void**)calloc(tableEntries, sizeof(void*));
    if (map->table == nullptr) {
      map->hbits = 0; // Reset on failure
      WARN("Intrusive address map initialization failed: calloc(%zu entries) returned null", tableEntries);
      return ncclSystemError;
    }
  }

  int hbits = map->hbits;

  // 检查 for 地址 映射 大小 increase 之前 inserting. Maintain 2:1 object:bucket ratio.
  if (map->count + 1 > 2 << hbits) {
    int oldHbits = hbits;
    int oldSize = 1 << oldHbits;
    int newHbits = hbits + 1;
    int newSize = 1 << newHbits;

    // 分配 a new table (don't 使用 realloc to 避免 数据 corruption 期间 rehashing)
    void** newTable = (void**)malloc(newSize * sizeof(void*));
    if (newTable == nullptr) {
      WARN("Intrusive address map resize failed: malloc(%d entries) returned null", newSize);
      return ncclSystemError;
    }

    // 初始化 所有 new buckets to nullptr
    for (int i = 0; i < newSize; i++) {
      newTable[i] = nullptr;
    }

    // Rehash 所有 existing entries from old table to new table
    for (int i = 0; i < oldSize; i++) {
      void* obj = map->table[i];

      while (obj) {
        void* next = readNextPtr(obj, nextFieldOffset);
        uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
        uint64_t b = ncclHashPointer(newHbits, (void*)objKey);
        writeNextPtr(obj, nextFieldOffset, newTable[b]);
        newTable[b] = obj;
        obj = next;
      }
    }

    // 释放 old table 并且 update to new table
    free(map->table);
    map->table = newTable;
    map->hbits = newHbits;
  }

  // 将对象插入合适的桶中
  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* currentNext = readNextPtr(object, nextFieldOffset);

  // 检查 若 下一个 指针 is 已经 non-NULL (object might 已经 be 入 a 列表)
  if (currentNext != nullptr) {
    INFO(NCCL_INIT,
         "Intrusive map: inserting object %p with non-NULL next pointer %p (key=0x%lx). "
         "Object may already be in another list or this is intentional reuse.",
         object, currentNext, (unsigned long)key);
  }

  writeNextPtr(object, nextFieldOffset, map->table[b]);
  map->table[b] = object;
  map->count += 1;

  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                             int nextFieldOffset, uintptr_t key, void** object) {
  // Runtime 校验
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Output object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  *object = nullptr;

  // 空的 映射 is 不 an 错误 - 仅 means key 不 已找到
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      *object = obj;
      return ncclSuccess;
    }
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key 不 已找到 is 不 an 错误 - *object is 已经 nullptr
  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key) {
  // Runtime 校验
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Removing from 空的 映射 is 不 an 错误 - it's idempotent
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* prev = nullptr;
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      void* next = readNextPtr(obj, nextFieldOffset);

      // Update 上一个 指针 to skip 当前 object
      if (prev == nullptr) {
        // Removing from 头 of bucket
        map->table[b] = next;
      } else {
        // Removing from 中间/末尾 of 列表
        writeNextPtr(prev, nextFieldOffset, next);
      }

      map->count -= 1;

      // 若 此 was 最后一个 entry, clean up the table (相同 pattern as non-intrusive 映射)
      if (map->count == 0) {
        free(map->table);
        map->hbits = 0;
        map->count = 0;
        map->table = nullptr;
      }

      return ncclSuccess;
    }
    prev = obj;
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key 不 已找到 is 不 an 错误 - remove is idempotent
  return ncclSuccess;
}
