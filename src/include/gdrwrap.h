/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/gdrwrap.h — GPU Direct RDMA 包装
 * ----------------------------------------------------------------------------
 * 通过 dlopen 动态加载 cuGPUDirectRDMA / libcuda 相关符号，封装 GDR(GPU Direct
 * 访问)能力，使 NCCL 在无 GDR 环境下也能编译（运行时按需解析）。
 */

#ifndef NCCL_GDRWRAP_H_
#define NCCL_GDRWRAP_H_

#include "nccl.h"
#include "alloc.h"
#include <stdint.h> // for standard [u]intX_t types
#include <stdio.h>
#include <stdlib.h>
#include <mutex>

// 这些 可以 已使用 若 GDR 库 isn't 线程 safe
std::mutex& getGdrMutex();
#define GDRLOCKCALL(cmd, ret) do {                      \
    std::lock_guard<std::mutex> lock(getGdrMutex());   \
    ret = cmd;                                          \
} while(false)

#define GDRCHECK(cmd) do {                              \
    int e;                                              \
    /* GDRLOCKCALL(cmd, e); */                          \
    e = cmd;                                            \
    if( e != 0 ) {                                      \
      WARN("GDRCOPY failure %d", e);                    \
      return ncclSystemError;                           \
    }                                                   \
} while(false)

// 此 需要 as the GDR 内存 is mapped WC
#if !defined(__NVCC__)
#if defined(__PPC__)
static inline void wc_store_fence(void) { asm volatile("sync" : : : "memory"); }
#elif defined(__x86_64__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wc_store_fence(void) { std::atomic_thread_fence(std::memory_order_release); }
#else
#include <stdatomic.h>
static inline void wc_store_fence(void) { atomic_thread_fence(memory_order_release); }
#endif
#endif
#endif

//#定义 GDR_DIRECT 1
#ifdef GDR_DIRECT
// 调用 the GDR API 库 代码 directly rather than via
// dlopen() 包装函数
#include <gdrapi.h>

static ncclResult_t wrap_gdr_symbols(void) { return ncclSuccess; }
static gdr_t wrap_gdr_open(void) { gdr_t g = gdr_open(); return g; }
static ncclResult_t wrap_gdr_close(gdr_t g) { GDRCHECK(gdr_close(g)); return ncclSuccess; }
static ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) {
  GDRCHECK(gdr_pin_buffer(g, addr, size, p2p_token, va_space, handle));
  return ncclSuccess;
}

static bool ncclGdrPinV2Available(void) {
#if defined(GDR_API_MAJOR_VERSION) && defined(GDR_API_MINOR_VERSION)
  return (GDR_API_MAJOR_VERSION > 2) || (GDR_API_MAJOR_VERSION == 2 && GDR_API_MINOR_VERSION >= 5);
#else
  return false;
#endif
}
static ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t* handle) {
  if (!ncclGdrPinV2Available()) {
    WARN("gdr_pin_buffer_v2 not available; GDRCopy >= 2.5 required");
    return ncclInternalError;
  }
  GDRCHECK(gdr_pin_buffer_v2(g, addr, size, flags, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  GDRCHECK(gdr_unpin_buffer(g, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info) {
  GDRCHECK(gdr_get_info(g, handle, info));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size) {
  GDRCHECK(gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size) {
  GDRCHECK(gdr_unmap(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static void wrap_gdr_runtime_get_version(int *major, int *minor) {
  gdr_runtime_get_version(major, minor);
  return ncclSuccess;
}
static void wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor) {
  gdr_driver_get_version(g, major, minor);
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size) {
  GDRCHECK(gdr_copy_to_mapping(handle, map_d_ptr, h_ptr, size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size) {
  GDRCHECK(gdr_copy_from_mapping(handle, h_ptr, map_d_ptr, size));
  return ncclSuccess;
}

#else
// Dynamically 句柄 dependency the GDR API 库

/* Extracted from gdrapi.h (v2.1 Nov 2020) */
/* Exception: gdr_pin_flags / GDR_PIN_FLAG_FORCE_PCIE extracted from gdrapi.h (v2.5) */

typedef enum gdr_pin_flags {
  GDR_PIN_FLAG_DEFAULT     = 0,
  GDR_PIN_FLAG_FORCE_PCIE  = 1
} gdr_pin_flags_t;

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
};
typedef struct gdr_info gdr_info_t;

/* End of gdrapi.h */

ncclResult_t wrap_gdr_symbols(void);

gdr_t wrap_gdr_open(void);
ncclResult_t wrap_gdr_close(gdr_t g);
ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
bool ncclGdrPinV2Available(void);
ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t *handle);
ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);
ncclResult_t wrap_gdr_runtime_get_version(int *major, int *minor);
ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor);
ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);

#endif // GDR_DIRECT

// 全局的 GDR driver 句柄; 设置 一旦 期间 NCCL 初始化.
extern gdr_t ncclGdrCopy;

#include "alloc.h"

typedef struct gdr_mem_desc {
  void *gdrDevMem;
  void *gdrMap;
  size_t gdrOffset;
  size_t gdrMapSize;
  gdr_mh_t gdrMh;
} gdr_mem_desc_t;

static gdr_t ncclGdrInit() {
  int libMajor, libMinor, drvMajor, drvMinor;
  gdr_t handle = NULL;
  // Dynamically 加载 GDRAPI 库 symbols
  if (wrap_gdr_symbols() == ncclSuccess) {
    handle = wrap_gdr_open();

    if (handle != NULL) {
      ncclResult_t res;

      // Query the 版本 of libgdrapi
      NCCLCHECKGOTO(wrap_gdr_runtime_get_version(&libMajor, &libMinor), res, error);

      // Query the 版本 of gdrdrv driver
      NCCLCHECKGOTO(wrap_gdr_driver_get_version(handle, &drvMajor, &drvMinor), res, error);

      // 仅 支持 GDRAPI 2.1 并且 later
      if (libMajor < 2 || (libMajor == 2 && libMinor < 1) || drvMajor < 2 || (drvMajor == 2 && drvMinor < 1)) {
        goto error;
      }
      else
        INFO(NCCL_INIT, "GDRCOPY enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
    }
  }
  return handle;
error:
  if (handle != NULL) (void) wrap_gdr_close(handle);
  return NULL;
}

template <typename T>
static ncclResult_t ncclGdrCudaCalloc(T** ptr, T** devPtr, size_t nelem, void** gdrHandle, struct ncclMemManager* manager, uint32_t pinFlags = 0) {
  gdr_info_t info;
  size_t mapSize;
  gdr_mh_t mh;
  char *devMem;
  void *gdrMap;

  mapSize = ncclSizeOfT<T>()*nelem;

  // GDRCOPY Pinned 缓冲区 has to be a 最小 of a GPU_PAGE_SIZE
  ALIGN_SIZE(mapSize, GPU_PAGE_SIZE);
  // GDRCOPY Pinned 缓冲区 has to be GPU_PAGE_SIZE 已对齐 too
  NCCLCHECK(ncclCudaCalloc(&devMem, mapSize+GPU_PAGE_SIZE-1, manager));
  uint64_t alignedAddr = (((uint64_t) devMem) + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  size_t align = alignedAddr - (uint64_t)devMem;

  if (ncclGdrPinV2Available() || pinFlags == GDR_PIN_FLAG_FORCE_PCIE) {
    // 若 pingFlags 被设为 to FORCE_PCIE, 我们会 错误 出 若 我们可以't honnor it.
    NCCLCHECK(wrap_gdr_pin_buffer_v2(ncclGdrCopy, alignedAddr, mapSize, pinFlags, &mh));
  } else {
    // 追踪(NCCL_INIT, "GDRCOPY: Pin 缓冲区 0x%lx (%p) align %zu 大小 %zu", alignedAddr, devMem, align, mapSize);
    NCCLCHECK(wrap_gdr_pin_buffer(ncclGdrCopy, alignedAddr, mapSize, 0, 0, &mh));
  }

  NCCLCHECK(wrap_gdr_map(ncclGdrCopy, mh, &gdrMap, mapSize));
  //追踪(NCCL_INIT, "GDRCOPY : mapped %p (0x%lx) at %p", devMem, alignedAddr, gdrMap);

  NCCLCHECK(wrap_gdr_get_info(ncclGdrCopy, mh, &info));

  // Will 偏移 ever be non zero ?
  ssize_t off = info.va - alignedAddr;

  gdr_mem_desc_t* md;
  NCCLCHECK(ncclCalloc(&md, 1));
  md->gdrDevMem = devMem;
  md->gdrMap = gdrMap;
  md->gdrMapSize = mapSize;
  md->gdrOffset = off+align;
  md->gdrMh = mh;
  *gdrHandle = md;

  *ptr = (T *)((char *)gdrMap+off);
  if (devPtr) *devPtr = (T *)(devMem+off+align);

  TRACE(NCCL_INIT, "GDRCOPY : allocated devMem %p gdrMap %p offset %lx mh %lx mapSize %zu at %p",
       md->gdrDevMem, md->gdrMap, md->gdrOffset, md->gdrMh.h, md->gdrMapSize, *ptr);

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclGdrCudaCopy(void *gdrHandle, T* dst, T* src, size_t nelem) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_copy_to_mapping(md->gdrMh, dst, src, nelem*ncclSizeOfT<T>()));
  return ncclSuccess;
}

static ncclResult_t ncclGdrCudaRead(void* gdrHandle, void* dst, const void* src, size_t size) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  return wrap_gdr_copy_from_mapping(md->gdrMh, dst, src, size);
}

static ncclResult_t ncclGdrCudaFree(void* gdrHandle, struct ncclMemManager* manager) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_unmap(ncclGdrCopy, md->gdrMh, md->gdrMap, md->gdrMapSize));
  NCCLCHECK(wrap_gdr_unpin_buffer(ncclGdrCopy, md->gdrMh));
  NCCLCHECK(ncclCudaFree(md->gdrDevMem, manager));
  free(md);

  return ncclSuccess;
}

// 辅助: 分配 内存 accessible from CPU (二者之一 GDR 或者 主机 内存)
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, struct ncclMemManager* manager, bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle, manager));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;  // Mark as host allocated by nulling GDR handle
  }
  return ncclSuccess;
}

// 辅助: 释放 内存 已分配 by allocMemCPUAccessible
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle, struct ncclMemManager* manager) {
  if (gdrHandle != NULL) {
    // 若 a GDR 句柄 exists, it was GDR 内存
    NCCLCHECK(ncclGdrCudaFree(gdrHandle, manager));
  } else {
    // 否则, it was 主机 内存 (或者 GDR was off)
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

#endif // End include guard
