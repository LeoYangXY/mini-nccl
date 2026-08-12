/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 Google LLC.  All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/device/network/unpack/unpack.h — 网络 unpack 原语实现
 * ----------------------------------------------------------------------------
 * 实现 GPU 端的“解包”原语：把网络传输(Net)收到的打包数据，在 device kernel 内
 * 直接还原为目标张量布局（并可在解包时完成 reduce）。是网络接收路径的最后一步。
 * 源自 Google/NVIDIA 协作。
 */

#ifndef NET_DEVICE_UNPACK_H
#define NET_DEVICE_UNPACK_H

#include "unpack_defs.h"

#include "op128.h"
#include "bitops.h"
#include "device.h"
#include "common.h"

// #定义 ALIGNED_LOAD

inline __device__ void load64gpu(const uint64_t* ptr, uint64_t& v) {
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];" : "=l"(v) : "l"(ptr) : "memory");
#else
  asm volatile("ld.volatile.global.u64 {%0}, [%1];" : "=l"(v) : "l"(ptr) : "memory");
#endif
}

#define PAGE_META_SIZE 16
#define META_LOAD_SIZE 16
#define DATA_LOAD_SIZE 16

// 映射 内部 association of 句柄 with 组 并且 对等端 索引 (被调用 一旦 at 初始化 time)
inline __device__ void ncclNetDeviceUnpackSetup(void* ohandle, const int group, const int index) {
  struct unpackNetDeviceHandle* handle = (struct unpackNetDeviceHandle*)ohandle;
  // coverity[index_parm:假]
  ncclShmem.groups[group].devicePlugin.unpack.g_meta[index] = handle->meta;
  ncclShmem.devicePlugin.unpack.bounce_buf = handle->bounce_buf;
  // coverity[index_parm:假]
  ncclShmem.groups[group].devicePlugin.unpack.head[index] = handle->head;
}

inline __device__ void ncclNetDeviceIncrementHead(const int group, const int index) {
  // coverity[index_parm:假]
  ncclShmem.groups[group].devicePlugin.unpack.head[index]++;
}

inline __device__ void ncclNetDeviceSaveHead(void* ohandle, const int group, const int index) {
  struct unpackNetDeviceHandle* handle = (struct unpackNetDeviceHandle*)ohandle;
  // coverity[index_parm:假]
  handle->head = ncclShmem.groups[group].devicePlugin.unpack.head[index];
}

template <uint8_t sz>
inline __device__ void bulkLoad(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<sz>* reg,
                                const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off) {
  bulkLoad<1>(t, len, cpy_src, cpy_dst, reg, w, g_meta, s_meta, src_off, dst_off);
}

template <>
inline __device__ void bulkLoad<1>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<1> reg[16],
                                   const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off,
                                   uint64_t dst_off) {
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 16; i++) {
      reg[i] = ld_volatile_global<1>((uintptr_t)((uint8_t*)(cpy_src + data_s) + i));
    }
#endif

    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 16; i++) {
      st_global<1>((uintptr_t)((uint8_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<2>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<2> reg[8],
                                   const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off,
                                   uint64_t dst_off) {
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 8; i++) {
      reg[i] = ld_volatile_global<2>((uintptr_t)((uint16_t*)(cpy_src + data_s) + i));
    }
#endif

    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 8; i++) {
      st_global<2>((uintptr_t)((uint16_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<4>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<4> reg[4],
                                   const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off,
                                   uint64_t dst_off) {
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 4; i++) {
      reg[i] = ld_volatile_global<4>((uintptr_t)((uint32_t*)(cpy_src + data_s) + i));
    }
#endif

    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 4; i++) {
      st_global<4>((uintptr_t)((uint32_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<8>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<8> reg[2],
                                   const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off,
                                   uint64_t dst_off) {
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 2; i++) {
      reg[i] = ld_volatile_global<8>((uintptr_t)((uint64_t*)(cpy_src + data_s) + i));
    }
#endif

    NVCC_PRAGMA_UNROLL_AUTO
    for (int i = 0; i < 2; i++) {
      st_global<8>((uintptr_t)((uint64_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<16>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<16> reg[1],
                                    const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off,
                                    uint64_t dst_off) {
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
    reg[0] = ld_volatile_global<16>((uintptr_t)(cpy_src + data_s));
    st_global<16>((uintptr_t)(cpy_dst + data_s), reg[0]);
  }
}

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
inline __device__ int ppw(const int nbytes, int nw) {
  int v = DIVUP(nbytes, SLICE_PAGE_SIZE);
  v = DIVUP(v, nw);
  while (v > WARP_SHM_PAGE_CNT) {
    v = DIVUP(v, 2);
  }
  return v;
}

// 该函数 被称为 by 所有 线程
// 打包 数据 从 内部 iovec 到 supplied flat 缓冲区 使用 所有 the
// 线程
template <int Recv>
inline __device__ void ncclNetDeviceUnpack(const int tid, const int tidInBlock, const int nworkers, const int group,
                                           int mask, int Src, int workSize);

template <>
inline __device__ void ncclNetDeviceUnpack</*Recv=*/0>(const int tid, const int tidInBlock, const int nworkers,
                                                       const int group, int mask, int Src, int workSize) {
  // 发送 解包 空的
}

inline __device__ void ncclNetDeviceUnpackInner(const int tid, const int tidInBlock, const int nworkers,
                                                const int group, const int index, void* src, const int nbytes,
                                                const uint64_t step);

template <>
inline __device__ void ncclNetDeviceUnpack</*Recv=*/1>(const int tid, const int tidInBlock, const int nworkers,
                                                       const int group, int mask, int Src, int workSize) {
  while (mask != 0) {
    int ix = __ffs(mask) - 1; // Get the first set bit of the mask (this should correlate to a peer index)
    mask &= mask - 1; // Drop the first set bit of the mask

    // 打包 数据 从 内部 iovec 到 supplied flat srcs 缓冲区 使用 所有 the 线程
    // + 源 必要的 在 ... 中 情形 of accessing 用户 缓冲区 directly
    ncclNetDeviceUnpackInner(tid, tidInBlock, nworkers,
                             group /* in case they need to use split warps shared memory partitioning*/, ix,
                             ncclShmem.groups[group].srcs[ix + Src], workSize,
                             ncclShmem.groups[group].devicePlugin.unpack.head[ix]);
  }
}

inline __device__ void ncclNetDeviceUnpackInner(const int tid, const int tidInBlock, const int nworkers,
                                                const int group, const int index, void* src, const int nbytes,
                                                const uint64_t step) {
  // from 源/集合通信/设备/common_kernel.h
  const int w = tid / WARP_SIZE;        // Warp number
  const int nw = nworkers / WARP_SIZE;  // Number of warps
  const int t = tid % WARP_SIZE;        // Thread (inside the warp)

  BytePack<16> reg;
  loadMeta meta;

  uint64_t head;
  struct netUnpackMeta* g_meta_struct;
  void* bounce_buf;

  loadMeta* g_meta;
  loadMeta* s_meta;
  uint64_t meta_cnt;

  // hack 头 使用 每个-线程束
  head = step;
  g_meta_struct = ncclShmem.groups[group].devicePlugin.unpack.g_meta[index];
  bounce_buf = ncclShmem.devicePlugin.unpack.bounce_buf;

  __syncwarp();

  head %= NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH;

  g_meta = g_meta_struct->mem[head];

  // Currently, 甚至/奇数 组 perform 发送/接收 separately. We don't really 需要 space for 发送 side.
  // 总计 大小 is N 页 每个 线程束 * 16 B 每个 页 * 20 线程束 最大值 = 320 * N 字节, N == WARP_SHM_PAGE_CNT
  static_assert(ncclShmemScratchWarpSize() >= WARP_SHM_SIZE, "Each warp must have enough scratch space");
  // (loadMeta*) (ncclShmem.devicePlugin.解包.meta + shm_off);
  s_meta = (loadMeta*)ncclScratchForWarp(tidInBlock / WARP_SIZE);

  load64gpu(g_meta_struct->cnt + head, meta_cnt);

  int PPW = ppw(nbytes, nw);

  // Coverity reports a potential overflow 但 入 reality PPW is 微小 所以 there's 无 需要 存储 it 入 an uint64_t.
  // coverity[overflow_before_widen]
  for (uint64_t meta_s = w * PPW; meta_s < meta_cnt; meta_s += nw * PPW) {
    uint64_t iter_meta_cnt = meta_cnt - meta_s;
    iter_meta_cnt = iter_meta_cnt < PPW ? iter_meta_cnt : PPW;

    // 待办: 此 加载 大小 需要 work 否则 已对齐, 但 自 the two are 两者 16...
    if (t < PPW * PAGE_META_SIZE / META_LOAD_SIZE && t < iter_meta_cnt) {
      // 避免 最后 iter 加载 garbage 数据
      load128((const uint64_t*)(g_meta + (meta_s + t)), reg.u64[0], reg.u64[1]);

      storeShmem128(shmemCvtPtr((uint64_t*)(s_meta + (w * PPW + t))), reg.u64[0], reg.u64[1]);
    }

    __syncwarp();

    for (int x = 0; x < iter_meta_cnt; x++) {
      int meta_idx = x + w * PPW;

      // 加载 页 offs
      loadShmem128(shmemCvtPtr((uint64_t*)(s_meta + meta_idx)), meta.r64[0], meta.r64[1]);

      if (meta.len >= DATA_LOAD_SIZE) {
        // 快速 路径, 但 需要 adapt to 对齐 问题

        // bulk 拷贝 数据
        uint8_t align_off = (meta.src_off | meta.dst_off) % DATA_LOAD_SIZE;
        align_off = align_off & -align_off;  // keep the lowest bit
        if (align_off == 0) {
          // 0x16
          bulkLoad<16>(t, meta.len, (char*)bounce_buf + meta.src_off, (char*)src + meta.dst_off, &reg, w, g_meta,
                       s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x8) {
          bulkLoad<8>(t, meta.len, (char*)bounce_buf + meta.src_off, (char*)src + meta.dst_off, (BytePack<8>*)&reg, w,
                      g_meta, s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x4) {
          bulkLoad<4>(t, meta.len, (char*)bounce_buf + meta.src_off, (char*)src + meta.dst_off, (BytePack<4>*)&reg, w,
                      g_meta, s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x2) {
          bulkLoad<2>(t, meta.len, (char*)bounce_buf + meta.src_off, (char*)src + meta.dst_off, (BytePack<2>*)&reg, w,
                      g_meta, s_meta, meta.src_off, meta.dst_off);
        } else {
          // 若 (align_off & 0x1)
          bulkLoad<1>(t, meta.len, (char*)bounce_buf + meta.src_off, (char*)src + meta.dst_off, (BytePack<1>*)&reg, w,
                      g_meta, s_meta, meta.src_off, meta.dst_off);
        }
      }

      // 必须为 less than 16 字节
      if (t < meta.len % DATA_LOAD_SIZE) {
        volatile char* cpy_src = (char*)bounce_buf + meta.src_off + (meta.len / DATA_LOAD_SIZE) * DATA_LOAD_SIZE + t;
        volatile char* cpy_dst = (char*)src + meta.dst_off + (meta.len / DATA_LOAD_SIZE) * DATA_LOAD_SIZE + t;
        *cpy_dst = *cpy_src;
      }
    }

    __syncwarp();
  }
}

#endif  // NET_DEVICE_UNPACK_DEFS_H_
