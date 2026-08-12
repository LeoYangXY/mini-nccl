/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/mlx5/mlx5dvsymbols.h — MLX5 符号动态加载 [第三方/Mellanox]
 * ----------------------------------------------------------------------------
 * 在 NCCL_BUILD_MLX5DV 下声明从 libmlx5dv 动态加载的符号，
 * 通过 dlsym 获取 MLX5 direct verbs 函数指针。
 */

#ifndef NCCL_MLX5DV_SYMBOLS_H_
#define NCCL_MLX5DV_SYMBOLS_H_

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5/mlx5dvcore.h"
#endif

#include "nccl.h"

/* MLX5 Direct Verbs Function Pointers*/
struct ncclMlx5dvSymbols {
  bool (*mlx5dv_internal_is_supported)(struct ibv_device* device);
  int (*mlx5dv_internal_get_data_direct_sysfs_path)(struct ibv_context* context, char* buf, size_t buf_len);
  /* DMA-BUF support */
  struct ibv_mr* (*mlx5dv_internal_reg_dmabuf_mr)(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                                  int fd, int access, int mlx5_access);
  int (*mlx5dv_internal_query_device)(struct ibv_context* ctx_in, struct mlx5dv_context* attrs_out);
  struct ibv_qp* (*mlx5dv_internal_create_qp)(struct ibv_context* context, struct ibv_qp_init_attr_ex* qp_attr,
                                              struct mlx5dv_qp_init_attr* mlx5_qp_attr);
};

/* Constructs MLX5 direct verbs symbols per rdma-core linking or dynamic loading mode */
ncclResult_t buildMlx5dvSymbols(struct ncclMlx5dvSymbols* mlx5dvSymbols);

#endif  // NCCL_MLX5DV_SYMBOLS_H_
