/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/trees.h — tree(树)算法结构与枚举声明
 * ----------------------------------------------------------------------------
 * 定义 TREE 算法所需的树结构：tree 类型(二叉树/多叉树)、节点上下行端口、树相关的
 * 常量。配合 graph/connect 把 graph 的 tree 排列翻译成实际 channel 连接。
 */

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

#include "nccl.h"

ncclResult_t ncclGetBtree(int nranks, int rank, int* u0, int* d1, int* d0, int* parentChildType);
ncclResult_t ncclGetDtree(int nranks, int rank, int* u0, int* d0_0, int* d0_1, int* parentChildType0, int* u1,
                          int* d1_0, int* d1_1, int* parentChildType1);

#endif
