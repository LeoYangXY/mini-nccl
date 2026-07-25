/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * trees.cc —— 生成 AllReduce 的“树形(tree)”通信拓扑
 * ----------------------------------------------------------------------------
 * 在 mini-nccl 链路中的位置（与 rings.cc 平行，二选一作为算法）：
 *   graph 阶段在 ring 之外，还会生成 tree 拓扑。tree 算法把 rank 组织成
 *   二叉树/双二叉树：上行阶段做 reduce（把子节点的数据规约到父节点），
 *   下行阶段做 broadcast（把父节点的结果广播到子节点）。
 *
 * 本文件提供两个生成器：
 *   - ncclGetBtree : 生成单棵二叉树（上行 up / 下行 down0,down1）。
 *   - ncclGetDtree : 生成“双二叉树”，由一棵 btree 加上一棵镜像/平移树构成，
 *                    用于提高容错与带宽利用率（两棵树交替使用）。
 * 下面的 ASCII 图说明了 rank 在树中的位置与父子关系（原注释翻译如下）。
 * ============================================================================
 */

#include "nccl.h"

#define RANK_TO_INDEX(r) (rank > root ? rank - 1 : rank)

/* Btree which alternates leaves and nodes.
 * Assumes root is 0, which conveniently builds a tree on powers of two,
 * (because we have pow2-1 ranks) which lets us manipulate bits.
 * Find first non-zero bit, then :
 * Find the parent :
 *   xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
 *   xx11[0] -> xx10[0] (3,7,11 below)
 * Find the children :
 *   xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
 *   xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)
 *
 * Illustration :
 * 0---------------8
 *          ______/ \______
 *         4               12
 *       /   \            /  \
 *     2       6       10     \
 *    / \     / \     /  \     \
 *   1   3   5   7   9   11    13
 */
// 生成单棵二叉树(btree)。基于二进制位操作快速确定父子关系：找到 rank 最低位的
// 置位 bit，父节点 up = (rank ^ bit) | (bit<<1)（越界则回退），子节点 down0/down1
// 为 rank ± lowbit。每个节点只有一个 up（父）和最多两个 down（子）。
//   u     : 输出父节点 rank（-1 表示是根）
//   d0/d1 : 输出两个子节点 rank（-1 表示无该子节点）
//   parentChildType : 本节点相对父节点是第 0 还是第 1 个子节点
ncclResult_t ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
  int up, down0, down1;
  int bit;
  for (bit = 1; bit < nranks; bit <<= 1) {
    if (bit & rank) break;
  }

  if (rank == 0) {
    *u = -1;
    *d0 = -1;
    // Child rank is > 0 so it has to be our child 1, not 0.
    *d1 = nranks > 1 ? bit >> 1 : -1;
    return ncclSuccess;
  }

  up = (rank ^ bit) | (bit << 1);
  // if smaller than the parent, we are his first child, otherwise we're his second
  if (up >= nranks) up = (rank ^ bit);
  *parentChildType = (rank < up) ? 0 : 1;
  *u = up;

  int lowbit = bit >> 1;
  // down0 is always within bounds
  down0 = lowbit == 0 ? -1 : rank - lowbit;

  down1 = lowbit == 0 ? -1 : rank + lowbit;
  // Make sure down1 is within bounds
  while (down1 >= nranks) {
    down1 = lowbit == 0 ? -1 : rank + lowbit;
    lowbit >>= 1;
  }
  *d0 = down0;
  *d1 = down1;

  return ncclSuccess;
}

/* Build a double binary tree. Take the previous tree for the first tree.
 * For the second tree, we use a mirror tree (if nranks is even)
 *
 * 0---------------8                   3----------------11
 *          ______/ \                 / \______
 *         4         \               /         7
 *       /   \        \             /        /   \
 *     2       6       10         1        5      9
 *    / \     / \     /  \       / \      / \    / \
 *   1   3   5   7   9   11     0   2    4   6  8   10
 *
 * or shift it by one rank (if nranks is odd).
 *
 * 0---------------8            1---------------9
 *          ______/ \______              ______/ \______
 *         4               12           5                0
 *       /   \            /           /   \            /
 *     2       6       10           3       7       11
 *    / \     / \     /  \         / \     / \     /  \
 *   1   3   5   7   9   11       2   4   6   8  10   12
 */
// 生成“双二叉树(double binary tree)”：第一棵树直接用 btree；第二棵树在 nranks 为偶数时
// 取镜像树(rank 取反)，为奇数时整体平移 1 个 rank。双二叉树让 AllReduce 可在两棵树间
// 交替，提升带宽与容错。
//   第一组 s0/d0_0/d0_1/parentChildType0 对应第一棵树，s1/d1_0/d1_1/parentChildType1 对应第二棵。
ncclResult_t ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1,
                          int* d1_0, int* d1_1, int* parentChildType1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
  // Second tree ... mirror or shift
  if (nranks % 2 == 1) {
    // shift
    int shiftrank = (rank - 1 + nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : (u + 1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0 + 1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1 + 1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks - 1 - rank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : nranks - 1 - u;
    *d1_0 = d0 == -1 ? -1 : nranks - 1 - d0;
    *d1_1 = d1 == -1 ? -1 : nranks - 1 - d1;
  }
  return ncclSuccess;
}
