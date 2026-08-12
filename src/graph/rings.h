/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/graph/rings.h — 环形(Ring)拓扑构建接口声明
 * ----------------------------------------------------------------------------
 * 对应实现见 src/graph/rings.cc。
 *
 * 在 AllReduce 全链路中的位置：
 *   graph/search.cc 为每个通道搜索出“每个 rank 的前驱/后继”(prev/next 数组)，
 *   本模块负责把这种“链接关系”展开成一条线性的 rank 序列，供后续
 *   graph/connect.cc 建链以及 device 端 kernel 按序收发数据使用。
 */

/*
 * 把 prev/next 邻接关系展开为每个通道的线性环序列。
 *
 * 参数：
 *   nrings : 环(通道)的数量，即需要构建多少条独立的 ring
 *   rings  : [输出] 长度 nrings*nranks 的数组；
 *            rings[r*nranks + i] 表示第 r 条环上第 i 个位置放的是哪个 rank
 *   rank   : 本进程的 rank 编号，作为遍历环的起点，同时用于日志与校验
 *   nranks : 通信域内的总 rank 数
 *   prev   : [输入] 长度 nrings*nranks；prev[r*nranks + x] 表示第 r 条环上 rank x 的前驱
 *   next   : [输入] 长度 nrings*nranks；next[r*nranks + x] 表示第 r 条环上 rank x 的后继
 *
 * 返回：ncclSuccess 表示所有环都闭合且覆盖了全部 rank；
 *       否则返回 ncclInternalError(环断开或有 rank 缺失)。
 */
ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next);
