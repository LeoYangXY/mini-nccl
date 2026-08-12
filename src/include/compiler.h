/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/compiler.h — 编译器兼容宏（可移植内建指令）
 * ----------------------------------------------------------------------------
 * 统一定义跨编译器(gcc/clang/msvc)的宏：强制内联、对齐、likely/unlikely、属性
 * 包装等，使 NCCL 源码在不同工具链下都能编译。
 */

#ifndef NCCL_PORTABLE_INTRINSICS_H
#define NCCL_PORTABLE_INTRINSICS_H

#ifdef __cplusplus
extern "C++" {
#endif

#include <atomic>

#ifdef __cplusplus
}
#endif

// 编译器 detection 宏
#if defined(__GNUC__) || defined(__clang__)
#define NCCL_COMPILER_GCC 1
#include "compiler/gcc.h"
#elif defined(_MSC_VER)
#define NCCL_COMPILER_MSVC 1
#include "compiler/msvc.h"
#else
#error "Unsupported compiler"
#endif

#endif // NCCL_PORTABLE_INTRINSICS_H
