/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/argcheck.h — 参数校验(argument check)接口
 * ----------------------------------------------------------------------------
 * 声明 ncclInvalidArgument 等参数合法性检查：对用户传入的 comm/handle/数据类型/
 * 指针等做前置校验，及早返回清晰的错误码。
 */

#ifndef NCCL_ARGCHECK_H_
#define NCCL_ARGCHECK_H_

#include "core.h"
#include "info.h"

struct ncclArgsInfo {
  struct ncclInfo info;
  struct ncclArgsInfo* next;
};

ncclResult_t PtrCheck(const void* ptr, const char* opname, const char* ptrname);
ncclResult_t CommCheck(struct ncclComm* ptr, const char* opname, const char* ptrname);
ncclResult_t ArgsCheck(struct ncclInfo* info);
ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname);
ncclResult_t ncclArgsGlobalCheck(struct ncclArgsInfo* info);

#endif
