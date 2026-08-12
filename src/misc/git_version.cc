/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/misc/git_version.cc — git 版本字符串导出
 * ----------------------------------------------------------------------------
 * 由 generate_git_version.py 生成的 nccl_git_version.h 提供分支/提交哈希，本文件
 * 拼出 “NCCL git version ...” 字符串并通过 ncclGetGitVersion() 暴露，便于排查
 * 版本问题。
 */

#include "nccl_git_version.h"

// 预处理该字符串，使得对库文件执行 strings 命令时能够快速看到版本号。
#define NCCL_GIT_VERSION_STRING "NCCL git version " NCCL_GIT_BRANCH " " NCCL_GIT_COMMIT_HASH
const char* ncclGetGitVersion(void) {
  return NCCL_GIT_VERSION_STRING;
}
