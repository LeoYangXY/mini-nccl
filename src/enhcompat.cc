/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/enhcompat.cc — 旧版 CUDA 运行时静态库兼容层(weak symbol 桩)
 * ----------------------------------------------------------------------------
 * 作用：让 libnccl_static.a 能够与“较老版本的 libcudart_static.a”一起链接成功。
 *
 * 背景：NCCL 使用了一些较新的 CUDA Runtime API(主要与 CUDA Graph 捕获相关)。
 *   如果用户的 CUDA 静态运行时版本过旧，里面根本不存在这些符号，静态链接时就会
 *   报 “undefined reference”，导致整个程序链接失败。
 *
 * 解决办法：在这里为这些 API 定义一份“弱符号(weak symbol)”桩实现。
 *   - 弱符号的特点：如果链接时能在真正的 libcudart 中找到强符号定义，
 *     链接器会优先选用真实实现，这里的桩会被自动丢弃；
 *   - 只有当真实实现确实不存在时，才会回退到这里的桩函数。
 *   - 桩函数一律返回 cudaErrorStubLibrary(34)，表示“这是桩库，功能不可用”，
 *     NCCL 上层看到该错误码后会退化到不使用 CUDA Graph 的路径，而不是崩溃。
 *
 * 注意：这里刻意使用 (...) 变参声明来规避真实函数原型，因为在旧版本头文件中
 *   这些函数的类型可能根本没有定义；同时用 visibility("hidden") 把符号限制在
 *   库内部，避免污染最终可执行文件的动态符号表。
 */

/* 定义弱符号，使 libnccl_static.a 可以和更旧的 libcudart_static.a 协同工作 */

enum cudaError_t {
  cudaErrorStubLibrary = 34   // CUDA 官方错误码：表示调用到的是桩(stub)库，功能不可用
};

extern "C" {

// 查询流的 CUDA Graph 捕获状态(v2 版本)：NCCL 用它判断当前是否处于图捕获中
cudaError_t cudaStreamGetCaptureInfo_v2(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaStreamGetCaptureInfo_v2(...) {
  return cudaErrorStubLibrary;
}

// 创建 CUDA 用户对象：NCCL 用它把自身资源生命周期挂到 CUDA Graph 上
cudaError_t cudaUserObjectCreate(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaUserObjectCreate(...) {
  return cudaErrorStubLibrary;
}

// 让 CUDA Graph 持有某个用户对象的引用，保证图存活期间资源不被释放
cudaError_t cudaGraphRetainUserObject(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaGraphRetainUserObject(...) {
  return cudaErrorStubLibrary;
}

// 在图捕获过程中更新流的依赖关系(用于把 NCCL 内部节点正确串入捕获图)
cudaError_t cudaStreamUpdateCaptureDependencies(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaStreamUpdateCaptureDependencies(...) {
  return cudaErrorStubLibrary;
}

// 按名字获取 CUDA Driver API 的函数入口地址(动态解析驱动符号)
cudaError_t cudaGetDriverEntryPoint(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaGetDriverEntryPoint(...) {
  return cudaErrorStubLibrary;
}
}
