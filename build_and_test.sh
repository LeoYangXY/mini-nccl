#!/usr/bin/env bash
#
# mini-nccl 一键脚本：编译 libnccl.so + 编译 all_reduce_perf + 跑 all-reduce 验证
#
# 用法:
#   ./build_and_test.sh                    # 默认: 2 卡, 8B ~ 128MB
#   ./build_and_test.sh -g 4               # 4 卡
#   ./build_and_test.sh -g 2 -b 1M -e 1G   # 自定义数据量区间
#   ./build_and_test.sh -c                 # 先清理 build/ 再全量重编
#   ./build_and_test.sh -a sm_90           # 手动指定架构(默认按 GPU 自动探测)
#   ./build_and_test.sh -L                 # 只编译, 不跑测试
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ---------- 默认参数 ----------
GPUS=2                       # 参与测试的 GPU 数
BEGIN=8                      # 起始数据量 (-b)
END=128M                     # 结束数据量 (-e)
FACTOR=2                     # 步长倍率 (-f)
ARCH=""                      # 计算架构, 空 = 自动探测
JOBS="$(nproc)"
CLEAN=0
BUILD_ONLY=0
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

usage() { sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while getopts ":g:b:e:f:a:j:chL" opt; do
  case "$opt" in
    g) GPUS="$OPTARG" ;;
    b) BEGIN="$OPTARG" ;;
    e) END="$OPTARG" ;;
    f) FACTOR="$OPTARG" ;;
    a) ARCH="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    c) CLEAN=1 ;;
    L) BUILD_ONLY=1 ;;
    h) usage ;;
    *) echo "未知选项: -$OPTARG" >&2; usage ;;
  esac
done

# ---------- 环境检查 ----------
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "[!] 找不到 nvcc: $CUDA_HOME/bin/nvcc (用 CUDA_HOME=... 指定, 需 >= 12.0)" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "[!] 找不到 nvidia-smi" >&2; exit 1; }

# PATH 里的 nvcc 可能是旧版本(如 11.5), 一律用 CUDA_HOME 下的
export CUDA_HOME
echo "[env] CUDA_HOME=$CUDA_HOME  $($CUDA_HOME/bin/nvcc --version | tail -1)"

# ---------- 自动探测计算架构 (H20/H100 -> sm_90) ----------
if [ -z "$ARCH" ]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  [ -n "$CC" ] || CC="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -oE '[0-9]+' | head -1)0"
  case "$CC" in
    90|100|110|120) ARCH="sm_$CC" ;;
    *) echo "[!] 无法识别 GPU 架构(compute_cap=$CC), 请用 -a sm_90 手动指定" >&2; exit 1 ;;
  esac
fi
NVCC_GENCODE="-gencode=arch=compute_${ARCH#sm_},code=$ARCH"
echo "[env] NVCC_GENCODE=$NVCC_GENCODE"

# ---------- 修权限: 两个 python 生成脚本需要有可执行位, 否则 make 失败 ----------
chmod +x src/misc/generate_git_version.py src/device/generate.py

# ---------- 清理 ----------
if [ "$CLEAN" = 1 ]; then
  echo "[clean] rm -rf build tests/build"
  rm -rf build tests/build
fi
# device 内核源码由 generate.py 生成; 若目录为空说明上次生成失败, 必须清掉重来
if [ -d build/obj/device/gensrc ] && [ -z "$(ls -A build/obj/device/gensrc 2>/dev/null)" ]; then
  echo "[clean] 检测到空的 build/obj/device/gensrc, 清理后重新生成"
  rm -rf build/obj/device
fi

# ---------- 编译库 ----------
echo "[1/3] 编译 libnccl.so ..."
make -j"$JOBS" lib NVCC_GENCODE="$NVCC_GENCODE"
ls -l build/lib/libnccl.so.2

# ---------- 编译测试 ----------
echo "[2/3] 编译 tests/all_reduce_perf ..."
make -C tests -j"$JOBS" NCCL_HOME="$REPO_DIR/build" CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE"
[ "$BUILD_ONLY" = 1 ] && { echo "[done] 仅编译, 跳过测试"; exit 0; }

# ---------- 运行 ----------
echo "[3/3] 运行 all_reduce_perf (${GPUS} 卡, ${BEGIN} ~ ${END}) ..."
LD_LIBRARY_PATH="$REPO_DIR/build/lib" \
  ./tests/build/all_reduce_perf -b "$BEGIN" -e "$END" -f "$FACTOR" -g "$GPUS"

echo
echo "[done] 判定标准: 'Out of bounds values : 0 OK' 即结果正确"
