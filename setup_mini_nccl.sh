#!/bin/bash
# 一键配置 mini-nccl 运行环境并跑通 AllReduce test
# 机器: 驱动535(最高CUDA12.2); mini-nccl 需 CUDA12.2+ 头文件
# 本脚本随 mini-nccl 仓库一起存放, 默认构建脚本所在目录的源码。
set -e

CUDA_VERSION=12.2.2
CUDA_RUN=cuda_12.2.2_535.104.05_linux.run
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/${CUDA_RUN}"
CUDA_HOME=/usr/local/cuda-12.2

# mini-nccl 源码目录: 默认取本脚本所在目录 (即仓库根)
MINI_NCCL_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE=/root/.mini_nccl_setup
mkdir -p "$CACHE"

echo "[0/6] 配置 zsh git 补全 ..."
if command -v zsh >/dev/null 2>&1; then
  if ! grep -q "compinit" ~/.zshrc 2>/dev/null; then
    cat >> ~/.zshrc <<'ZSHEOF'

# enable zsh completion (git etc.)
autoload -Uz compinit && compinit
zstyle ':completion:*' menu select
ZSHEOF
    echo "  已写入 ~/.zshrc (重开终端或 source ~/.zshrc 生效)"
  else
    echo "  ~/.zshrc 已有 compinit, 跳过"
  fi
else
  echo "  未安装 zsh, 跳过"
fi

echo "[1/6] 检查 CUDA toolkit ..."
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "  已存在: $($CUDA_HOME/bin/nvcc --version | tail -1)"
else
  echo "  未安装, 下载并仅装 toolkit(不动驱动) ..."
  pushd "$CACHE" >/dev/null
  wget -c -q "$CUDA_URL" -O "$CUDA_RUN"
  sh "$CUDA_RUN" --toolkit --silent --override --installpath="$CUDA_HOME"
  popd >/dev/null
fi

echo "[2/6] 检查 mini-nccl 仓库 ..."
if [ -d "$MINI_NCCL_DIR/src" ]; then
  echo "  已存在: $MINI_NCCL_DIR"
else
  echo "  未找到 $MINI_NCCL_DIR/src, 请先将 mini-nccl 抽取到该路径" >&2
  exit 1
fi

echo "[3/6] 编译 mini-nccl 库 (sm_90, 适配 H20) ..."
cd "$MINI_NCCL_DIR"
if [ -f build/lib/libnccl.so ]; then
  echo "  已编译, 跳过"
else
  make -j"$(nproc)" lib CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
fi

echo "[4/6] 编译仓库自带 tests (all_reduce) ..."
cd "$MINI_NCCL_DIR/tests"
make -j"$(nproc)" NCCL_HOME="$MINI_NCCL_DIR/build" CUDA_HOME="$CUDA_HOME" 2>&1 | tail -3

echo "[5/6] 生成 python all_reduce test ..."
cat > /root/allreduce_test.py <<'PYEOF'
import os,time,torch,torch.distributed as dist
import torch.multiprocessing as mp
def worker(rank,world):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"]="127.0.0.1"
    os.environ["MASTER_PORT"]=os.environ.get("MASTER_PORT","29555")
    dist.init_process_group("nccl",rank=rank,world_size=world)
PYEOF
cat >> /root/allreduce_test.py <<'PYEOF'
    t=torch.randn(16777216,device="cuda")
    for _ in range(10): dist.all_reduce(t)
    torch.cuda.synchronize()
    s=time.perf_counter()
    for _ in range(200): dist.all_reduce(t)
    torch.cuda.synchronize()
    e=time.perf_counter()
    if rank==0: print(f"[r0] avg={(e-s)/200*1e6:.1f}us bw={2*(world-1)/world*(16777216*4)/((e-s)/200)/1e9:.2f}GB/s",flush=True)
    dist.destroy_process_group()
if __name__=="__main__": mp.spawn(worker,args=(2,),nprocs=2,start_method="spawn")
PYEOF

echo "===== 运行 C 基准 allreduce (仓库自带 tests, 用我们编译的 lib) ====="
LD_LIBRARY_PATH="$MINI_NCCL_DIR/build/lib" \
"$MINI_NCCL_DIR/tests/build/all_reduce_perf" -b 8 -e 128M -f 2 -g 2 2>&1 | tail -6

echo "===== 运行 python test (LD_PRELOAD 我们的 NCCL) ====="
if python3 -c "import torch" 2>/dev/null; then
  LD_PRELOAD="$MINI_NCCL_DIR/build/lib/libnccl.so.2" \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL \
  MASTER_PORT=29700 \
  python3 /root/allreduce_test.py 2>&1 | grep -E "r0|error|Error" | tail -5
else
  echo "  (未检测到 torch, 跳过 python test; 如需请先 pip install torch)"
fi
echo "===== 完成 ====="
