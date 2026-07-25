#
# mini-nccl: lean single-node NCCL (bootstrap + topology + ring/tree + Simple/LL/LL128)
#
.PHONY : all lib staticlib clean install tests test

default : lib
all : lib staticlib
lib staticlib clean install :
	${MAKE} -C src $@ BUILDDIR=$(abspath ./build)

# 构建仓库自带的 all_reduce 测试 (tests/ 子目录, 默认链接本仓库 build/lib)
tests : lib
	${MAKE} -C tests NCCL_HOME=$(abspath ./build)

# 一键构建并运行 all_reduce 验证 (0 OK 即正确)
test : tests
	LD_LIBRARY_PATH=$(abspath ./build/lib) ./tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
