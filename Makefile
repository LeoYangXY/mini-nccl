#
# mini-nccl: lean single-node NCCL (bootstrap + topology + ring/tree + Simple/LL/LL128)
#
.PHONY : all lib staticlib clean install

default : lib
all : lib staticlib
lib staticlib clean install :
	${MAKE} -C src $@ BUILDDIR=$(abspath ./build)
