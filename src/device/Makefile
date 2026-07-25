#
# mini-nccl device kernels: ring/tree algorithms with Simple/LL/LL128 protocols.
# Symmetric kernels are removed (stubbed on the host side).
# Derived from NVIDIA NCCL src/device/Makefile (Apache-2.0).
#

SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/device

MANIFEST := $(OBJDIR)/manifest
DEVGLUE_OBJ  := $(OBJDIR)/device_glue.o

# Only build AllReduce kernels (all redops/types) for RING/TREE algorithms
# with all protocols (SIMPLE, LL, LL128).
ONLY_FUNCS ?= AllReduce * * (RING|TREE) *

INCFLAGS  = -I. -I.. -I$(BUILDDIR)/include -I../include -I../include/plugin
NVCUFLAGS += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"
CXXFLAGS  += $(INCFLAGS)

ifeq ($(NCCL_OS_LINUX), 1)
NVCUFLAGS += -DNCCL_OS_LINUX
endif

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.kernel = $(NVCC) $(NVCUFLAGS) -dw $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(or $3,$(suffix $2)),$1,$2)
endef

DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -MM -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -MM -c $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef

all: $(MANIFEST)

$(OBJDIR)/gensrc: generate.py
	@mkdir -p $@
	(which python3 >/dev/null || \
	  (bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'; \
	   printf "\n$${bar}\nERROR: Building NCCL requires a Python 3 installation invokable as 'python3'.\n$${bar}\n\n" 1>&2; \
	   exit 1)) \
	&& ./generate.py $@ "$(ONLY_FUNCS)"

# The trailing ";" is necessary to make this an "empty recipe":
# https://www.gnu.org/software/make/manual/html_node/Empty-Recipes.html
$(OBJDIR)/gensrc/rules.mk: $(OBJDIR)/gensrc ;

-include $(OBJDIR)/gensrc/rules.mk
# "gensrc/rules.mk" populates $(LIB_OBJS_GEN)

SRCS = common.cu onerank.cu

LIB_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS)) $(LIB_OBJS_GEN)

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/%.d: $(OBJDIR)/gensrc/%
	$(call DEPENDS,$@,$<)

# Device link: combine all device .o into device_glue.o
$(DEVGLUE_OBJ): $(LIB_OBJS)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

# Manifest lists device objs for src/Makefile; they get linked into libnccl.so
$(MANIFEST): $(LIB_OBJS) $(DEVGLUE_OBJ)
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(OBJDIR)/genobj/*.d)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
