#
# mini-nccl: lean single-node NCCL build (2-GPU P2P/SHM, ring/tree, LL/LL128/Simple)
# Derived from NVIDIA NCCL src/Makefile (Apache-2.0). See LICENSE.txt.
#
include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := nccl.h nccl_device.h \
	$(patsubst include/%,%,$(wildcard include/nccl_device/*.h include/nccl_device/*/*.h include/nccl_device/*/*/*.h))

LIBSRCFILES := \
	bootstrap.cc channel.cc collectives.cc debug.cc enqueue.cc group.cc \
	init.cc proxy.cc transport.cc mnnvl.cc allocator.cc dev_runtime.cc dev_runtime_segments.cc \
	sym_kernels.cc sym_kernels_host_stub.cc ce_coll.cc mem_manager.cc \
	$(wildcard graph/*.cc) \
	$(wildcard misc/*.cc) \
	$(wildcard transport/*.cc) \
	$(wildcard register/*.cc) \
	$(wildcard nccl_device/*.cc) \
	$(wildcard scheduler/*.cc) \
	$(wildcard devcomm/*.cc) \
	$(wildcard param/*.cc) \
	os/linux.cc os/linux_ipcsocket.cc os/linux_socket_pair.cc os/linux_stubs.cc

ifneq ($(NVTX), 0)
	LIBSRCFILES += init_nvtx.cc
endif

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig

##### target files
CUDARTLIB  ?= cudart_static

# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
ifeq ($(CUDARTLIB), cudart_static)
	LIBSRCFILES += enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl
INCPLUGIN  := include/plugin

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest

GIT_VERSION_FILE := $(OBJDIR)/include/nccl_git_version.h
$(GIT_VERSION_FILE): ALWAYS_REBUILD
	@mkdir -p $(dir $@)
	@./misc/generate_git_version.py $@

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device

# Empty target to force rebuild
.PHONY: ALWAYS_REBUILD

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ../makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

# Device objs (from device/Makefile manifest, incl. common.cu.o) are linked in so libnccl.so
# contains device code that tests resolve when linking with -lnccl.
$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $$(cat $(DEVMANIFEST)) $(LDFLAGS) -Wl,--version-script=libnccl.map
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ) $$(cat $(DEVMANIFEST))

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_device/%.h: include/nccl_device/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(dir $@)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include -Iinclude -I$(INCPLUGIN) $(CXXFLAGS) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include -Iinclude -I$(INCPLUGIN) $(CXXFLAGS) -MM $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/misc/git_version.o: $(GIT_VERSION_FILE)

clean :
	$(MAKE) -C device clean
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v -r $(BUILDDIR)/include/* $(PREFIX)/include/
