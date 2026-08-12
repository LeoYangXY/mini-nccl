/*
 * src/include/os/linux.h — Linux 平台 OS 抽象层头
 * ----------------------------------------------------------------------------
 * 集中包含 Linux 下所需的系统头（socket、pthread、mmap、dlfcn 等），
 * 并定义跨平台代码在 Linux 上依赖的宏与类型别名。
 */

#ifndef NCCL_OS_LINUX_H_
#define NCCL_OS_LINUX_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <getopt.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <linux/types.h>
#include <endian.h>
#include <sys/resource.h>
#include <link.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>

#define NCCL_INVALID_SOCKET -1
typedef int ncclSocketDescriptor;

typedef cpu_set_t ncclAffinity;

typedef pid_t ncclPid_t;

#define NCCL_POLLIN POLLIN
#define NCCL_POLLERR POLLHUP

#endif
