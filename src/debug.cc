/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <ctime>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include "param.h"
#include "param/param.h"
#include <mutex>
#include "os.h"
#include "utils.h"
#include "env.h"
#include <cinttypes>

/* ============================================================================
 * debug.cc —— NCCL 日志/调试系统（单机多卡最小通信库 mini-nccl）
 * ----------------------------------------------------------------------------
 * 职责：提供 INFO/WARN/TRACE 日志宏，按日志级别(NCCL_DEBUG)与子系统
 * (NCCL_DEBUG_SUBSYS)过滤输出。AllReduce 跑不通或想看拓扑/连接细节时，最常用的
 * 就是在这里定义的两个环境变量。
 *
 * 调试 AllReduce 最常用的 NCCL_* 环境变量（定义在各自模块的 DEFINE_NCCL_PARAM 处）：
 *   - NCCL_DEBUG=WARN|INFO|TRACE  日志级别，TRACE 最详细（会打印每次 API 调用）。
 *   - NCCL_DEBUG_SUBSYS=BOOTSTRAP,GRAPH,P2P,PROXY,NET,COLL,...  只打印关心的子系统。
 *   - NCCL_PROTO=Simple|LL|LL128  强制 AllReduce 使用的协议（调试性能差异时用）。
 *   - NCCL_ALGO=Ring|Tree         强制 AllReduce 使用的算法。
 *   - NCCL_P2P=0|1                是否启用 GPU 间 P2P 直连（排查跨卡访问问题用）。
 *   - NCCL_NET=... / NCCL_SOCKET_IFNAME=...  网络/网卡相关（节点间通信时）。
 *   - NCCL_TOPO_FILE=xxx.xml      用静态 XML 指定拓扑，跳过实时探测（排查拓扑识别错）。
 *   - NCCL_CONF_FILE=xxx.conf     从文件批量读这些变量（~/.nccl.conf 默认）。
 * 例：NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=BOOTSTRAP,GRAPH 看建连与拓扑搜索全过程。
 * ============================================================================
 */

#define NCCL_DEBUG_RESET_TRIGGERED (-2)

int ncclDebugLevel = -1;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = stdout;
static std::mutex ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
static bool ncclWarnSetDebugInfo = false;

static thread_local int tid = -1;

// clang-格式 off
// ===== NCCL 日志/调试相关环境变量（本段集中定义，学习时最常用的几个）=====
// 这些 DEFINE_NCCL_PARAM 的第二参数是环境变量名（如 NCCL_DEBUG），运行时通过
// getenv 读取；在 mini-nccl 里常用于排查 全规约 建连/拓扑/性能问题。
DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamOneOf<ncclDebugLogLevel>(makeOptions(
                    makeOption("VERSION", NCCL_LOG_VERSION, "Prints the NCCL version info only"),
                    makeOption("WARN", NCCL_LOG_WARN, "Prints only messages indicating a fatal error."),
                    makeOption("INFO", NCCL_LOG_INFO, "Prints debug message"),
                    makeOption("ABORT", NCCL_LOG_ABORT, ""),
                    makeOption("TRACE", NCCL_LOG_TRACE, "Prints replayable trace info on all calls")
                  )), "Set debug output level, the option is inclusive for any level that is less verbose than the set value");

// NCCL_DEBUG_SUBSYS：按“子系统”过滤日志(逗号分隔)。例如只关心建连与图搜索时设
// NCCL_DEBUG_SUBSYS=BOOTSTRAP,图。默认已含 初始化/ENV/BOOTSTRAP。其余可选 COLL/P2P/SHM/
// 网络/图/TUNING/ALLOC/调用/代理/NVLS/REG/剖析/RAS/销毁，所有 表示全部。
DEFINE_NCCL_PARAM(ncclParamDebugSubsys, uint64_t, NCCL_DEBUG_SUBSYS,
                  NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  (ncclParamBitsetOf<ncclDebugLogSubSys, uint64_t>(makeOptions(
                    makeOption("INIT", NCCL_INIT, "NCCL and comm initialization (included in default)"),
                    makeOption("COLL", NCCL_COLL, "Collective operations"),
                    makeOption("P2P", NCCL_P2P, "Peer-to-peer transport"),
                    makeOption("SHM", NCCL_SHM, "Shared memory transport"),
                    makeOption("NET", NCCL_NET, "Network transport"),
                    makeOption("GRAPH", NCCL_GRAPH, "Graph search and topology"),
                    makeOption("TUNING", NCCL_TUNING, "Algorithm tuning"),
                    makeOption("ENV", NCCL_ENV, "Parameter settings by config file, EnvVar or EnvPlugins (included in default)"),
                    makeOption("ALLOC", NCCL_ALLOC, "Device memory allocation"),
                    makeOption("ALLOC_HOST", NCCL_ALLOC_HOST, "Host memory allocation"),
                    makeOption("CALL", NCCL_CALL, "API call tracing"),
                    makeOption("PROXY", NCCL_PROXY, "Proxy thread operations"),
                    makeOption("NVLS", NCCL_NVLS, "NVLink SHARP operations"),
                    makeOption("BOOTSTRAP", NCCL_BOOTSTRAP, "Bootstrap network (included in default)"),
                    makeOption("REG", NCCL_REG, "Buffer registration"),
                    makeOption("PROFILE", NCCL_PROFILE,   "Profiling"),
                    makeOption("RAS", NCCL_RAS, "Reliability, availability, serviceability"),
                    makeOption("DESTROY", NCCL_DESTROY, "Communicator destroy, abort, revoke, and plugin unload/close operations"),
                    makeOption("ALL", NCCL_ALL, "All categories")
                  ))), "Filter debug output by (comma-separated)");

// NCCL_WARN_ENABLE_DEBUG_INFO：一旦打出 WARN 级别消息，就自动把调试级别提升到 信息，
// 方便在出错后看到更多上下文。
DEFINE_NCCL_PARAM(ncclParamWarnEnableDebugInfo, bool, NCCL_WARN_ENABLE_DEBUG_INFO, false,
                  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "If enabled, the debug level will be set to INFO after a WARN level debug message is logged.");
// clang-格式 on

// NCCL_DEBUG_TIMESTAMP_LEVELS：给哪些级别的日志行加时间戳(位掩码,如 WARN/信息/追踪)。
DEFINE_NCCL_PARAM(ncclParamDebugTimestampLevel, uint32_t, NCCL_DEBUG_TIMESTAMP_LEVELS, (1u << NCCL_LOG_WARN),
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(
                    makeOptions(makeOption("VERSION", (1u << NCCL_LOG_VERSION), "on NCCL version info message"),
                                makeOption("WARN", (1u << NCCL_LOG_WARN), "on Explicit error message"),
                                makeOption("INFO", (1u << NCCL_LOG_INFO), "on Debug message"),
                                makeOption("ABORT", (1u << NCCL_LOG_ABORT), ""),
                                makeOption("TRACE", (1u << NCCL_LOG_TRACE), "on Replayable trace message"),
                                makeOption("ALL",
                                           (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN | 1u << NCCL_LOG_INFO |
                                            1u << NCCL_LOG_ABORT | 1u << NCCL_LOG_TRACE),
                                           "on All messages"))),
                  "Set which log lines get a timestamp depending upon the level of the log");

// NCCL_DEBUG_TIMESTAMP_FORMAT：日志时间戳的打印格式(传给 strftime)。
DEFINE_NCCL_PARAM(ncclParamDebugTsFormat, const char*, NCCL_DEBUG_TIMESTAMP_FORMAT, "[%F %T] ",
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the format used when printing debug log messages");

// NCCL_DEBUG_FILE：把调试日志写到文件而非 stdout。可用 %h(主机名)/%p(PID) 占位符。
DEFINE_NCCL_PARAM(ncclParamDebugFile, const char*, NCCL_DEBUG_FILE, nullptr,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the NCCL debug logging output to a file. The filename format can be set to "
                  "filename.%h.%p where %h is replaced with the hostname "
                  "and %p is replaced "
                  "with the process PID. This does not accept the ~ character as part of the path, "
                  "please convert to a relative or absolute path first.");

typedef const char* (*ncclGetEnvFunc_t)(const char*);

static ncclResult_t getHostNameForLog(char* hostname, int maxlen, const char delim) {
  ncclResult_t ret = getHostName(hostname, maxlen, delim);
  if (ret != ncclSuccess) return ret;

  for (int i = 0; i < maxlen - 1 && hostname[i]; ++i) {
    // Replace 特殊的 characters 入 hostnames with dashes
    switch (hostname[i]) {
    case '%':
    case '/':
      hostname[i] = '-';
      break;
    default:
      break;
    }
  }
  return ncclSuccess;
}

// 该函数 必须为 被调用 with ncclDebugLock 已加锁!
static void ncclDebugInit() {
  int tempNcclDebugLevel = -1;
  if (ncclDebugLevel == NCCL_DEBUG_RESET_TRIGGERED && ncclDebugFile != stdout) {
    // 完成 the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  tempNcclDebugLevel = ncclParamDebugLevel();

  ncclWarnSetDebugInfo = ncclParamWarnEnableDebugInfo();

  // Determine 该 调试 层级 will have timestamps.
  ncclDebugTimestampLevels = ncclParamDebugTimestampLevel();

  // 存储 a 拷贝 的 timestamp 格式 with space 为了 subseconds, 若 已使用.
  const char* tsFormat = ncclParamDebugTsFormat();
  ncclDebugTimestampSubsecondsStart = -1;
  // 查找 何処 the subseconds are 在 ... 中 格式.
  for (int i = 0; tsFormat[i] != '\0'; ++i) {
    if (tsFormat[i] == '%' && tsFormat[i + 1] == '%') {
      // 下一个 two chars are "%"
      // 跳过 下一个 character, too, 并且 restart checking 之后 那个.
      ++i;
      continue;
    }
    if (tsFormat[i] == '%' &&                               // Found a percentage
        ('1' <= tsFormat[i + 1] && tsFormat[i + 1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i + 2] == 'f'                                // Two characters later is an "f"
    ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i + 1] - '0';
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; 回退 on 默认值.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      for (int j = 0; j < ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i + j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      strcpy(ncclDebugTimestampFormat + i + ncclDebugTimestampSubsecondDigits, tsFormat + i + replaceLen);
      break;
    }
  }
  if (ncclDebugTimestampSubsecondsStart == -1) {
    if (strlen(tsFormat) < sizeof(ncclDebugTimestampFormat)) {
      strcpy(ncclDebugTimestampFormat, tsFormat);
    } else {
      strcpy(ncclDebugTimestampFormat, "[%F %T] ");
    }
  }

  // Replace underscore with spaces... 这是 hard to 放置 spaces 入 command 行 参数.
  for (int i = 0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i] == '_') ncclDebugTimestampFormat[i] = ' ';
  }

  // 缓存 pid 并且 hostname
  getHostNameForLog(hostname, 1024, '.');
  pid = ncclOsGetPid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = ncclParamDebugFile();
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char* dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c - 1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
      case '%': // Double %
        *dfn++ = '%';
        break;
      case 'h': // %h = hostname
        dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%s", hostname);
        break;
      case 'p': // %p = pid
        dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", pid);
        break;
      default: // Echo everything we don't understand
        *dfn++ = '%';
        if ((dfn - debugFn) < PATH_MAX) {
          *dfn++ = ncclDebugFileEnv[c - 1];
        }
        break;
      }
      if ((dfn - debugFn) > PATH_MAX) {
        // snprintf wanted to overfill 该缓冲区: 设置 dfn 到 末尾
        // of 该缓冲区 (for null char) 并且 it will naturally 退出
        // the 循环.
        dfn = debugFn + PATH_MAX;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE* file = fopen(debugFn, "w");
      if (file != nullptr) {
#if defined(NCCL_OS_LINUX)
        setlinebuf(file); // disable block buffering
#elif defined(NCCL_OS_WINDOWS)
        setvbuf(file, NULL, _IOLBF, 0); // disable block buffering
#endif
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  ncclDebugMask = ncclParamDebugSubsys();
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, tempNcclDebugLevel, std::memory_order_release);
}

static void ncclDebugLogV(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, va_list vargs) {
  int gotLevel = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire);

  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) {
    level = NCCL_LOG_INFO;
    flags = ncclDebugNoWarn;
  }

  // Save 最后一个 错误 (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vcopy;
    va_copy(vcopy, vargs);
    (void)vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vcopy);
    va_end(vcopy);
  }

  if (gotLevel >= 0 && (gotLevel < level || (flags & ncclDebugMask) == 0)) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  if (ncclDebugLevel < 0) ncclDebugInit();
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) {
    return;
  }

  if (tid == -1) {
    tid = ncclOsGetTid();
  }

  char buffer[1024];
  size_t len = 0;

  // WARNs come with an 额外的 newline at the beginning.
  if (level == NCCL_LOG_WARN) {
    buffer[len++] = '\n';
  }

  // Add the timestamp to 该缓冲区 若y are turned on for 此 层级.
  if (ncclDebugTimestampLevels & (1 << level)) {
    if (ncclDebugTimestampFormat[0] != '\0') {
      struct timespec ts;
      clockRealtime(&ts);
      time_t nowTimeT = ts.tv_sec;
      long nowNs = ts.tv_nsec;
      std::tm nowTm;
      ncclOsLocaltime(&nowTimeT, &nowTm);

      // Add the subseconds portion 若 这是 part 的 格式.
      char localTimestampFormat[sizeof(ncclDebugTimestampFormat)];
      const char* pformat = ncclDebugTimestampFormat;
      if (ncclDebugTimestampSubsecondsStart != -1) {
        pformat = localTimestampFormat;   // Need to use the local version which has subseconds
        memcpy(localTimestampFormat, ncclDebugTimestampFormat, ncclDebugTimestampSubsecondsStart);
        snprintf(localTimestampFormat + ncclDebugTimestampSubsecondsStart, ncclDebugTimestampSubsecondDigits + 1,
                 "%0*" PRIu64, ncclDebugTimestampSubsecondDigits,
                 (uint64_t)(nowNs / (1000000000L / ncclDebugTimestampMaxSubseconds)));
        strcpy(localTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits,
               ncclDebugTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits);
      }

      // 格式 the time. 若 it runs 脱离 space, 回退 on a simpler 格式.
      int adv = std::strftime(buffer + len, sizeof(buffer) - len, pformat, &nowTm);
      if (adv == 0 && ncclDebugTimestampFormat[0] != '\0') {
        // Ran 脱离 space. 回退 on 默认值. 此 should never 失败.
        adv = std::strftime(buffer + len, sizeof(buffer) - len, "[%F %T] ", &nowTm);
      }
      len += adv;
    }
  }
  len = std::min(len, sizeof(buffer) - 1);  // prevent overflows

  // Add hostname, pid 并且 tid portion 的 日志 行.
  if (level != NCCL_LOG_VERSION) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s:%d:%d ", hostname, pid, tid);
    len = std::min(len, sizeof(buffer) - 1);  // prevent overflows
  }

  int cudaDev = 0;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    (void)cudaGetDevice(&cudaDev);
  }

  const char* fileStr = file ? file : "<unknown>";
  const char* funcStr = func ? func : "<unknown>";

  // Add 层级 特定的 formatting. The 格式 string 从 调用 site is incorporated into 此 prefix.
  if (level == NCCL_LOG_WARN) {
    if (func && func[0]) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d (%s) NCCL WARN %s\n", cudaDev, fileStr, line,
                      funcStr, fmt);
    } else {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d NCCL WARN %s\n", cudaDev, fileStr, line, fmt);
    }
    if (ncclWarnSetDebugInfo) {
      COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_LOG_INFO), std::memory_order_release);
    }
  } else if (level == NCCL_LOG_INFO) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] NCCL INFO %s\n", cudaDev, fmt);
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "NCCL CALL %s\n", fmt);
  } else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %f %s:%d NCCL TRACE %s\n", cudaDev, timestamp, funcStr,
                    line, fmt);
  } else {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s\n", fmt);
  }

  // 若 prefixed 格式 string overflows, 确保 这是 仍 terminated with a newline.
  if (len > sizeof(buffer) - 1) {
    // snprintf 已经 placed a \0 at sizeof(缓冲区)-1
    buffer[sizeof(buffer) - 2] = '\n';
  }

  // Add the 消息 as 给定的 由 调用 site.
  // The 调用 site's 格式 string 已经 incorporated into `缓冲区` 连同 our prefix.
  va_list vcopy;
  va_copy(vcopy, vargs);
  (void)vfprintf(ncclDebugFile, buffer, vcopy);
  va_end(vcopy);
}

// 内部 仅 通用 logging 函数 已使用 由 信息, WARN 并且 追踪 宏
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

/* Exported ABI logging function exported to the dynamically loadable Net
 * transport modules so they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  const char* file = nullptr;
  const char* func = nullptr;
  if (level == NCCL_LOG_WARN) {
    file = filefunc;
  } else if (level == NCCL_LOG_TRACE) {
    func = filefunc;
  }
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

// Non-deprecated 版本 for 内部 使用.
extern "C"
#if !defined(NCCL_OS_WINDOWS)
  __attribute__((visibility("default")))
#endif
  void ncclResetDebugInitInternal() {
  // Cleans up from a 前一个 ncclDebugInit() 并且 reruns.
  // 使用 此 之后 changing NCCL_DEBUG 并且 related 参数 在 ... 中 environment.
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // Let ncclDebugInit() 知道 to 完成 the reset.
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_DEBUG_RESET_TRIGGERED), std::memory_order_release);
}

// 就地 of: NCCL_API(void, ncclResetDebugInit);
#ifdef pncclResetDebugInit
#undef pncclResetDebugInit
#endif
#if defined(NCCL_OS_LINUX)
__attribute__((visibility("default"))) __attribute__((alias("ncclResetDebugInit")))
#endif
void pncclResetDebugInit();
extern "C"
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((visibility("default"))) __attribute__((weak)) __attribute__((
    deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
#endif
  void ncclResetDebugInit();

extern "C" void ncclResetDebugInit() {
  // 这是 now deprecated as part 的 NCCL API. It 将会 removed
  // 从 API 在 ... 中 future. 这是 仍 可用 as an
  // 导出的符号。
  ncclResetDebugInitInternal();
}

DEFINE_NCCL_PARAM(ncclParamSetThreadName, bool, NCCL_SET_THREAD_NAME, false,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Allow NCCL to give meaningful names to NCCL CPU threads via pthread_setname_np");

void ncclSetThreadName(std::thread& thread, const char* fmt, ...) {
  // pthread_setname_np 是非标准的 GNU 扩展
  // needs 以下内容 特性 测试 宏
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() == false) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread.native_handle(), threadName);
#endif
}
