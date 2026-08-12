/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/graph/xml.cc — 拓扑/图的 XML 解析与导出
 * ----------------------------------------------------------------------------
 * 实现 NCCL 拓扑与算法图的 XML 读写：把 ncclTopoSystem 导出为 XML（NCCL_TOPO_DUMP_FILE），
 * 或从 NCCL_TOPO_FILE / NCCL_GRAPH_FILE 载入用户自定义的拓扑/图。是“软件自定义
 * 拓扑”与“自定义 graph”的序列化支撑。
 */

#include <ctype.h>
#include <float.h>
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"
#include "os.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// 构造虚拟拓扑字符串时使用的“任意大的数”
#define NCCL_MAX_XML_DEPTH 1024

/*******************/
/* XML File Parser */
/*******************/

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char* name;
  xmlHandlerFunc_t func;
};

#if defined(NCCL_OS_LINUX)

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o] = c;
      if (o == MAX_STR_LEN - 1) {
        value[o] = '\0';
        WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
        return ncclInternalError;
      }
      o++;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  char quote = c;  // Remember which quote type we started with
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o] = c;
    if (o == MAX_STR_LEN - 1) {
      value[o] = '\0';
      WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != quote);
  value[o - 1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN - 1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o - 1] = '\0';
  *last = c;
  return ncclSuccess;
}

// 把 3 字符的字符串整体左移一位，并在末尾追加字符 c
#define SHIFT_APPEND(s, c) \
  do { \
    s[0] = s[1]; \
    s[1] = s[2]; \
    s[2] = c; \
  } while (0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // 从一个中性的、以 \0 结尾的字符串开始。
  char end[4] = "...";

  // 注入上一次读取的所有后续字符。此处无需
  // 检查 -->，因为名称中不可能出现 > 字符。
  for (int i = 0; i < strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // 遇到 "-->" 时停止
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // 读取 XML 元素名
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // 检查是否为注释
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name + 3, c));
    return xmlGetNode(file, node);
  }

  // 检查是否为闭合标签
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // 重新读取名称，因为第一次调用时读到了 '/'
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // 获取属性
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // 实际上我们仍需消费掉多余属性，因此多读一个。
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes + xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // 全部完成
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h = 0; h < nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

#endif

/**************/
/* XML Writer */
/**************/

// exp == 1 表示序列化；exp == 0 表示反序列化
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {
    struct ncclXmlNode* node = &xml->nodes[n];

    // 对 "父"，我们把基址偏移 1，以便区分
    // 真正的 NULL 指针与指向第一个节点的指针。
    if (node->parent) {
      node->parent =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));
    }

    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i = 0; i < indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a = 0; a < node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent + 2, file, node->subs[s]));
    }
    for (int i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    INFO(NCCL_GRAPH | NCCL_ENV, "Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}

static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent,
                                            struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {
    struct ncclXmlNode* srcNode = srcParent->subs[i];
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode));
    } else {
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}

#if NCCL_OS_LINUX

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"nvlink", ncclTopoXmlLoadNvlink}, {"c2c", ncclTopoXmlLoadC2c}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"net", ncclTopoXmlLoadNet}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci},
                                  {"gpu", ncclTopoXmlLoadGpu},
                                  {"nic", ncclTopoXmlLoadNic},
                                  {"pcilink", ncclTopoXmlLoadPciLink}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 4));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci}, {"nic", ncclTopoXmlLoadNic}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading topology %s", name);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = {{"cpu", ncclTopoXmlLoadCpu}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

#endif // NCCL_OS_LINUX (XML file parsers)

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
#if NCCL_OS_LINUX
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = {{"system", ncclTopoXmlLoadSystem}};
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
#elif NCCL_OS_WINDOWS
  (void)xmlTopoFile;
  (void)xml;
  (void)warn;
#endif
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName,
                                    const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclOsTopoGetStrFromSys(path, fileName, strValue, MAX_STR_LEN));
  if (strValue[0] != '\0') NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue));
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoSetAttrFromNvml(struct ncclXmlNode* pciNode, nvmlDevice_t device, const char* attrName) {
  nvmlPciInfo_t pciInfo;
  ncclResult_t ret = ncclNvmlDeviceGetPciInfo(device, &pciInfo);
  if (ret != ncclSuccess) return ret;

  char strValue[MAX_STR_LEN];
  strValue[0] = '\0';
  if (strcmp(attrName, "vendor") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", pciInfo.pciDeviceId & 0xFFFF);
  } else if (strcmp(attrName, "device") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", (pciInfo.pciDeviceId >> 16) & 0xFFFF);
  } else if (strcmp(attrName, "subsystem_vendor") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", pciInfo.pciSubSystemId & 0xFFFF);
  } else if (strcmp(attrName, "subsystem_device") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", (pciInfo.pciSubSystemId >> 16) & 0xFFFF);
  }
  if (strValue[0] == '\0') return ncclInternalError;

  NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue));
  TRACE(NCCL_GRAPH, "Read from NVML %s=%s", attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // 用操作系统相关的实现来设置亲和性
    unsigned int nodeNumber = (unsigned int)strtoul(numaId, NULL, 0);
    char affinityStr[MAX_STR_LEN];
    NCCLCHECK(ncclOsGetNumaNodeAffinity(nodeNumber, affinityStr, sizeof(affinityStr)));
    NCCLCHECK(xmlSetAttr(cpuNode, "affinity", affinityStr));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // 填充 CPU 的型号/厂商/模型信息
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#else
    WARN("getXmlFromCpu: Unknown CPU architecture");
    return ncclInternalError;
#endif
  }

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 指令返回的字符串寄存器顺序
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

#if NCCL_OS_LINUX
    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
#elif NCCL_OS_WINDOWS
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    cpuid0.ebx = cpuInfo[1];
    cpuid0.edx = cpuInfo[3];
    cpuid0.ecx = cpuInfo[2];
#endif
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
#if NCCL_OS_LINUX
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
#elif NCCL_OS_WINDOWS
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    cpuid1.val = cpuInfo[0];  // EAX contains the processor info
#endif
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }
  return ncclSuccess;
}

// 检查字符串是否为 BDF 格式。
// BDF(总线-设备-功能)格式为 "BBBB:BB:DD.F"，其中 B、D、F 为十六进制数字。
// 后面可能还有尾随字符。
int isHex(char c) {
  return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}
int checkBDFFormat(char* bdf) {
  if (strlen(bdf) != 12) return 0;
  if ((bdf[4] != ':') || (bdf[7] != ':') || (bdf[10] != '.')) return 0;
  if ((isHex(bdf[0]) == 0) || (isHex(bdf[1]) == 0) || (isHex(bdf[2]) == 0) || (isHex(bdf[3]) == 0) ||
      (isHex(bdf[5]) == 0) || (isHex(bdf[6]) == 0) || (isHex(bdf[8]) == 0) || (isHex(bdf[9]) == 0) ||
      (isHex(bdf[11]) == 0))
    return 0;
  return 1;
}

ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  ncclResult_t ret = ncclSuccess;
  const char* vendor = NULL;
  struct ncclXmlNode* parent = NULL;
  char* peers = NULL;

  // 先填充信息，再处理父节点
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  nvmlDevice_t device;
  bool nvmlDeviceFound = false;

#if NCCL_OS_LINUX
  char* path = NULL;
  const char* deviceClass = NULL;
  NOWARN(ncclOsGetPciPath(busId, &path), NCCL_GRAPH);
  if (path) NCCLCHECKGOTO(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"), ret, exit);

  NCCLCHECKGOTO(xmlGetAttr(pciNode, "class", &deviceClass), ret, exit);
  if (deviceClass == NULL || deviceClass[0] == '\0' || strncmp(deviceClass, "0x03", 4) == 0) {
    ncclResult_t nvmlRet;
    NOWARN(nvmlRet = ncclNvmlDeviceGetHandleByPciBusId(busId, &device), NCCL_GRAPH);
    if (nvmlRet == ncclSuccess) nvmlDeviceFound = true;
  }

#elif NCCL_OS_WINDOWS
  char* parentBusId = NULL;
  char deviceClass[MAX_STR_LEN];
  deviceClass[0] = '\0';
  bool isGpuDevice = false;
  if (ncclOsGetPciDeviceClassByBusId(busId, deviceClass, sizeof(deviceClass)) == ncclSuccess &&
      deviceClass[0] != '\0') {
    NCCLCHECK(xmlSetAttr(pciNode, "class", deviceClass));
    TRACE(NCCL_GRAPH, "Read from Windows SetupDi class=%s", deviceClass);
    isGpuDevice = (strncmp(deviceClass, "0x03", 4) == 0);
  } else {
    TRACE(NCCL_INIT, "ncclTopoGetXmlFromSys: Could not get device class for %s", busId);
  }
  if (isGpuDevice) {
    ncclResult_t winNvmlRet;
    NOWARN(winNvmlRet = ncclNvmlDeviceGetHandleByPciBusId(busId, &device), NCCL_GRAPH);
    if (winNvmlRet == ncclSuccess) nvmlDeviceFound = true;
  }
#endif

  int index;
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "vendor", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "vendor"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "device", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "device"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "device", "device"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "subsystem_vendor"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "subsystem_device", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "subsystem_device"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTO(xmlGetAttrIndex(pciNode, "link_speed", &index), ret, exit);
  if (index == -1) {
    if (nvmlDeviceFound) {
      unsigned int linkGen = 0;
      if (ncclNvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen) == ncclSuccess && linkGen > 0) {
        const char* speeds[] = {
          "", "2.5 GT/s PCIe", "5.0 GT/s PCIe", "8.0 GT/s PCIe", "16.0 GT/s PCIe", "32.0 GT/s PCIe", "64.0 GT/s PCIe"
        };
        if (linkGen <= 6) {
          NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", speeds[linkGen]), ret, exit);
        }
      } else {
        NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", "16.0 GT/s"), ret, exit);
      }
    }
#if NCCL_OS_LINUX
    else if (path) {
      char deviceSpeedStr[MAX_STR_LEN];
      float deviceSpeed = FLT_MAX;
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr, sizeof(deviceSpeedStr)), ret, exit);
      sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
      char portSpeedStr[MAX_STR_LEN];
      float portSpeed = FLT_MAX;
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr, sizeof(portSpeedStr)), ret, exit);
      sscanf(portSpeedStr, "%f GT/s", &portSpeed);
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr), ret,
                    exit);
    }
#endif
    else {
#if NCCL_OS_LINUX
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", ""), ret, exit);
#elif NCCL_OS_WINDOWS
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", "16.0 GT/s"), ret, exit);
#endif
    }
  }
  // 从 NVML(共享)获取链路宽度，Linux 下回退到 sysfs，再不行用默认值
  NCCLCHECKGOTO(xmlGetAttrIndex(pciNode, "link_width", &index), ret, exit);
  if (index == -1) {
    if (nvmlDeviceFound) {
      unsigned int linkWidth = 0;
      if (ncclNvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth) == ncclSuccess && linkWidth > 0) {
        NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", linkWidth), ret, exit);
      } else {
        NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", 16), ret, exit);
      }
    }
#if NCCL_OS_LINUX
    else if (path) {
      char strValue[MAX_STR_LEN];
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "max_link_width", strValue, MAX_STR_LEN), ret, exit);
      int deviceWidth = strtol(strValue, NULL, 0);
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "../max_link_width", strValue, MAX_STR_LEN), ret, exit);
      int portWidth = strtol(strValue, NULL, 0);
      NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth, portWidth)), ret, exit);
    }
#endif
    else {
#if NCCL_OS_LINUX
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_width", ""), ret, exit);
#elif NCCL_OS_WINDOWS
      NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", 16), ret, exit);
#endif
    }
  }

  NCCLCHECKGOTO(xmlGetAttr(pciNode, "vendor", &vendor), ret, exit);
  if (vendor != NULL && strcmp(vendor, "0x1000") == 0) {
    // 博通(BCM)交换机，查找 P2P 连接
    int nlinks;
    NCCLCHECKGOTO(ncclOsGetBcmLinks(busId, &nlinks, &peers), ret, exit);
    for (int l = 0; l < nlinks; l++) {
      char* target = peers + l * BUSID_SIZE;
      struct ncclXmlNode* linkNode;
      NCCLCHECKGOTO(xmlGetSubKv(pciNode, "pcilink", &linkNode, "target", target), ret, exit);
      if (linkNode == NULL) {
        NCCLCHECKGOTO(xmlAddNode(xml, pciNode, "pcilink", &linkNode), ret, exit);
        NCCLCHECKGOTO(xmlSetAttr(linkNode, "target", target), ret, exit);
      }
    }
    free(peers);
    peers = NULL;
  }

  parent = pciNode->parent;
  if (parent == NULL) {
#ifdef NCCL_OS_LINUX
    if (path) {
      // 暂存该信息，以备下一步是 CPU 时使用
      char numaIdStr[MAX_STR_LEN];
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "numa_node", numaIdStr, MAX_STR_LEN), ret, exit);

      // 在 PCI 树中上溯一层。回退两个 "/" 并沿上级 PCI
      // 交换机继续，若抵达 CPU 根复合体则停止。
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path) - 1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start > 0 && path[start] != '/') start--;
          // 检查父路径是否形如 "BBBB:BB:DD.F"。
          if (checkBDFFormat(path + start + 1) == 0) {
            // 这是一个 CPU 根复合体。创建 CPU 标签并在此停止。
            struct ncclXmlNode* topNode;
            NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
            NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);
            if (parent == NULL) {
              NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
              NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
              NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
            }
          } else if (slashCount == 2) {
            // 沿上级 PCI 交换机继续
            for (int i = strlen(path) - 1; i > 0; i--) {
              if (path[i] == '/') {
                NCCLCHECKGOTO(xmlFindTagKv(xml, "pci", &parent, "busid", path + i + 1), ret, exit);
                if (parent == NULL) {
                  NCCLCHECKGOTO(xmlAddNode(xml, NULL, "pci", &parent), ret, exit);
                  NCCLCHECKGOTO(xmlSetAttr(parent, "busid", path + i + 1), ret, exit);
                }
                break;
              }
            }
          }
        }
        if (parent) break;
      }
      free(path);
      path = NULL;
    }
#elif NCCL_OS_WINDOWS
    INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: Windows - creating parent node");
    if (nvmlDeviceFound) {
      char numaIdStr[MAX_STR_LEN] = "0";
      INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: Using NUMA node %s (Windows default)", numaIdStr);

      // 用 Windows 设置 API 获取 PCI 设备的父节点
      ncclResult_t result = ncclOsGetPciDeviceParent(device, &parentBusId);

      if (result == ncclSuccess && parentBusId != NULL) {
        // 检查父节点是否为合法的 PCI 设备(符合 BDF 格式)或 CPU 根复合体
        if (checkBDFFormat(parentBusId) == 1) {
          // 沿上级 PCI 交换机继续
          NCCLCHECKGOTO(xmlFindTagKv(xml, "pci", &parent, "busid", parentBusId), ret, exit);
          if (parent == NULL) {
            NCCLCHECKGOTO(xmlAddNode(xml, NULL, "pci", &parent), ret, exit);
            NCCLCHECKGOTO(xmlSetAttr(parent, "busid", parentBusId), ret, exit);
          }
        } else {
          // 这是 a CPU 根 complex. 创建 a CPU tag 并且 停止 there.
          struct ncclXmlNode* topNode;
          NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
          NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);
          if (parent == NULL) {
            NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
            NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
            NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
          }
        }
      } else {
        // 获取父节点失败——默认当作 CPU 根复合体
        INFO(NCCL_GRAPH, "ncclTopoGetXmlFromSys: could not get PCI parent for %s, defaulting to CPU root complex",
             busId);
        struct ncclXmlNode* topNode;
        NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
        NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);
        if (parent == NULL) {
          NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
          NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
          NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
        } else {
          INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: CPU node already exists (default)");
        }
      }
    }
#endif
    else {
      // /sys 中无信息，把 GPU 挂到“未知 CPU”上
      NCCLCHECKGOTO(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"), ret, exit);
      if (parent == NULL) {
        struct ncclXmlNode* topNode;
        NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
        NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
        NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
        NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", "-1"), ret, exit);
        NCCLCHECKGOTO(ncclTopoGetXmlFromCpu(parent, xml), ret, exit);
      }
    }
    pciNode->parent = parent;
    // 保持 PCI 子设备按 PCI 总线 ID 排序(问题 #820)
    // Coverity complains about dereferenced 父 being NULL
    // 但 此 can never happen.
    // coverity[var_deref_op]
    int subIndex = parent->nSubs;
    const char* newBusId;
    NCCLCHECKGOTO(xmlGetAttrStr(pciNode, "busid", &newBusId), ret, exit);
    for (int s = 0; s < parent->nSubs; s++) {
      const char* busId;
      NCCLCHECKGOTO(xmlGetAttr(parent->subs[s], "busid", &busId), ret, exit);
      if (busId != NULL && strcmp(newBusId, busId) < 0) {
        subIndex = s;
        break;
      }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      ret = ncclInternalError;
      goto exit;
    }
    for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s - 1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECKGOTO(ncclTopoGetXmlFromSys(parent, xml), ret, exit);
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECKGOTO(ncclTopoGetXmlFromCpu(parent, xml), ret, exit);
  }
exit:
  free(peers);
#if NCCL_OS_LINUX
  free(path);
#elif NCCL_OS_WINDOWS
  free(parentBusId);
#endif
  return ret;
}

ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, nvmlDevice_t nvmlDev, struct ncclXml* xml,
                                   struct ncclXmlNode** gpuNodeRet) {
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) {
    *gpuNodeRet = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int cudaMajor, cudaMinor;
    if (nvmlDev == NULL) {
      cudaDeviceProp devProp;
      CUDACHECK(cudaGetDeviceProperties(&devProp, dev));
      cudaMajor = devProp.major;
      cudaMinor = devProp.minor;
    } else {
      NCCLCHECK(ncclNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor * 10 + cudaMinor));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct ncclXmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));
  if (nvlNode == NULL) {
    // NVML NVLink 检测
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      INFO(NCCL_GRAPH, "No NVML device handle. Skipping nvlink detection.");
      maxNvLinks = 0;
    }

    for (int l = 0; l < maxNvLinks; ++l) {
      // 检查这条 NVLink 是否可用于 P2P
      unsigned canP2P;
      if ((ncclNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) ||
          !canP2P) {
        continue;
      }

      // 确保 NVLink 已建立。上一次调用应当已训练好链路。
      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = l;
        // fv.值 将包含 NV_FEATURE_ENABLED 或 NV_FEATURE_DISABLED
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS))
          isActive = (nvmlEnableState_t)fv.value.uiVal;
      } else /* FALLTHRU to GetNvLinkState if before SM90 */
#endif
      {
        (void)ncclNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
      }
      if (isActive != NVML_FEATURE_ENABLED) continue;

      // 尝试判断 NVLink 另一端连接的是什么
      nvmlPciInfo_t remoteProc = {};
      if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

      // 为调用 ncclDeviceType 制作 总线 ID 的小写副本
      // PCI 系统路径都是小写的。
      // NVML 可能为不可见的远端设备(例如 Windows 上的 NVSwitch)返回空或不可打印的 busId，
      // 此时改用哨兵值。
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      if (p[0] == '\0') {
        strncpy(lowerId, "fffffff:ffff:ff", NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
      } else {
        for (int c = 0; c < NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
          if (p[c] && !isprint((unsigned char)p[c])) {
            strncpy(lowerId, "fffffff:ffff:ff", NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
            break;
          }
          lowerId[c] = tolower(p[c]);
          if (p[c] == 0) break;
        }
      }

      NCCLCHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count + 1));
      }
    }
  }
#if CUDART_VERSION >= 11080
  struct ncclXmlNode* c2cNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));
  if (c2cNode == NULL) {
    if (sm >= 90) {
      int c2cLinksCount = 0;
      nvmlFieldValue_t fv;
      fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
      if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS)) {
        c2cLinksCount = fv.value.uiVal;
        int bw = 0;
        int count = 0;
        for (int l = 0; l < c2cLinksCount; l++) {
          nvmlFieldValue_t fvs[2];
          fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
          fvs[0].scopeId = l;
          fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
          fvs[1].scopeId = l;
          if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == ncclSuccess) && (fvs[0].nvmlReturn == NVML_SUCCESS) &&
              (fvs[0].value.uiVal == 1) && (fvs[1].nvmlReturn == NVML_SUCCESS)) {
            bw = fvs[1].value.uiVal;
            count++;
          }
        }
        if (count > 0) {
          NCCLCHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
          NCCLCHECK(xmlSetAttrInt(c2cNode, "bw", bw));
          NCCLCHECK(xmlSetAttrInt(c2cNode, "count", count));
        }
      }
    }
  }
#endif
  // 填充目标类别
  for (int s = 0; s < gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "nvlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      if (strcmp(busId, "fffffff:ffff:ff") == 0) {
        // 远端 NVLink 设备在 VM 内不可见。假定为 NVSwitch。
        INFO(NCCL_GRAPH, "NVLink target %s not visible, assuming NVSwitch", busId);
        NCCLCHECK(xmlSetAttr(sub, "tclass", PCI_NVSWITCH_CLASS));
      } else {
        char deviceClass[MAX_STR_LEN];
        deviceClass[0] = '\0';
        if (ncclOsGetPciDeviceClassByBusId(busId, deviceClass, sizeof(deviceClass)) == ncclSuccess &&
            deviceClass[0] != '\0') {
          NCCLCHECK(xmlSetAttr(sub, "tclass", deviceClass));
          TRACE(NCCL_GRAPH, "Read NVLink target class: tclass=%s for busId=%s", deviceClass, busId);
        } else {
          INFO(NCCL_GRAPH, "Could not get device class for NVLink target %s, assuming NVSwitch", busId);
          NCCLCHECK(xmlSetAttr(sub, "tclass", PCI_NVSWITCH_CLASS));
        }
      }
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));
  NCCLCHECK(xmlSetAttrIfUnset(node, "class", "0x03"));
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));
  nvmlDevice_t nvmlDev;
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
  return ncclSuccess;
}

// 返回路径的子系统名，即路径末尾
// sysPath/subsystem 所指向的那一段。
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
#if NCCL_OS_LINUX
  char subSysPath[PATH_MAX];
  snprintf(subSysPath, sizeof(subSysPath), "%s/subsystem", sysPath);
  char* path = ncclOsRealpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path + offset + 1);
    free(path);
  }
#elif NCCL_OS_WINDOWS
  (void)sysPath;
  subSys[0] = '\0';
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* tagName, const char* pciPath, const char* netName,
                             struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent) {
  NCCLCHECK(xmlFindTagKv(xml, tagName, netNode, "name", netName));

  if (*netNode != NULL) return ncclSuccess;

  struct ncclXmlNode* parent = NULL;
  if (forceParent) {
    parent = forceParent;
  } else {
    const char* pciSysPath = pciPath;
    if (pciSysPath) {
      char subSystem[PATH_MAX];
      NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
      // 这不是 PCI 设备(虚拟设备、USB 等)。
      if (strcmp(subSystem, "pci") != 0 && !forceParent) {
        INFO(NCCL_NET | NCCL_GRAPH,
             "Topology detection: network path (name = %s) %s is not a PCI device (%s). Attaching to first CPU",
             netName, pciSysPath, subSystem);
        pciSysPath = NULL;
      }
    }

    if (pciSysPath) {
      int offset;
      for (offset = strlen(pciSysPath) - 1; pciSysPath[offset] != '/'; offset--);
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      strcpy(busId, pciSysPath + offset + 1);
      NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
      NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
      NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
    } else {
      // 虚拟网卡，没有 PCI 设备，挂到第一个 CPU 上
      NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
    }
  }

  if (parent == NULL) {
    // 尚不存在 CPU 节点，创建一个默认节点
    struct ncclXmlNode* topNode;
    NCCLCHECK(xmlFindTag(xml, "system", &topNode));
    if (topNode) {
      NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
      NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
      NCCLCHECK(xmlSetAttr(parent, "numaid", "0"));
    }
  }
  if (parent == NULL) {
    WARN("ncclTopoFillNet: parent is NULL even after fallback for %s", netName);
    return ncclInternalError;
  }

  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // 我们知道这块网卡还不存在(我们在本函数开头已搜索过)，
  // 因此可以直接添加。
  NCCLCHECK(xmlAddNode(xml, nicNode, tagName, netNode));
  NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node, int* keep) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
    *keep = 1;
  } else {
    // 拷贝 nSubs 与 subs，因为在递归裁剪时它们可能变化。
    struct ncclXmlNode** subs = NULL;
    NCCLCHECK(ncclCalloc(&subs, MAX_SUBS));
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs * sizeof(struct ncclXmlNode*));
    *keep = 0;
    ncclResult_t subsRes = ncclSuccess;
    for (int s = 0; s < nSubs; s++) {
      int k = 0;
      subsRes = ncclTopoTrimXmlRec(subs[s], &k);
      if (subsRes != ncclSuccess) {
        break;
      }
      *keep += k;
    }
    free(subs);
    NCCLCHECK(subsRes);
    // 若节点既没有子节点也没有 保留 属性，则移除它
    if (*keep == 0 && // Trim PCI switches, CPUs with no used GPU/NIC under them, or pruned NICs
        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0 || strcmp(node->name, "nic") == 0 ||
         strcmp(node->name, "net") == 0)) {
#ifdef ENABLE_TRACE
      const char* name;
      const char* busid;
      NCCLCHECK(xmlGetAttr(node, "name", &name));
      NCCLCHECK(xmlGetAttr(node, "busid", &busid));
      TRACE(NCCL_GRAPH, "Removing node %s %s %s\n", node->name, name, busid);
#endif
      NCCLCHECK(xmlRemoveNode(node));
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  int keep = 0;
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes, &keep));
  return ncclSuccess;
}

#if NCCL_OS_LINUX

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"net", ncclTopoXmlGraphLoadNet}, {"gpu", ncclTopoXmlGraphLoadGpu}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"channel", ncclTopoXmlGraphLoadChannel}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = {{"graph", ncclTopoXmlGraphLoadGraph}};
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = {{"graphs", ncclTopoXmlGraphLoadGraphs}};
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

#elif NCCL_OS_WINDOWS

/* Stub implementations for Windows */

ncclResult_t xmlGetChar(FILE* file, char* c) {
  (void)file;
  (void)c;
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  (void)file;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  (void)file;
  (void)name;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  (void)file;
  (void)start;
  (void)next;
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  (void)file;
  (void)node;
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  (void)file;
  (void)xml;
  (void)head;
  (void)handlers;
  (void)nHandlers;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  (void)file;
  (void)xmlGraph;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  (void)xmlGraphFile;
  (void)xml;
  return ncclSuccess;
}

#endif
