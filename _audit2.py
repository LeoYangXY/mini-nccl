#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, re

has_cjk = re.compile(r'[\u4e00-\x{9fff}]')

# Comprehensive whitelist of English tokens allowed to stay in Chinese comments.
WHITE = set("""
nccl nccls ncclenqueue ncclcomm ncclgroup ncclstream ncclresult ncclerror nccldev ncclproxy
cuda cudamem cudastream cudagraph cudaevent cudadevice cudaipc cudalaunch
gpu gpus cpu cpus
p2p nvlink nvls nvswitch nvml mnnvl gdrcopy gdr shm fifo uuid hash
rank ranks peerranks nranks peer peers window windows segment segments
va fabric partition partitions clique pxn pxb pxn p2c phb nvb collnet collnet
topo toporanks timer timers bootstrap bootstraptag comm comms
channel channels ring rings tree trees protocol proto protocols proxy proxies plugin plugins
tuner tunerv6 heap pool pools mem memory kernel kernels thread threads block blocks
warp warps stream streams device devices host hosts event events signal signals
flag flags handle handles ptr pointer pointers buffer buffers alloc allocator dealloc
free malloc calloc memcpy memset copy copies load store loads stores read reads write writes
flush flushes sync synchronize barrier barriers reduce reduction reduce_scatter allreduce
broadcast allgather allgatherv scatter scatter gather recv receive receives send sends
sender receiver src dst op ops net network socket sockets packet packets request requests
response responses context contexts group groups collective collectives head tail heads tails
offset offsets size sizes sizes count counts index indexes type types field fields member members
struct structs class classes enum enums union template templates namespace namespaces const static
volatile inline virtual abstract public private protected int uint uint32 uint64 uint32_t uint64_t
int32 int64 int32_t int64_t size_t ssize_t bool true false null nullptr void char float double
elt eltype pack packs packsize multimem lsa mnnvl symptr ncclsymptr devcomm team teams present
absent static_assert assert coverity doxygen ptx sass ld st red atomic spinlock mutex locks lock
spin poll polls busy idle ready pending done complete completes finished finish start started stop
init initialize initialized setup teardown configure configured register unregister map unmap
encode decode compress estimate compute calculate verify validate check checks test debug profile
trace log logs print dump report error errors exception abort fail fails failed success succeed
succeeded return returns returned call calls called invoke callback trigger wait release acquire
create creates created destroy destroys connect connection connections disconnect message messages
payload credits credit semaphore etc api que queue queues
ncclgin gin ginoutbox gininbox session sessions scratch wins window
op128 op64 op128 ncclginoutboxsession ncclgininboxa2asession
pow2up ngroup ngroup pow2 round delta xor
timing timings
msa mlopart
hwlatencies baselatencies llmaxbws perChMaxRingLL128Bws perChMaxTreeLL128Bws perChMaxTreeBws
perChMaxNVLSTreeBws nBytesInOut hostStreamPlanTask ncclAddWorkBatchToPlan
idivFast idivRcp alignof instantiation
headRank nHeads nChannels channelHi channelLo refCount
NCCL_CHECK_MODE DEBUG_LOCAL DEBUG_GLOBAL NCCL_NO_CACHE
NCCL_PROTO_PROTO NCCL_PROTO ncclSpace ncclShadowPool isFull ncuts
makeOptions auto opts
""".split())

# common english function words / fragments that indicate a BROKEN translation
BROKEN_WORDS = set("""
hasn't haven't hadn't must mustn't should shouldn't can't cannot could couldn't won't
wouldn't aren't isn't wasn't weren't ain't their they're theirs we're you're you've we've
it's that's there's what's they've i'm i've he's she's let's lets
the a an all each every any none both either neither some many few most
first last next previous this that these those which what who where when why how
we you our your my its it they he she i us them me
is are was were be been am has have had having do does did doing
will would shall can may might could should
to from into onto by with without within on in at of for as and or but nor
not no yes if unless whenever because since due so therefore thus then than however moreover otherwise
before after during while between among over under above below here there about regarding concerning
need needs needed want try tried use uses using make makes made get gets got set puts put take taken
give gave keep kept hold held find found see saw show shown know known say said think thought consider
assume assumed expect hoped wish ensure ensured allow allowed prevent avoid cause caused
participate participate yet already still also just only even odd
connector hasn yet
""".split())

tok_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]{1,}')

exts = ['*.cc','*.h','*.hpp','*.c','*.cu']
files = []
for e in exts:
    files += glob.glob(f'src/**/{e}', recursive=True)

broken = []
for f in files:
    try:
        lines = open(f, encoding='utf-8', errors='replace').readlines()
    except:
        continue
    for i, ln in enumerate(lines,1):
        s = ln.lstrip()
        if not s.startswith('//'):
            continue
        if not has_cjk.search(ln):
            continue
        # tokenize english words
        words = tok_re.findall(ln)
        bad = [w for w in words if w.lower() not in WHITE and w.lower() in BROKEN_WORDS]
        # also flag lines where english fragment looks like a leftover clause:
        # e.g. "a window is 已创建" -> 'a','window'? window whitelisted. 'a' broken.
        if bad:
            broken.append(f"{f}:{i}: {ln.rstrip()}")

with open('_broken2.txt','w',encoding='utf-8') as o:
    o.write('\n'.join(broken))
print("genuinely broken mixed lines:", len(broken))
