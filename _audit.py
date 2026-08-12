#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, re, json

has_cjk = re.compile(r'[\u4e00-\u9fff]')
long_en = re.compile(r'[A-Za-z]{5,}')
url_re = re.compile(r'https?://')
cov_re = re.compile(r'coverity\[')
# lines that are almost entirely code (contain typical code tokens) -> skip from "prose" classification
code_tok = re.compile(r'[;{}]\s*$|::\s|\b(NULL|nullptr|return|sizeof|static_cast)\b')
cjk_word = re.compile(r'[\u4e00-\u9fff][A-Za-z]{0,3}')  # chinese adjacent to english

exts = ['*.cc','*.h','*.hpp','*.c','*.cu']
files = []
for e in exts:
    files += glob.glob(f'src/**/{e}', recursive=True)
files += glob.glob('include/**/*.h', recursive=True) if os.path.isdir('include') else []

broken = []   # contain CJK AND a long english word (likely broken translation)
pure_en = []  # pure english comment prose (no cjk), excluding coverity/url/doxygen/code
density = []  # per-file: comment lines, cjk comment lines, english-only comment lines

for f in files:
    try:
        with open(f, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except:
        continue
    cmt=0; cjk_cmt=0; en_only=0
    for i, ln in enumerate(lines,1):
        s=ln.lstrip()
        if not s.startswith('//'):
            # also catch block comments? skip for now
            continue
        cmt+=1
        body=ln.rstrip('\n')
        if has_cjk.search(body):
            cjk_cmt+=1
        if url_re.search(body) or cov_re.search(body):
            continue
        if has_cjk.search(body):
            if long_en.search(body):
                broken.append(f"{f}:{i}: {body.strip()}")
        else:
            # pure english comment line
            t=body.strip()
            if code_tok.search(t):
                continue
            if t.startswith('///') or t.startswith('//!') or t.startswith('//!'):
                continue
            if long_en.search(t):
                pure_en.append(f"{f}:{i}: {t}")
                en_only+=1
    density.append((f, cmt, cjk_cmt, en_only))

with open('_broken.txt','w',encoding='utf-8') as o:
    o.write('\n'.join(broken))
with open('_pureen.txt','w',encoding='utf-8') as o:
    o.write('\n'.join(pure_en))
# density sorted by total comment lines desc
density.sort(key=lambda x:-x[1])
with open('_density.txt','w',encoding='utf-8') as o:
    for f,c,cj,en in density:
        o.write(f"{c:4d} {cj:4d} {en:4d}  {f}\n")

print("files:", len(files))
print("broken mixed lines:", len(broken))
print("pure-english prose lines:", len(pure_en))
print("top commented files:", len(density))
