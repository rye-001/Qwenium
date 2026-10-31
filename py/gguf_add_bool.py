#!/usr/bin/env python3
"""Copy a GGUF v3 file adding one bool KV = true (tensor infos and data unchanged).

Used for the tanh-GELU gemma4v reference (tests/fixtures/embed_image/README.md):
    python3 py/gguf_add_bool.py models/embeddinggemma-2-mmproj-BF16.gguf /tmp/mmproj-gelu.gguf clip.use_gelu
"""
import struct, sys, shutil
src, dst, key = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(src, 'rb'); buf = f.read(1 << 20); pos = 0
def rd(fmt):
    global pos
    v = struct.unpack_from('<' + fmt, buf, pos); pos += struct.calcsize('<' + fmt); return v[0]
def rstr():
    global pos
    n = rd('Q'); s = buf[pos:pos + n].decode(); pos += n; return s
SC = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def skip(t):
    global pos
    if t in SC: rd(SC[t])
    elif t == 8: rstr()
    elif t == 9:
        et = rd('I'); n = rd('Q')
        for _ in range(n): skip(et)
    else: raise ValueError(t)
assert buf[:4] == b'GGUF'; pos = 4
ver = rd('I'); nt = rd('Q'); nkv = rd('Q'); assert ver == 3
kv0 = pos; align = 32
for _ in range(nkv):
    k = rstr(); t = rd('I')
    if k == key: raise SystemExit(f'{key} already present')
    if k == 'general.alignment': align = rd('I'); continue
    skip(t)
kv1 = pos
for _ in range(nt):
    rstr(); nd = rd('I'); pos += 8 * nd; rd('I'); rd('Q')
ti1 = pos
data0 = (ti1 + align - 1) // align * align
kb = key.encode()
newkv = struct.pack('<Q', len(kb)) + kb + struct.pack('<I?', 7, True)
head = buf[:4] + struct.pack('<IQQ', ver, nt, nkv + 1) + buf[kv0:kv1] + newkv + buf[kv1:ti1]
head += b'\0' * ((-len(head)) % align)
with open(dst, 'wb') as o:
    o.write(head); f.seek(data0); shutil.copyfileobj(f, o, 1 << 24)
print(f'wrote {dst}: +{key}=true, data moved {data0} -> {len(head)}')
