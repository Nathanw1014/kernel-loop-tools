#!/usr/bin/env python3
# cpusamp.py <out.tsv>: every 100 ms, per-CPU scaling_cur_freq (kHz) and busy fraction from /proc/stat, until killed.
import sys,time,glob
cpus=sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq'),key=lambda p:int(p.split('/')[5][3:]))
def stat():
    r={}
    for l in open('/proc/stat'):
        if l.startswith('cpu') and l[3].isdigit():
            f=l.split(); v=list(map(int,f[1:])); r[f[0]]=(sum(v)-v[3]-v[4],sum(v))
    return r
o=open(sys.argv[1],'w'); prev=stat()
while True:
    time.sleep(0.1); t=time.time(); s=stat()
    fr=[open(p).read().strip() for p in cpus]
    bz=[]
    for k in sorted(s,key=lambda x:int(x[3:])):
        db=s[k][0]-prev[k][0]; dt=s[k][1]-prev[k][1]; bz.append(f'{db/dt:.2f}' if dt else '0')
    prev=s; o.write(f'{t:.3f}\t'+','.join(fr)+'\t'+','.join(bz)+'\n'); o.flush()
