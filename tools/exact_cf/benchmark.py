#!/usr/bin/env python3
"""Counter-free CF/ARS performance comparison with rotating trial order.

Reports the biased a930583 CF baseline separately; it is not an exact sampler.
Setup is included in cold batches, excluded from warm batches. Standard Python,
Git and Clang only. Counters are collected using a separate diagnostic build.
"""
import argparse,ctypes as C,csv,json,os,platform,statistics,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=9);a=p.parse_args()
w=a.output.resolve();w.mkdir(parents=True,exist_ok=True)
current=(R/'src/c/cb_core.c').read_text();old=subprocess.check_output(['git','show','a930583:src/c/cb_core.c'],cwd=R,text=True)
variants={'exact_cf':(current,[]),'exact_ars':(current,['-DCB_ENABLE_CF=0']),'uncorrected_cf':(old,['-DCF_BASELINE']),
          'exact_cf_public':(current,[]),'exact_ars_public':(current,['-DCB_ENABLE_CF=0'])}
libs={};P=C.POINTER(C.c_double)
for name,(source,flags) in variants.items():
 core=w/(name+'.c');core.write_text(source);probe=w/(name+'_probe.c');probe.write_text((R/'tools/exact_cf/probe.c').read_text().replace('CORE_FILE',str(core)))
 for counts in (False,True):
  so=w/(name+('_counts' if counts else '')+'.so');subprocess.run(['clang','-O3','-shared','-fPIC',*flags,*(['-DCB_ARS_DIAGNOSTICS=1'] if counts else []),'-I',str(R/'src/c'),str(probe),'-lm','-o',str(so)],check=True,capture_output=True)
  lib=C.CDLL(str(so));lib.cf_benchmark.argtypes=[C.c_double,C.c_double,C.c_int,C.c_int,C.c_int,P];libs[name,counts]=lib
rows=[];v=(C.c_double*5)();names=list(variants)
for chi,nu in [(150.5,301),(250,500),(500,1000),(2500,5000),(500000,1e6),(249.5,500),(10000,1e6)]:
 for n in (1,64,100000):
  for cold in (False,True):
   times={name:[] for name in names};reps=max(10,min(10000,1000000//n))
   for trial in range(a.trials):
    order=names[trial%len(names):]+names[:trial%len(names)]
    for name in order:
     path=(6 if cold else 5) if name.endswith('_public') else ((4 if cold else 1) if name=='exact_ars' else (3 if cold else 0))
     assert libs[name,False].cf_benchmark(chi,nu,n,reps,path,v)
     times[name].append(v[0]*1e9/n)
   for name in names:
    path=(6 if cold else 5) if name.endswith('_public') else ((4 if cold else 1) if name=='exact_ars' else (3 if cold else 0))
    assert libs[name,True].cf_benchmark(chi,nu,n,reps,path,v)
    props,evals,squeezes,tails=list(v)[1:]
    assert libs[name,False].cf_benchmark(chi,nu,n,10000,2,v)
    rows.append(dict(chi=chi,nu=nu,n=n,cold=cold,variant=name,ns_per_sample=statistics.median(times[name]),min_ns=min(times[name]),max_ns=max(times[name]),stats_setup_us=v[0]*1e6,proposals_per_sample=props,full_evaluations_per_sample=evals,squeezes_per_sample=squeezes,tails_per_sample=tails))
 print(chi,nu,'done',flush=True)
with (w/'benchmark.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(w/'benchmark.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
(w/'metadata.json').write_text(json.dumps(dict(baseline='a930583 (biased CF)',platform=platform.platform(),compiler='clang -O3',trials=a.trials,source_sha256=hashlib.sha256(current.encode()).hexdigest(),timing='Fresh stream per timed group; warm batches amortize initial ARS build; cold batches replace stats per batch; separately timed stats updates; counter-free timing builds.'),indent=2)+'\n')
for nu in (301,500,1000,5000,1e6):
 rs=[r for r in rows if r['chi']==nu/2 and r['nu']==nu and r['n']==100000 and not r['cold']]
 print(nu,[(r['variant'],round(r['ns_per_sample'],2))for r in rs])
