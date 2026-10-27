"""Run one showcase GPU render and record process wall time and peak host RSS."""
import argparse,subprocess,time,resource,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('view');p.add_argument('output');p.add_argument('--spp',type=int,default=512);p.add_argument('--resolution',default='640x400');p.add_argument('--integrator',default='path');p.add_argument('--photons');p.add_argument('--set',action='append',default=[]);a=p.parse_args()
cmd=['build/owe','render',a.scene,'--backend','gpu','--device','1','--detector',a.view,'--integrator',a.integrator,'--spp',str(a.spp),'--resolution',a.resolution,'--out',a.output]
if a.photons:cmd+=['--photons',a.photons]
for edit in a.set:cmd+=['--set',edit]
Path(a.output).parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
with open(a.output+'.log','w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
record={'command':cmd,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'peak_host_rss_MiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,'texture_budget':'engine default; no override'}
Path(a.output+'.timing.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
raise SystemExit(r.returncode)
