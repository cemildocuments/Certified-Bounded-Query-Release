"""Run the reproducible finite-domain mechanism study."""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,time,platform,warnings,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import OptimizeWarning
sys.path.insert(0,str(Path(__file__).parent/'src'))
from bounded_queries import solve,risks
from certify import certify,save_certificate
warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected')
ROOT=Path(__file__).resolve().parent

def task(job):
    name=job['id'];force=job.get('_force',False);kwargs={k:v for k,v in job.items() if k not in ['id','_force']}
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if not force and (out/f'{name}.json').exists() and json.loads((out/f'{name}.json').read_text()).get('dual_certified'):return {'id':name,'cached':True}
    start=time.perf_counter()
    m,a=solve(**kwargs)
    p,report,payload=certify(a,m['n'],m['a_num'],m['epsilon'])
    a['P_raw']=a['P'];a['P']=p
    rr,_=risks(a['X'],a['Z'],p,a['W']);a['risks']=rr
    from dual_certificate import bound_certificate
    bound=bound_certificate(a,payload,m['epsilon'])
    m.update(bound);payload['optimality']=bound
    payload['adjacency']='box' if kwargs.get('box',False) else 'replacement'
    m.update(report);m['mse_max_certified']=float(rr.max());m['mse_mean_certified']=float(rr.mean())
    save_certificate(payload,out/f'{name}.certificate.json.gz')
    m['id']=name;m['total_seconds']=time.perf_counter()-start
    np.savez_compressed(out/f'{name}.npz',**a)
    (out/f'{name}.json').write_text(json.dumps(m,indent=2)+'\n')
    return {'id':name,'seconds':round(m['total_seconds'],2),'mse':m.get('mse_max_certified',m['mse_max']),'mix':m.get('mix_weight')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/main.json');p.add_argument('--workers',type=int,default=8);p.add_argument('--force',action='store_true');args=p.parse_args()
    jobs=[dict(j,_force=args.force) for j in json.loads((ROOT/args.config).read_text())];started=time.time()
    manifest={'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'numpy':np.__version__,'workers':args.workers,'config_sha256':hashlib.sha256((ROOT/args.config).read_bytes()).hexdigest(),'start_time_unix':started}
    import scipy;manifest['scipy']=scipy.__version__
    (ROOT/'results'/'environment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(task,j):j['id'] for j in jobs}
        for f in as_completed(futures):
            try:print(json.dumps(f.result()),flush=True)
            except Exception as e:
                error={'id':futures[f],'error':repr(e)};errors.append(error);print(json.dumps(error),flush=True)
    (ROOT/'results'/'run_summary.json').write_text(json.dumps({'elapsed_seconds':time.time()-started,'jobs':len(jobs),'errors':errors},indent=2)+'\n')
    if errors:sys.exit(1)
