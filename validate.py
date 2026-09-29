"""Independent exact checks and seeded sampler validation."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,gzip,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from certify import verify_certificate,ExactSampler
from dual_certificate import bound_certificate

def check(f):
 verify_certificate(f)
 with gzip.open(f,'rt') as stream:p=json.load(stream)
 key=f.name.replace('.certificate.json.gz','');a=dict(np.load(ROOT/'results'/f'{key}.npz'));assert a['X'].tolist()==p['states'];assert a['E'].tolist()==p['edges'];b=bound_certificate(a,p,p['epsilon_requested'])
 assert b['risk_lower_fraction']==p['optimality']['risk_lower_fraction']
 assert b['risk_upper_fraction']==p['optimality']['risk_upper_fraction']
 return {'id':key,'verified':True,'relative_gap':b['relative_optimality_gap']}

def sample(job):
 eps,state,seed=job;key=f'core_n8_e{eps:g}_balanced';f=ROOT/'results'/f'{key}.certificate.json.gz';s=ExactSampler(f,seed);a=np.load(ROOT/'results'/f'{key}.npz');idx=np.where(np.all(a['X']==state,axis=1))[0][0];N=100000
 start=time.perf_counter();draw=np.fromiter((s.sample_index(idx) for _ in range(N)),int,N);elapsed=time.perf_counter()-start
 error=(a['Z'][draw]-np.asarray(state))@a['W'].T;loss=np.mean(error**2,axis=1);se=loss.std(ddof=1)/np.sqrt(N)
 return {'epsilon':eps,'state':state,'seed':seed,'draws':N,'empirical_mse':float(loss.mean()),'expected_mse':float(a['risks'][idx]),'standard_error':float(se),'mean_coordinate_error':(a['Z'][draw]-state).mean(0).tolist(),'sampling_seconds':elapsed}

if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=8) as p:checks=list(p.map(check,sorted((ROOT/'results').glob('*.certificate.json.gz'))))
 (ROOT/'results/exact_verification.json').write_text(json.dumps(checks,indent=2))
 jobs=[(e,x,20260928+j) for e in [.5,1,4] for j,x in enumerate([[0,0],[3,2],[8,0]])]
 with ProcessPoolExecutor(max_workers=8) as p:out=list(p.map(sample,jobs))
 (ROOT/'results/sampler_validation.json').write_text(json.dumps(out,indent=2));print('Verified',len(checks),'certificates; sampled',sum(x['draws'] for x in out),'outputs.')
