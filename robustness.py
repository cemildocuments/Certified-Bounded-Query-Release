"""Sensitivity checks for the CPM comparison and workload transfer."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.optimize import OptimizeWarning
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from baselines import enhanced_cpm
from bounded_queries import workload,risks
warnings.filterwarnings('ignore',category=OptimizeWarning)
def refine(job):
 eps,kind=job;a=enhanced_cpm(8,eps,workload(2,kind),65);a.pop('risk');return dict(epsilon=eps,workload=kind,**a)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=8) as p:out=list(p.map(refine,[(e,k) for e in [.25,.5,1,2,4] for k in ['balanced','skew']]))
 (ROOT/'results/cpm_refinement.json').write_text(json.dumps(out,indent=2))
 rows=[]
 for e in [.25,.5,1,2,4]:
  for trained in ['balanced','skew']:
   a=np.load(ROOT/'results'/f'core_n8_e{e:g}_{trained}.npz')
   for target in ['balanced','skew']:
    rr,_=risks(a['X'],a['Z'],a['P'],workload(2,target));rows.append(dict(epsilon=e,trained=trained,target=target,mse_max=float(rr.max()),mse_mean=float(rr.mean())))
 pd.DataFrame(rows).to_csv(ROOT/'results/workload_transfer.csv',index=False)
 print('Robustness checks complete')
