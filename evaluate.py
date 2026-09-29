"""Compare analytical risks, verify certificates, and evaluate public cohorts."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,math,time,zipfile,io,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from bounded_queries import workload
from baselines import hist_baselines,hex_covariance,staircase_factor,matrix_factor,enhanced_cpm
from certify import verify_certificate
warnings.filterwarnings('ignore',category=OptimizeWarning)

def lattice_cov(eps):
 a=math.exp(-eps);s1=a/(1-a)**2;s3=a*(1+4*a+a*a)/(1-a)**4
 v=((10*s3+2*s1)/3)/(1+6*s1)
 return np.array([[v,-v/2],[-v/2,v]])

def evaluate(f):
 m=json.loads(f.read_text());A=np.load(f.with_suffix('.npz'));W=A['W'];n=m['n'];eps=m['epsilon'];r=W.shape[1]
 risks=hist_baselines(n,eps,W);risks['Certified joint']=A['risks'];extra={}
 if r==2:
  shape=float(np.trace(W@hex_covariance()@W.T)/len(W));fac,g=staircase_factor(eps)
  risks['K-norm']=np.full(len(A['X']),shape*12/eps**2)
  risks['K-staircase']=np.full(len(A['X']),shape*fac)
  risks['Hexagonal geometric']=np.full(len(A['X']),np.trace(W@lattice_cov(eps)@W.T)/len(W))
  kind=m.get('workload','balanced');fac,mat=matrix_factor(kind)
  risks['Optimized matrix Laplace']=np.full(len(A['X']),fac/eps**2)
  cpm=enhanced_cpm(n,eps,W);risks['Joint CPM family']=cpm.pop('risk');extra['cpm']=cpm
  extra['matrix_strategy']=mat;extra['staircase_gamma']=g
 rows=[{'id':m['id'],'n':n,'r':r,'epsilon':eps,'workload':m.get('workload','balanced'),'method':k,'worst_mse':float(v.max()),'uniform_mse':float(v.mean())} for k,v in risks.items()]
 return rows,{'id':m['id'],'risks':{k:v.tolist() for k,v in risks.items()},'extra':extra}

def public_labels():
 with zipfile.ZipFile(ROOT/'data/adult.zip') as z:
  rows=[]
  for name in ['adult.data','adult.test']:
   d=pd.read_csv(z.open(name),header=None,skipinitialspace=True,comment='|');rows.append(d)
  d=pd.concat(rows,ignore_index=True).dropna(how='all');v=d.iloc[:,4].astype(int).to_numpy();adult=np.select([v<=9,v<=12],[0,1],2)
 with zipfile.ZipFile(ROOT/'data/bank_marketing.zip') as z:
  with zipfile.ZipFile(io.BytesIO(z.read('bank.zip'))) as z2:
   d=pd.read_csv(z2.open('bank-full.csv'),sep=';');bank=d['marital'].map({'married':0,'single':1,'divorced':2}).to_numpy()
 return {'Adult':adult,'Bank Marketing':bank}

if __name__=='__main__':
 start=time.time();files=sorted((ROOT/'results').glob('core_*.json'))+sorted((ROOT/'results').glob('dim3_*.json'))
 with ProcessPoolExecutor(max_workers=8) as pool:out=list(pool.map(evaluate,files))
 pd.DataFrame([v for a,b in out for v in a]).to_csv(ROOT/'results/comparisons.csv',index=False)
 curves={b['id']:b for a,b in out};(ROOT/'results/baseline_details.json').write_text(json.dumps(curves,indent=2))
 rows=[];datasets={}
 for name,y in public_labels().items():
  n=8;rng=np.random.default_rng(20260928);v=rng.permutation(y);keep=len(y)//n*n;coh=v[:keep].reshape(-1,n);X=np.column_stack([(coh==j).sum(1) for j in range(2)])
  datasets[name]={'records':len(y),'cohorts':len(X),'discarded_remainder':len(y)-keep,'category_counts':np.bincount(y).tolist(),'seed':20260928}
  for eps in [.25,.5,1,2,4]:
   for kind in ['balanced','skew']:
    key=f'core_n8_e{eps:g}_{kind}';A=np.load(ROOT/'results'/f'{key}.npz');index={tuple(v):i for i,v in enumerate(A['X'])};ix=np.array([index[tuple(v)] for v in X])
    for method,risk in curves[key]['risks'].items():
     risk=np.array(risk);rows.append({'dataset':name,'epsilon':eps,'workload':kind,'method':method,'mean_mse':float(risk[ix].mean()),'cohorts':len(X)})
 pd.DataFrame(rows).to_csv(ROOT/'results/public_cohorts.csv',index=False)
 (ROOT/'results/dataset_summary.json').write_text(json.dumps(datasets,indent=2))
 checks=[]
 for f in sorted((ROOT/'results').glob('*.certificate.json.gz')):checks.append({'file':f.name,'verified':verify_certificate(f)})
 (ROOT/'results/certificate_verification.json').write_text(json.dumps(checks,indent=2))
 (ROOT/'results/evaluation_run.json').write_text(json.dumps({'seconds':time.time()-start,'mechanisms':len(files),'certificates':len(checks)},indent=2))
 print('Evaluation complete',len(files),len(checks),time.time()-start,flush=True)
