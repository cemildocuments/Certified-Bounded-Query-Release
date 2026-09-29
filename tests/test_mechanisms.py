import sys,math,unittest,gzip,json,copy,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from bounded_queries import states,edges,scalar_kernel,workload
from baselines import cpm_regions,hex_covariance
from evaluate import lattice_cov
from anchor import rational_anchor
from certify import verify_certificate

class MechanismTests(unittest.TestCase):
 def test_replacement_graph(self):
  for r in [2,3]:
   X=states(4,r);H=np.column_stack([X,4-X.sum(1)])
   wanted={(i,j) for i in range(len(H)) for j in range(len(H)) if np.abs(H[i]-H[j]).sum()==2}
   self.assertEqual(wanted,set(map(tuple,edges(X))))
 def test_unbiased_geometric(self):
  for n in [2,8]:
   for a in [.2,.7,.95]:
    K,z=scalar_kernel(n,a);np.testing.assert_allclose(K.sum(1),1,atol=1e-13);np.testing.assert_allclose(K@z,np.arange(n+1),atol=1e-12)
 def test_cpm_moments(self):
  for n in [2,8]:
   for e in [.125,.5,2,4]:
    length,p,mu,m2,D=cpm_regions(n,e);self.assertTrue(np.all(length>0));self.assertTrue(np.all(p>0));self.assertAlmostEqual(p.sum(),1,places=9);np.testing.assert_allclose(mu@p,2*np.arange(n+1)/n-1,atol=1e-8);self.assertTrue(np.all(m2@p-(mu@p)**2>=0))
 def test_hexagonal_shells(self):
  for k in range(1,8):
   shell=np.array([(x,y) for x in range(-k,k+1) for y in range(-k,k+1) if max(abs(x),abs(y),abs(x+y))==k])
   self.assertEqual(len(shell),6*k);self.assertEqual(sum(shell[:,0]**2),(10*k**3+2*k)//3)
  for eps in [.5,2,4]:
   P=np.array([(x,y) for x in range(-100,101) for y in range(-100,101)]);w=np.exp(-eps*np.max(np.column_stack([abs(P),abs(P.sum(1))]),axis=1));w/=w.sum();np.testing.assert_allclose(P.T@(P*w[:,None]),lattice_cov(eps),atol=1e-10)
 def test_exact_anchor(self):
  Z,zd,P,D,ids=rational_anchor(2,2,700000,1.)
  for x,row in zip(states(2),P):
   self.assertEqual(sum(row),D);self.assertTrue(min(row)>0)
   for j in range(2):self.assertEqual(sum(row[k]*Z[k,j] for k in range(len(Z))),x[j]*zd*D)
 def test_certificate_rejects_corruption(self):
  source=ROOT/'results/core_n4_e1_balanced.certificate.json.gz'
  with gzip.open(source,'rt') as f: original=json.load(f)
  variants=[]
  bad=copy.deepcopy(original);bad['probability_numerators'].pop();variants.append(bad)
  bad=copy.deepcopy(original);bad['edges'].pop();variants.append(bad)
  bad=copy.deepcopy(original);bad['epsilon_requested']=.5;variants.append(bad)
  bad=copy.deepcopy(original);bad['probability_numerators'][0][0]=str(int(bad['probability_numerators'][0][0])+1);variants.append(bad)
  with tempfile.TemporaryDirectory() as temp:
   for k,bad in enumerate(variants):
    target=Path(temp)/f'bad{k}.json.gz'
    with gzip.open(target,'wt') as f:json.dump(bad,f)
    with self.assertRaises(AssertionError):verify_certificate(target)
 def test_workload_dependencies(self):
  W=workload();np.testing.assert_equal(W[0]+W[1],W[3]);np.testing.assert_equal(W[2]+W[3],0)

if __name__=='__main__':unittest.main()
