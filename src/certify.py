"""Exact integer certification and moment-preserving repair of finite mechanisms."""
from __future__ import annotations
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import reduce
from math import gcd
import gzip,json,math,time,bisect,random
import numpy as np

def rational_budget(epsilon):
    with localcontext() as c:
        c.prec=90; scale=10**30
        numerator=int(Decimal(str(epsilon)).exp()*scale)-1
    f=Fraction(numerator,scale)
    return f.numerator,f.denominator

def certify(arrays,n,a_num,epsilon,bits=52):
    t=time.perf_counter();X=arrays['X'];ids=arrays['indices'];P=arrays['P'];E=arrays['E']
    from anchor import rational_anchor
    S,M=P.shape;r=X.shape[1]
    znum,delta,anchor,D0,ids=rational_anchor(n,r,a_num,epsilon)
    L=int(znum[-1,0]-znum[0,0]);A=-int(znum[0,0]);D=1<<bits;Dt=D*L
    nums=np.asarray([[int(round(float(v)*D))*L for v in row] for row in P],dtype=object)
    piv=[0]+[int(np.where(np.all(ids==np.eye(r,dtype=int)[j]*(n+2),axis=1))[0][0]) for j in range(r)]
    other=[k for k in range(M) if k not in piv]
    for i,x in enumerate(X):
        rounded=[int(nums[i,k]//L) for k in other]
        missing=D-sum(rounded); partial=[]
        for j in range(r):
            residual=int(x[j])*D*delta-sum(v*int(znum[k,j]) for v,k in zip(rounded,other))
            value=residual+A*missing;nums[i,piv[j+1]]=value;partial.append(value)
        nums[i,0]=missing*L-sum(partial)
    rn,rd=rational_budget(epsilon);mix=Fraction(0);negative=0;violations=0
    for p,q in zip(nums.flat,anchor.flat):
        if p<0:
            negative+=1;candidate=Fraction(-p*D0,q*Dt-p*D0)
            if candidate>mix:mix=candidate
    for u,v in E:
        for k in range(M):
            violation=rd*nums[u,k]-rn*nums[v,k]
            if violation>0:
                violations+=1;margin=rn*anchor[v,k]-rd*anchor[u,k]
                if margin<=0:raise ValueError('Anchor must have strict privacy slack.')
                candidate=Fraction(violation*D0,violation*D0+margin*Dt)
                if candidate>mix:mix=candidate
    md=1<<64;mn=(mix.numerator*md+mix.denominator-1)//mix.denominator
    if mn==0:md=1
    out=(md-mn)*nums*D0+mn*anchor*Dt;den=md*Dt*D0
    common=reduce(gcd,(int(v) for v in out.flat),den)
    if common>1:out=out//common;den//=common
    for i,x in enumerate(X):
        assert all(v>=0 for v in out[i]);assert sum(out[i])==den
        for j in range(r):assert sum(out[i,k]*znum[k,j] for k in range(M))==int(x[j])*den*delta
    for u,v in E:
        assert all(rd*out[u,k]<=rn*out[v,k] for k in range(M))
    # Bounded moment constraints are checked over integers, without tolerances.
    Pf=np.array([[int(v)/den for v in row] for row in out])
    report={'certified':True,'rounding_bits':bits,'mix_weight':mn/md,'negative_entries_before':negative,'violated_privacy_inequalities_before':violations,'certificate_seconds':time.perf_counter()-t,'probability_denominator_bits':den.bit_length(),'rational_privacy_numerator':str(rn),'rational_privacy_denominator':str(rd),'normalization_exact':True,'moments_exact':True,'privacy_exact':True}
    payload={'n':n,'epsilon_requested':epsilon,'r':r,'states':X.tolist(),'output_numerators':[[str(v) for v in row] for row in znum],'output_denominator':str(delta),'probability_numerators':[[str(v) for v in row] for row in out],'probability_denominator':str(den),'privacy_ratio_numerator':str(rn),'privacy_ratio_denominator':str(rd),'edges':E.tolist(),'report':report}
    return Pf,report,payload

def save_certificate(payload,path):
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(payload,f,separators=(',',':'),sort_keys=True)

def verify_certificate(path):
    with gzip.open(path,'rt') as f:p=json.load(f)
    from bounded_queries import states,edges
    expected=states(p['n'],p['r'])
    assert p['states']==expected.tolist()
    assert p['edges']==edges(expected,p.get('adjacency')=='box').tolist()
    rn_limit,rd_limit=rational_budget(p['epsilon_requested'])
    assert int(p['privacy_ratio_numerator'])*rd_limit<=rn_limit*int(p['privacy_ratio_denominator'])
    X=p['states'];Z=[[int(v) for v in z] for z in p['output_numerators']];zd=int(p['output_denominator'])
    N=[[int(v) for v in row] for row in p['probability_numerators']];d=int(p['probability_denominator']);rn=int(p['privacy_ratio_numerator']);rd=int(p['privacy_ratio_denominator'])
    assert d>0 and rd>0 and rn>0 and zd>0
    assert len(N)==len(X) and len(Z)>0
    assert all(len(row)==len(Z) for row in N)
    assert all(len(row)==p['r'] for row in Z)
    for x,row in zip(X,N):
        assert min(row)>=0 and sum(row)==d
        for j in range(p['r']):assert sum(v*z[j] for v,z in zip(row,Z))==x[j]*d*zd
    for u,v in p['edges']:
        for a,b in zip(N[u],N[v]):assert rd*a<=rn*b
    return True

class ExactSampler:
    """Integer inverse-CDF sampling. Default randomness is the OS random source."""
    def __init__(self,path,seed=None):
        with gzip.open(path,'rt') as f:self.p=json.load(f)
        self.rng=random.SystemRandom() if seed is None else random.Random(seed)
        self.den=int(self.p['probability_denominator']);self.cdf=[]
        for row in self.p['probability_numerators']:
            c=[];s=0
            for v in row:s+=int(v);c.append(s)
            self.cdf.append(c)
    def sample_index(self,state_index):
        return bisect.bisect_right(self.cdf[state_index],self.rng.randrange(self.den))
