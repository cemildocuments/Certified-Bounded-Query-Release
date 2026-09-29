"""Analytical baselines for fixed-size categorical histogram releases."""
import math,itertools
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize,minimize_scalar,linprog
from scipy import sparse
from bounded_queries import scalar_kernel,states,workload

@lru_cache(None)
def cpm_regions(n,eps,b_fraction=None):
    # Central piecewise mechanism, equations in Yamamoto and Shibuya (2026).
    m=n-1; delta=2/n; e=math.exp(eps)
    A=delta*e*math.expm1(m*eps)/math.expm1(eps)
    B=(1+m*delta)*math.exp((m+1)*eps)
    U=eps/math.log((A+math.sqrt(A*A+4*B))/2)
    b=U*((m+3)/(m+4) if b_fraction is None else b_fraction)
    eb=math.exp(eps/b);top=math.exp((m+1)*eps)
    q=eb/(2*(eb+1)**2*(e-1))*((e-1)*(top-1)-delta*e/eb*(m*top-(m+1)*math.exp(m*eps)+1))
    shift=top/(2*q*(eb+1))
    radii=np.array([eb/(2*q*(eb+1))*(1+delta*e/eb*((m+1-j)*math.exp(m*eps)-math.expm1(m*eps)/math.expm1(eps))) for j in range(1,m+2)])
    D=shift+radii[0]
    lengths=np.r_[2*shift,2*(radii[:-1]-radii[1:]),2*radii[-1]]
    density=q*np.exp(-eps*np.arange(n,-1,-1));mass=lengths*density
    assert abs(mass.sum()-1)<1e-7
    means=[];seconds=[]
    for x in range(n+1):
        t=2*x/n-1;c=shift*t
        first=[((c-radii[0])**2-D**2+D**2-(c+radii[0])**2)/2/lengths[0]]
        second=[((c-radii[0])**3+D**3+D**3-(c+radii[0])**3)/3/lengths[0]]
        for j in range(n-1):
            inner,outer=radii[j+1],radii[j]
            first.append(c);second.append(c*c+(outer**3-inner**3)/(3*(outer-inner)))
        first.append(c);second.append(c*c+radii[-1]**2/3)
        means.append(first);seconds.append(second)
    means=np.asarray(means);seconds=np.asarray(seconds)
    target=np.arange(n+1)*2/n-1
    assert np.max(np.abs(means@mass-target))<1e-7
    return lengths,mass,means,seconds,D

def cpm_variance(n,eps):
    _,p,_,s,_=cpm_regions(n,eps)
    t=2*np.arange(n+1)/n-1
    return (n*n/4)*(s@p-t*t)

def projection_map(W,r):
    # W acts on the first r coordinates; the total n is public.
    A=np.column_stack([W,np.zeros(W.shape[0])]);H=np.eye(r+1)-np.ones((r+1,r+1))/(r+1)
    return A@H

def hist_baselines(n,eps,W):
    r=W.shape[1];X=states(n,r);Hfull=np.column_stack([X,n-X.sum(1)])
    a=math.exp(-eps/2);K,z=scalar_kernel(n,a)
    vg=K@(z*z)-np.arange(n+1)**2
    B=projection_map(W,r);coefs=(B*B).sum(0)/len(W)
    rawcoefs=(W*W).sum(0)/len(W)
    out={
      'Bounded geometric + projection':(vg[Hfull]*coefs).sum(1),
      'Geometric + projection':np.full(len(X),coefs.sum()*2*a/(1-a)**2),
      'Laplace + projection':np.full(len(X),coefs.sum()*8/eps**2),
      'Independent bounded geometric':(vg[X]*rawcoefs).sum(1),
      'CPM + projection':(cpm_variance(n,eps/2)[Hfull]*coefs).sum(1),
      'Independent CPM':(cpm_variance(n,eps/2)[X]*rawcoefs).sum(1)
    }
    return out

@lru_cache(None)
def staircase_factor(eps):
    a=math.exp(-eps);k=np.arange(max(80,int(math.ceil(50/eps))),dtype=float);w=a**k
    def integral(p,g):return np.sum(w*(((k+g)**(p+1)-k**(p+1))+a*((k+1)**(p+1)-(k+g)**(p+1)))/(p+1))
    def obj(g):return 2*integral(3,g)/integral(1,g)
    o=minimize_scalar(obj,bounds=(0,1),method='bounded',options={'xatol':1e-13})
    return float(o.fun),float(o.x)

def hex_covariance():
    V=np.array([[1,0],[0,1],[-1,1],[-1,0],[0,-1],[1,-1]],float)
    C=np.zeros((2,2));area=0
    for a,b in zip(V,np.roll(V,-1,axis=0)):
        w=abs(a[0]*b[1]-a[1]*b[0])/2;area+=w
        C+=w*((np.outer(a,a)+np.outer(b,b))/6+(np.outer(a,b)+np.outer(b,a))/12)
    return C/area

@lru_cache(None)
def matrix_factor(kind):
    W=workload(2,kind);deltas=np.array([[1,0],[0,1],[1,-1]],float)
    def loss(flat):
        A=np.asarray(flat).reshape(-1,2)
        if np.linalg.matrix_rank(A)<2:return 1e12
        sensitivity=np.max(np.sum(np.abs(deltas@A.T),axis=1))
        return 2*sensitivity*sensitivity*np.sum((W@np.linalg.pinv(A))**2)/len(W)
    starts=[np.array([[1,0],[0,1],[-1,-1.]]),np.vstack([np.eye(2),[1,-1]]),np.array([[1,0],[0,1],[.5,.5]])]
    best=min((minimize(loss,A.ravel(),method='Powell',options={'maxiter':1500,'xtol':1e-9,'ftol':1e-10}) for A in starts),key=lambda z:z.fun)
    return float(best.fun),best.x.reshape(-1,2).tolist()

def enhanced_cpm(n,eps,W,steps=17):
    X=states(n,2);t=2*X/n-1;G=W.T@W/len(W);best=None
    levels=np.array(list(itertools.product(range(n+1),repeat=2)));M=len(levels)
    index={tuple(v):i for i,v in enumerate(levels)}
    # Optimizes risk within the published monotone joint-density family.
    for scalar_eps in np.linspace(eps/2,eps,steps):
        length,marginal,means,seconds,D=cpm_regions(n,float(scalar_eps))
        volume=np.prod(length[levels],axis=1);sc=volume/np.max(volume)
        equal=[];rhs=[]
        for axis in range(2):
            for j in range(n+1 if axis==0 else n):
                equal.append((levels[:,axis]==j)*sc);rhs.append(marginal[j])
        rr=[];cc=[];vv=[];rows=0
        for i,v in enumerate(levels):
            for j in range(2):
                if v[j]<n:
                    nxt=v.copy();nxt[j]+=1
                    rr.extend([rows,rows]);cc.extend([i,index[tuple(nxt)]]);vv.extend([1.,-1.]);rows+=1
            if np.all(v<n):
                rr.extend([rows,rows]);cc.extend([index[tuple(v+1)],i]);vv.extend([1.,-math.exp(eps)]);rows+=1
        A=sparse.coo_matrix((vv,(rr,cc)),shape=(rows,M+1)).tocsr()
        costs=[]
        for x,tt in zip(X,t):
            first=means[x[0],levels[:,0]];second=means[x[1],levels[:,1]]
            raw=G[0,0]*seconds[x[0],levels[:,0]]+G[1,1]*seconds[x[1],levels[:,1]]+2*G[0,1]*first*second
            costs.append(n*n/4*(raw-tt@G@tt))
        costs=np.asarray(costs)
        Arisk=sparse.hstack([sparse.csr_matrix(costs*sc),-np.ones((len(X),1))],format='csr')
        obj=np.r_[np.zeros(M),1.]
        res=linprog(obj,A_ub=sparse.vstack([A,Arisk]),b_ub=np.zeros(rows+len(X)),A_eq=np.column_stack([equal,np.zeros(len(equal))]),b_eq=rhs,bounds=(0,None),method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if res.success:
            prob=res.x[:-1]*sc
            if np.max(np.abs(np.array(equal)@res.x[:-1]-rhs))>1e-7:continue
            risk=costs@prob
            if best is None or max(risk)<best['max']:
                best={'max':float(max(risk)),'risk':risk,'scalar_epsilon':float(scalar_eps),'range_radius':float(n*D/2),'steps':steps}
    if best is None:raise RuntimeError('Enhanced CPM feasibility failed')
    return best

if __name__=='__main__':
    import json
    for eps in [.25,.5,1,2,4]:
        n=8;W=workload();out=hist_baselines(n,eps,W)
        factor,g=staircase_factor(eps);out['K-norm']=np.array([np.trace(W@hex_covariance()@W.T)/len(W)*12/eps**2]);out['General staircase']=np.array([np.trace(W@hex_covariance()@W.T)/len(W)*factor])
        ec=enhanced_cpm(n,eps,W)
        print(eps,{k:round(float(max(v)),6) for k,v in out.items()},'Enhanced CPM',ec['max'],ec['scalar_epsilon'],flush=True)
