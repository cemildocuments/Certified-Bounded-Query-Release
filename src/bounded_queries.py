"""Finite-domain bounded unbiased query mechanisms and exact feasibility checks."""
from __future__ import annotations
import itertools, math, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def states(n: int, r: int = 2):
    return np.asarray([x for x in itertools.product(range(n+1), repeat=r) if sum(x)<=n],dtype=int)

def edges(X, box=False):
    r=X.shape[1]; index={tuple(x):i for i,x in enumerate(X)}
    shifts=list(itertools.product([-1,0,1],repeat=r)) if box else [tuple(s*(np.arange(r)==j).astype(int)) for j in range(r) for s in [-1,1]] + [tuple((np.arange(r)==j).astype(int)-(np.arange(r)==k).astype(int)) for j in range(r) for k in range(r) if j!=k]
    pairs=set()
    for i,x in enumerate(X):
        for d in shifts:
            j=index.get(tuple(x+np.asarray(d)))
            if j is not None and i!=j:pairs.add((i,j))
    return np.asarray(sorted(pairs),dtype=int)

def scalar_kernel(n,a):
    x=np.arange(n+1)[:,None]; k=np.arange(n+1)[None,:]
    K=(1-a)/(1+a)*a**np.abs(x-k)
    K[:,0]=a**x[:,0]/(1+a); K[:,-1]=a**(n-x[:,0])/(1+a)
    z=np.arange(n+1,dtype=float); z[0]=-a/(1-a); z[-1]=n+a/(1-a)
    return K,z

def support_anchor(n,r,epsilon,margin=0.01,expansion=1.0):
    # Rational a (stored as integer / 10^6) gives reproducible output locations.
    a_num=int(math.ceil(max(math.exp(-max(epsilon-margin,epsilon*.9)/2),.5)*1_000_000))
    a=a_num/1_000_000
    if expansion!=1:
        b=expansion*a/(1-a); a_num=int(math.ceil(b/(1+b)*1_000_000));a=a_num/1_000_000
    from anchor import rational_anchor
    Znum,zd,nums,den,indices=rational_anchor(n,r,a_num,epsilon)
    Z=np.array([[int(v)/zd for v in row] for row in Znum])
    P0=np.array([[int(v)/den for v in row] for row in nums])
    return Z,P0,a_num,indices

def workload(r=2, kind='balanced'):
    if r==2:
        if kind=='balanced':return np.array([[1,0],[0,1],[-1,-1],[1,1]],float)
        if kind=='skew':return np.array([[3,0],[0,1],[1,1],[1,-1]],float)
        if kind=='coordinate':return np.eye(2)
        if kind=='contrast':return np.array([[1,-1],[1,1]],float)
    return np.vstack([np.eye(r),-np.ones((1,r)),np.cumsum(np.eye(r),axis=0)])

def risks(X,Z,P,W):
    errors=(Z[None,:,:]-X[:,None,:])@W.T
    C=np.mean(errors**2,axis=2)
    return np.sum(P*C,axis=1),C

def solve(n=4,epsilon=1.,r=2,kind='balanced',box=False,expansion=1., time_limit=1800):
    t=time.perf_counter();X=states(n,r);E=edges(X,box); W=workload(r,kind)
    Z,P0,a_num,indices=support_anchor(n,r,epsilon,expansion=expansion)
    S,M=len(X),len(Z);_,C=risks(X,Z,P0,W)
    # Privacy constraints in both directions for every actual record replacement.
    row=np.repeat(np.arange(len(E)*M),2)
    col=np.column_stack([(E[:,0,None]*M+np.arange(M)).ravel(),(E[:,1,None]*M+np.arange(M)).ravel()]).ravel()
    R=math.exp(epsilon-1e-7)
    vals=np.tile([1.,-R],len(E)*M)
    privacy=sparse.coo_matrix((vals,(row,col)),shape=(len(E)*M,S*M+1)).tocsr()
    risk=sparse.hstack([sparse.block_diag([c[None,:] for c in C]),-np.ones((S,1))],format='csr')
    Aub=sparse.vstack([privacy,risk],format='csr'); bub=np.zeros(Aub.shape[0])
    moments=np.vstack([np.ones(M),Z.T])
    Aeq=sparse.hstack([sparse.kron(sparse.eye(S),moments),sparse.csr_matrix((S*(r+1),1))],format='csr')
    beq=np.column_stack([np.ones(S),X]).ravel()
    obj=np.zeros(S*M+1);obj[-1]=1
    res=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=(0,None),method='highs',options={'time_limit':time_limit,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':1})
    if not res.success:raise RuntimeError(str(res.message))
    P=res.x[:-1].reshape(S,M)
    rr,_=risks(X,Z,P,W);ar,_=risks(X,Z,P0,W)
    # Tight-budget independent bounded geometric baseline.
    K,z=scalar_kernel(n,math.exp(-epsilon/2)); gi=np.asarray(list(itertools.product(range(n+1),repeat=r))); Zg=z[gi];Pg=np.ones((len(X),len(gi)))
    for j in range(r):Pg*=K[X[:,j,None],gi[None,:,j]]
    gr,_=risks(X,Zg,Pg,W)
    metrics={'n':n,'epsilon':epsilon,'r':r,'workload':kind,'box':box,'expansion':expansion,'states':S,'outputs':M,'directed_edges':len(E),'variables':S*M+1,'constraints':Aub.shape[0]+Aeq.shape[0],'seconds':time.perf_counter()-t,'mse_max':float(rr.max()),'mse_mean':float(rr.mean()),'anchor_max':float(ar.max()),'geometric_max':float(gr.max()),'geometric_mean':float(gr.mean()),'moment_error':float(np.abs(P@moments.T-np.column_stack([np.ones(S),X])).max()),'privacy_violation':float(np.max(P[E[:,0]]-math.exp(epsilon)*P[E[:,1]])),'a_num':a_num,'iterations':res.nit}
    return metrics,dict(X=X,Z=Z,P=P,P0=P0,E=E,W=W,risks=rr,geometric_risks=gr,indices=indices,dual_privacy=-res.ineqlin.marginals[:len(E)*M].reshape(len(E),M),dual_risk=-res.ineqlin.marginals[len(E)*M:],dual_moment=res.eqlin.marginals.reshape(S,r+1)[:,1:])

if __name__=='__main__':
    import json,sys
    n=int(sys.argv[1]) if len(sys.argv)>1 else 4
    eps=float(sys.argv[2]) if len(sys.argv)>2 else 1.
    m,a=solve(n,eps)
    print(json.dumps(m,indent=2))
    np.savez_compressed(f'research/pilot_n{n}_e{eps}.npz',**a)
