"""Rational strictly private anchors on an expanded, endpoint-complete count grid."""
from fractions import Fraction as F
from math import lcm,prod,exp,ceil
import itertools
import numpy as np

def rational_anchor(n,r,a_num,epsilon):
    outer=F(int(a_num),1_000_000); b=outer/(1-outer)
    lo=int(ceil(exp(-epsilon/2)*1_000_000))
    inside=F((lo+int(a_num))//2,1_000_000)
    if not inside<outer:raise ValueError('A strictly interior anchor is needed.')
    bi=inside/(1-inside)
    z=[-b]+[F(k) for k in range(n+1)]+[F(n)+b]
    old=[-bi]+[F(k) for k in range(1,n)]+[F(n)+bi]
    tau=(b-bi)/(2*(F(n)+2*b))
    T=[]
    for v in old:
        row=[tau/len(z)]*len(z)
        adjusted=(v-tau*F(n,2))/(1-tau)
        j=next(j for j in range(len(z)-1) if z[j]<=adjusted<=z[j+1])
        right=(adjusted-z[j])/(z[j+1]-z[j]);row[j]+=(1-tau)*(1-right);row[j+1]+=(1-tau)*right
        assert sum(row)==1 and sum(p*x for p,x in zip(row,z))==v
        T.append(row)
    K=[]
    for x in range(n+1):
        row=[inside**x/(1+inside)]+[(1-inside)/(1+inside)*inside**abs(x-k) for k in range(1,n)]+[inside**(n-x)/(1+inside)]
        K.append([sum(row[k]*T[k][j] for k in range(n+1)) for j in range(n+3)])
        assert sum(K[-1])==1 and sum(p*v for p,v in zip(K[-1],z))==x
    kd=lcm(*(p.denominator for row in K for p in row))
    ki=[[p.numerator*(kd//p.denominator) for p in row] for row in K]
    ids=np.asarray(list(itertools.product(range(n+3),repeat=r)),dtype=int)
    X=np.asarray([x for x in itertools.product(range(n+1),repeat=r) if sum(x)<=n],dtype=int)
    zd=lcm(*(p.denominator for p in z));zn=[p.numerator*(zd//p.denominator) for p in z]
    Znum=np.array([[zn[k] for k in row] for row in ids],dtype=object)
    N=np.array([[prod(ki[int(x[j])][int(v[j])] for j in range(r)) for v in ids] for x in X],dtype=object)
    return Znum,zd,N,kd**r,ids
