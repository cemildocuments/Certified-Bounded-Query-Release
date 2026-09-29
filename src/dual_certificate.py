"""Rational lower and upper risk certificates for the selected output grid."""
from fractions import Fraction
from decimal import Decimal,localcontext
from math import lcm
import numpy as np

def bound_certificate(a,payload,eps,bits=40):
 Q=1<<bits;X=a['X'];wr=[[Fraction(str(v)) for v in row] for row in a['W']];wd=lcm(*(v.denominator for row in wr for v in row));W=np.array([[int(v*wd) for v in row] for row in wr],dtype=object);S,M=a['P'].shape;r=X.shape[1]
 Z=np.array([[int(x) for x in row] for row in payload['output_numerators']],dtype=object);zd=int(payload['output_denominator']);cd=len(W)*zd*zd*wd*wd
 delta=Z[None,:,:]-X[:,None,:]*zd
 V=delta@W.T;C=np.sum(V*V,axis=2)
 mu=np.array([max(0,round(float(v)*Q)) for v in a['dual_risk']],dtype=object);ms=int(sum(mu));assert ms>0
 lam=np.array([[max(0,round(float(v)*Q)) for v in row] for row in a['dual_privacy']],dtype=object)
 beta=np.array([[round(float(v)*Q) for v in row] for row in a['dual_moment']],dtype=object)
 with localcontext() as ctx:
  ctx.prec=90;rd=10**30;rn=int(Decimal(str(eps)).exp()*rd)+1
 priv=np.zeros((S,M),dtype=object)
 for i,(u,v) in enumerate(a['E']):priv[u]+=rd*lam[i];priv[v]-=rn*lam[i]
 den=Q*ms*cd*rd
 coeff=mu[:,None]*C*Q*rd+priv*ms*cd-(beta@Z.T)*ms*(cd//zd)*rd
 lb=Fraction(int(sum(min(row) for row in coeff)),den)+Fraction(int(np.sum(beta*X)),Q)
 PN=np.array([[int(v) for v in row] for row in payload['probability_numerators']],dtype=object);pd=int(payload['probability_denominator'])
 ub=Fraction(int(max(np.sum(PN*C,axis=1))),pd*cd)
 assert ub>=lb
 return {'dual_certified':True,'risk_lower_bound':float(lb),'risk_upper_bound':float(ub),'absolute_optimality_gap':float(ub-lb),'relative_optimality_gap':float((ub-lb)/ub),'dual_bits':bits,'risk_lower_fraction':[str(lb.numerator),str(lb.denominator)],'risk_upper_fraction':[str(ub.numerator),str(ub.denominator)]}
