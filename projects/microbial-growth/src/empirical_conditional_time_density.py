"""Positive empirical conditional density of time given birth size/growth rate.

Kernel weights use log b and log alpha. Response kernels are normal densities
in g=alpha*t, truncated at zero. This coordinate change is NOT an assumption
that g is independent of alpha: alpha remains a kernel predictor.
"""
from __future__ import annotations
import numpy as np


def normal_sf(z):
    """Normal survival function; vectorized A&S approximation (<8e-8 error)."""
    z=np.asarray(z,dtype=float);x=np.abs(z);q=1/(1+.2316419*x)
    poly=q*(.319381530+q*(-.356563782+q*(1.781477937+q*(-1.821255978+q*1.330274429))))
    small=np.exp(-.5*x*x)/np.sqrt(2*np.pi)*poly
    return np.where(z>=0,small,1-small)


def normal_interval(lo,hi):
    # Use the smaller tail where possible to avoid cancellation near one.
    lo=np.asarray(lo);hi=np.asarray(hi)
    prob=np.where(lo>=0,normal_sf(lo)-normal_sf(hi),normal_sf(-hi)-normal_sf(-lo))
    return np.maximum(prob,0.)


class ConditionalTimeKDE:
    def __init__(self,birth,alpha,time,h_birth,h_alpha,h_growth):
        self.birth=np.asarray(birth,float);self.alpha=np.asarray(alpha,float)
        self.growth=self.alpha*np.asarray(time,float)
        if not (len(self.birth)==len(self.alpha)==len(self.growth)):
            raise ValueError('Training arrays must match')
        if np.any(self.birth<=0) or np.any(self.alpha<=0) or np.any(self.growth<=0):
            raise ValueError('Training values must be positive')
        if min(h_birth,h_alpha,h_growth)<=0:raise ValueError('Bandwidths must be positive')
        self.hb=float(h_birth);self.ha=float(h_alpha);self.hg=float(h_growth)
        self.logb=np.log(self.birth);self.loga=np.log(self.alpha)
        self.normalizer=normal_sf(-self.growth/self.hg)

    def weights(self,birth,alpha):
        b,a=np.broadcast_arrays(np.asarray(birth,float),np.asarray(alpha,float))
        if np.any(b<=0) or np.any(a<=0):raise ValueError('Conditioning values must be positive')
        z=-.5*((np.log(b).reshape(-1,1)-self.logb)/self.hb)**2
        z-=.5*((np.log(a).reshape(-1,1)-self.loga)/self.ha)**2
        z-=np.max(z,axis=1,keepdims=True)
        w=np.exp(z);return w/w.sum(axis=1,keepdims=True)

    def density(self,time,birth,alpha,chunk=128):
        t,b,a=np.broadcast_arrays(np.asarray(time,float),np.asarray(birth,float),np.asarray(alpha,float))
        shape=t.shape;t=t.ravel();b=b.ravel();a=a.ravel();result=np.empty(len(t))
        for i in range(0,len(t),chunk):
            s=slice(i,i+chunk);w=self.weights(b[s],a[s]);z=(a[s,None]*t[s,None]-self.growth)/self.hg
            k=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*self.hg*self.normalizer)
            result[s]=a[s]*np.sum(w*k,axis=1)
        return np.where(t>=0,result,0.).reshape(shape)

    def probability(self,lower,upper,birth,alpha,chunk=128):
        lo,hi,b,a=np.broadcast_arrays(np.asarray(lower,float),np.asarray(upper,float),np.asarray(birth,float),np.asarray(alpha,float))
        shape=lo.shape;lo=lo.ravel();hi=hi.ravel();b=b.ravel();a=a.ravel();out=np.empty(len(lo))
        if np.any(hi<lo):raise ValueError('Upper interval limit is below lower limit')
        for i in range(0,len(lo),chunk):
            s=slice(i,i+chunk);w=self.weights(b[s],a[s])
            zl=(a[s,None]*np.maximum(lo[s,None],0)-self.growth)/self.hg
            zu=(a[s,None]*np.maximum(hi[s,None],0)-self.growth)/self.hg
            out[s]=np.sum(w*normal_interval(zl,zu)/self.normalizer,axis=1)
        return np.clip(out,0,1).reshape(shape)

    def cdf(self,time,birth,alpha):
        return self.probability(np.minimum(np.asarray(time),0.),time,birth,alpha)

    def effective_sample_size(self,birth,alpha):
        w=self.weights(birth,alpha);return 1/np.sum(w*w,axis=1)

    def conditional_mean(self,birth,alpha):
        a=np.broadcast_arrays(np.asarray(birth),np.asarray(alpha))[1].ravel()
        w=self.weights(birth,alpha)
        mean_g=self.growth+self.hg*np.exp(-.5*(self.growth/self.hg)**2)/(np.sqrt(2*np.pi)*self.normalizer)
        return w@mean_g/a

    def sample(self,birth,alpha,rng):
        b,a=np.broadcast_arrays(np.asarray(birth,float),np.asarray(alpha,float))
        w=self.weights(b,a);u=rng.random(len(w));idx=np.sum(np.cumsum(w,axis=1)<u[:,None],axis=1)
        idx=np.minimum(idx,len(self.growth)-1);g=self.growth[idx]+self.hg*rng.normal(size=len(idx))
        bad=g<=0
        while bad.any():
            g[bad]=self.growth[idx[bad]]+self.hg*rng.normal(size=bad.sum());bad=g<=0
        return (g/a.ravel()).reshape(b.shape)
