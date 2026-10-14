"""Positive-growth conditional KDE for p(b_i | b_(i-1), alpha_i, F_i).

Gaussian kernels in log conditioning variables and positive-truncated Gaussian
response kernels in g=log[b_i/(b_(i-1)*F_i)]. Density in birth includes 1/b_i.
Optional older birth and elapsed time add conditioning kernels; elapsed time
is in linear hours, not log hours, so a zero clock is valid.
"""
import numpy as np
from empirical_conditional_time_density import normal_sf


class BirthTransitionKDE:
    def __init__(self,b_previous,alpha_current,factor,b_current,bandwidths,b_older=None,h_older=np.inf,elapsed=None,h_elapsed=np.inf):
        self.logb=np.log(np.asarray(b_previous,float))
        self.loga=np.log(np.asarray(alpha_current,float));self.logf=np.log(np.asarray(factor,float))
        self.g=np.log(np.asarray(b_current,float))-self.logb-self.logf
        self.hb,self.ha,self.hf,self.hg=map(float,bandwidths)
        self.logolder=None if b_older is None else np.log(np.asarray(b_older,float))
        self.holder=float(h_older)
        self.elapsed=None if elapsed is None else np.asarray(elapsed,float)
        self.helapsed=float(h_elapsed)
        if self.helapsed<=0 or (np.isfinite(self.helapsed) and self.elapsed is None):raise ValueError('Elapsed-time data and positive bandwidth required')
        if self.holder<=0 or (np.isfinite(self.holder) and self.logolder is None):raise ValueError('Older-birth data and positive bandwidth required')
        if np.any(self.g<=0) or min(bandwidths)<=0:raise ValueError('Positive growth and bandwidths required')
        self.norm=normal_sf(-self.g/self.hg)
        self.proposals=0;self.draws=0;self.fallback_draws=0

    def weights(self,b_previous,alpha_current,factor,b_older=None,elapsed=None):
        b,a,f=np.broadcast_arrays(b_previous,alpha_current,factor)
        logits=-.5*((np.log(b).reshape(-1,1)-self.logb)/self.hb)**2
        logits-=.5*((np.log(a).reshape(-1,1)-self.loga)/self.ha)**2
        logits-=.5*((np.log(f).reshape(-1,1)-self.logf)/self.hf)**2
        if np.isfinite(self.holder):
            older=np.broadcast_to(b_older,b.shape)
            logits-=.5*((np.log(older).reshape(-1,1)-self.logolder)/self.holder)**2
        if np.isfinite(self.helapsed):
            clock=np.broadcast_to(elapsed,b.shape)
            logits-=.5*((clock.reshape(-1,1)-self.elapsed)/self.helapsed)**2
        logits-=logits.max(1,keepdims=True);w=np.exp(logits)
        return w/w.sum(1,keepdims=True)

    def density(self,b_current,b_previous,alpha_current,factor,chunk=128,b_older=None,elapsed=None):
        outshape=np.broadcast_shapes(np.shape(b_current),np.shape(b_previous),np.shape(alpha_current),np.shape(factor))
        q,b,a,f=[x.ravel() for x in np.broadcast_arrays(b_current,b_previous,alpha_current,factor)]
        out=np.empty(len(q));growth=np.log(q)-np.log(b)-np.log(f)
        older=None if b_older is None else np.broadcast_to(b_older,outshape).ravel()
        clock=None if elapsed is None else np.broadcast_to(elapsed,outshape).ravel()
        for i in range(0,len(q),chunk):
            sl=slice(i,i+chunk);w=self.weights(b[sl],a[sl],f[sl],None if older is None else older[sl],None if clock is None else clock[sl])
            z=(growth[sl,None]-self.g)/self.hg
            kernel=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*self.hg*self.norm)
            out[sl]=np.sum(w*kernel,1)/q[sl]
        return np.where(growth>0,out,0.).reshape(outshape)

    def sample_fixed_context(self,b_previous,alpha_current,factor,rng,b_older=None,elapsed=None):
        """Exact mixture sampling, efficient for many replays of one fixed a/F.

Propose mixture indices using fixed a/F weights, accept using birth, optional
older-birth, and optional elapsed-time weights (which may vary by replay).
After 128 rejection rounds, unresolved draws use the full exact weights.
The fallback changes cost, not the target distribution.
"""
        b=np.asarray(b_previous,float);shape=b.shape;logb=np.log(b.ravel())
        logolder=None if not np.isfinite(self.holder) else np.log(np.broadcast_to(b_older,shape).ravel())
        clock=None if not np.isfinite(self.helapsed) else np.broadcast_to(elapsed,shape).ravel()
        base=-.5*((np.log(alpha_current)-self.loga)/self.ha)**2-.5*((np.log(factor)-self.logf)/self.hf)**2
        weights=np.exp(base-base.max());cdf=np.cumsum(weights);cdf/=cdf[-1]
        indices=np.full(len(logb),-1,dtype=int);pending=np.arange(len(logb))
        for _ in range(128):
            if not len(pending):break
            proposals=np.minimum(np.searchsorted(cdf,rng.random(len(pending))),len(self.g)-1)
            logaccept=-.5*((logb[pending]-self.logb[proposals])/self.hb)**2
            if logolder is not None:logaccept-=.5*((logolder[pending]-self.logolder[proposals])/self.holder)**2
            if clock is not None:logaccept-=.5*((clock[pending]-self.elapsed[proposals])/self.helapsed)**2
            accept=np.log(rng.random(len(pending)))<logaccept
            indices[pending[accept]]=proposals[accept];self.proposals+=len(pending);pending=pending[~accept]
        for j in pending:
            logits=base-.5*((logb[j]-self.logb)/self.hb)**2
            if logolder is not None:logits-=.5*((logolder[j]-self.logolder)/self.holder)**2
            if clock is not None:logits-=.5*((clock[j]-self.elapsed)/self.helapsed)**2
            w=np.exp(logits-logits.max());fullcdf=np.cumsum(w);fullcdf/=fullcdf[-1]
            indices[j]=min(np.searchsorted(fullcdf,rng.random()),len(self.g)-1)
        self.fallback_draws+=len(pending);self.draws+=len(logb)
        growth=self.g[indices]+self.hg*rng.normal(size=len(indices));bad=growth<=0
        while bad.any():
            growth[bad]=self.g[indices[bad]]+self.hg*rng.normal(size=bad.sum());bad=growth<=0
        birth=b.ravel()*factor*np.exp(growth)
        if not np.all(np.isfinite(birth)):raise FloatingPointError('Nonfinite generated birth; no clipping')
        return birth.reshape(shape),growth.reshape(shape)
