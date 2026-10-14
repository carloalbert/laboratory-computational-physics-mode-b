"""Marginal likelihood for a Hill growth hazard with one static normal offset.

Coordinates: (log u, log omega2, log eta, gamma=eta+kappa, log tau_a).
One offset is integrated over the ENTIRE lineage, not separately per cycle.
Growth passed here is G+mean_g, which must be positive.
"""
import numpy as np

BOUNDS=np.array([[np.log(.01),np.log(30.)],[np.log(.01),np.log(200.)],
                 [np.log(.1),np.log(80.)],[-30.,30.],[-8.,np.log(10.)]])


def pack(p,tau=None):
    return np.array([np.log(p['u']),np.log(p['omega2']),np.log(p['eta']),
                     p['eta']+p['kappa'],np.log(p['tau_a'] if tau is None else tau)])


def unpack(z):
    return dict(u=float(np.exp(z[0])),omega2=float(np.exp(z[1])),eta=float(np.exp(z[2])),
                kappa=float(z[3]-np.exp(z[2])),tau_a=float(np.exp(z[4])))


def logsumexp(x,axis=-1):
    m=np.max(x,axis=axis,keepdims=True)
    return np.squeeze(m+np.log(np.sum(np.exp(x-m),axis=axis,keepdims=True)),axis=axis)


class MarginalLikelihood:
    def __init__(self,X,g,hazard_order=32,offset_order=41):
        self.X=np.asarray(X,float);self.g=np.asarray(g,float)
        if self.X.shape!=self.g.shape or self.X.ndim!=2 or np.any(self.g<=0):
            raise ValueError('Expected lineage x cycle arrays, positive uncentered growth')
        self.logp=np.log(np.expm1(self.g))
        nodes,self.weights=np.polynomial.legendre.leggauss(hazard_order)
        self.ilogp=np.log(np.expm1(.5*self.g[...,None]*(nodes+1)))
        an,aw=np.polynomial.hermite.hermgauss(offset_order)
        self.standard_offsets=np.sqrt(2)*an;self.logweights=np.log(aw)-.5*np.log(np.pi)

    def evaluate(self,z,gradient=True,posterior=False):
        logu,logw,logeta,gamma,logtau=z;eta=np.exp(logeta);omega=np.exp(logw);tau=np.exp(logtau)
        offsets=tau*self.standard_offsets;Ls=[];Ds=[]
        # Chunk quadrature nodes to bound memory on a full cohort.
        for start in range(0,len(offsets),12):
            a=offsets[start:start+12]
            odds=eta*(self.logp[:,None,:]-logu)+gamma*self.X[:,None,:]+a[None,:,None]
            iodds=eta*(self.ilogp[:,None,:,:]-logu)+gamma*self.X[:,None,:,None]+a[None,:,None,None]
            H=1/(1+np.exp(np.clip(-iodds,-700,700)))
            integral=.5*self.g[:,None,:]*np.sum(self.weights*H,axis=-1)
            ll=np.sum(logw-np.logaddexp(0.,-odds)-omega*integral,axis=-1)
            Ls.append(ll)
            if gradient:
                e=1/(1+np.exp(np.clip(odds,-700,700)))
                D=self.weights*H*(1-H)
                dI=.5*self.g[:,None,:]*np.sum(D,axis=-1)
                s=e-omega*dI
                deta=eta*((self.logp[:,None,:]-logu)*e-omega*.5*self.g[:,None,:]*np.sum(D*(self.ilogp[:,None,:,:]-logu),axis=-1))
                Ds.append(np.stack([-eta*s.sum(-1),self.g.shape[1]-omega*integral.sum(-1),
                                    deta.sum(-1),(self.X[:,None,:]*s).sum(-1),a[None,:]*s.sum(-1)],axis=-1))
        conditional=np.concatenate(Ls,axis=1);joint=conditional+self.logweights
        scores=logsumexp(joint,axis=1);post=np.exp(joint-scores[:,None])
        grad=np.sum(post[:,:,None]*np.concatenate(Ds,axis=1),axis=(0,1)) if gradient else None
        if posterior:return scores,grad,offsets,post,conditional
        return scores,grad

    def objective(self,z):
        s,g=self.evaluate(z)
        # Per-lineage scaling improves BFGS conditioning without changing MLE.
        return -float(s.mean()),-g/len(s)


def minimize_bfgs(fun,start,maxiter=180):
    z=np.clip(np.array(start,float),BOUNDS[:,0],BOUNDS[:,1]);f,g=fun(z);H=np.eye(len(z));evaluations=1
    history=[];status='iteration_limit'
    for iteration in range(maxiter):
        projected=g.copy();projected[(z<=BOUNDS[:,0]+1e-9)&(g>0)]=0;projected[(z>=BOUNDS[:,1]-1e-9)&(g<0)]=0
        if np.max(abs(projected))<2e-5:status='gradient_tolerance';break
        direction=-H@projected
        if direction@g>=0:H=np.eye(len(z));direction=-projected
        direction/=max(1.,np.max(abs(direction))/1.)
        step=1.;accepted=False
        for _ in range(25):
            trial=np.clip(z+step*direction,BOUNDS[:,0],BOUNDS[:,1]);delta=trial-z
            if np.max(abs(delta))<1e-12:break
            nf,ng=fun(trial);evaluations+=1
            if np.isfinite(nf) and nf<=f+1e-4*g@delta:accepted=True;break
            step*=.5
        if not accepted:status='line_search_stalled';break
        y=ng-g;sy=delta@y
        if sy>1e-12:
            rho=1/sy;V=np.eye(len(z))-rho*np.outer(delta,y);H=V@H@V.T+rho*np.outer(delta,delta)
        else:H=np.eye(len(z))
        z,f,g=trial,nf,ng;history.append(float(f))
    return z,dict(objective=float(f),max_abs_gradient=float(np.max(abs(g))),iterations=iteration+1,
                  evaluations=evaluations,status=status,objective_history=history)
