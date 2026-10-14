"""Small invariance tests for the stochastic-jump implementation."""
import unittest
import numpy as np
import pandas as pd
from fit_conditional_effective_jumps import (
    FastReplay, segments, transitions, fit_jump, jump_log_density, simulate,
)


class EffectiveJumpTests(unittest.TestCase):
    def setUp(self):
        groups=[]
        for lineage in [1,2]:
            n=np.arange(9)
            groups.append(pd.DataFrame({'lineage':lineage,'generation':n,
                'length_birth':2.5+.2*np.cos(n+lineage),'alpha':.5+.02*n,
                'time':1+.12*np.sin(n),'f':.55}))
        self.groups=segments(groups)
        self.refs={'b':2.5,'alpha':.55,'t':1.,'g':.55}
        self.hazard={'u':1.47,'omega2':11.6,'eta':8.2,'q':.61,'kappa':0.}

    def test_reconstruction_and_jacobian(self):
        tr=transitions(self.groups)
        np.testing.assert_allclose(tr.next_b,tr.b*np.exp(tr.alpha*tr.t)*tr.f_eff)
        fit=fit_jump(tr,['b','t'],self.refs)
        difference=jump_log_density(tr,fit,self.refs,'next_b')-jump_log_density(tr,fit,self.refs)
        np.testing.assert_allclose(difference,-np.log(tr.b)-tr.alpha*tr.t)

    def test_fixed_and_sampled_size_update(self):
        rp=FastReplay(self.groups,2.5,.55,replicates=12)
        jump={'features':['b','t'],'coefficients':[np.log(.6),.1,-.05],'sigma':.08}
        for params in [None,jump]:
            b,t,f=simulate(rp,self.hazard,params,self.refs,seed=671)
            self.assertTrue(np.all(b[:,rp.mask]>0))
            self.assertTrue(np.all(t[:,rp.mask]>0))
            for i,g in enumerate(self.groups):
                L=len(g)
                reconstructed=b[:,i,:L-1]*np.exp(t[:,i,:L-1]*rp.alpha[i,:L-1])*f[:,i,:L-1]
                np.testing.assert_allclose(b[:,i,1:L],reconstructed)
                if params is None:
                    np.testing.assert_allclose(f[:,i,:L-1],np.broadcast_to(g.f_effective.iloc[1:].to_numpy(),(12,L-1)))

    def test_reproducibility(self):
        rp=FastReplay(self.groups,2.5,.55,replicates=3)
        jump={'features':[],'coefficients':[np.log(.6)],'sigma':.08}
        first=simulate(rp,self.hazard,jump,self.refs,seed=177)
        second=simulate(rp,self.hazard,jump,self.refs,seed=177)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
