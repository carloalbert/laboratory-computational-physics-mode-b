import unittest
import numpy as np
import pandas as pd
from fit_joint_growth_timing import fit_growth,growth_density,simulate,FastReplay,statistics


class JointGrowthTimingTests(unittest.TestCase):
    def test_growth_mle_and_normalization(self):
        rng=np.random.default_rng(891);n=10000
        previous=np.exp(.2*rng.normal(size=n));birth=np.exp(.3*rng.normal(size=n))
        alpha=np.exp(.02+.55*np.log(previous)+.12*np.log(birth)+.1*rng.normal(size=n))
        d=pd.DataFrame({'previous_alpha':previous,'birth':birth,'alpha':alpha});refs={'b':1.,'alpha':1.}
        fit=fit_growth(d,refs)
        np.testing.assert_allclose(fit['coefficients'],[.02,.55,.12],atol=.015)
        self.assertAlmostEqual(fit['sigma'],.1,delta=.005)
        grid=np.linspace(.01,4,10001);q=pd.DataFrame({'previous_alpha':np.ones(len(grid)),'birth':np.ones(len(grid)),'alpha':grid})
        self.assertAlmostEqual(float(np.trapezoid(np.exp(growth_density(q,fit,refs)),grid)),1.,places=6)

    def test_recursion_and_alpha_invariance(self):
        groups=[pd.DataFrame({'lineage':i,'generation':np.arange(n),'length_birth':np.ones(n)*2.5,'alpha':np.linspace(.4,.7,n),'time':np.ones(n),'f_effective':np.r_[np.nan,np.ones(n-1)*.6]}) for i,n in [(1,40),(2,35)]]
        rp=FastReplay(groups,2.5,.5,replicates=20);refs={'b':2.5,'alpha':.5}
        hazard={'b_star':2.5,'u':1.45,'omega2':11.,'eta':8.3,'kappa':-.15}
        growth={'coefficients':[0.,.5,.1],'sigma':.2}
        recorded=simulate(rp,hazard,growth,refs,True,771);joint=simulate(rp,hazard,growth,refs,False,771)
        np.testing.assert_array_equal(recorded[0],joint[0])
        b,a,t=joint
        self.assertTrue(np.all(a[:,rp.mask]>0));self.assertTrue(np.all(t[:,rp.mask]>0))
        np.testing.assert_allclose(a[:,:,0],np.broadcast_to([.4,.4],(20,2)))
        self.assertGreater(np.max(abs(recorded[1]-a)),.01)
        for j,g in enumerate(groups):
            n=len(g)
            np.testing.assert_allclose(b[:,j,1:n],.6*b[:,j,:n-1]*np.exp(a[:,j,:n-1]*t[:,j,:n-1]))
        # Statistics must use generated alpha, not the observed replay covariate.
        actual=statistics(rp,b,a,t)
        np.testing.assert_allclose(actual['mean_alpha'],a[:,rp.mask].mean(1))
        np.testing.assert_allclose(actual['mean_g'],(a[:,rp.mask]*t[:,rp.mask]).mean(1))


if __name__=='__main__':unittest.main()
