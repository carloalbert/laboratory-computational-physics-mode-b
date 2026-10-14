import unittest
import numpy as np
import pandas as pd
from fit_second_order_birth_transitions import BirthTransitionKDE,simulate,dataset,normal_sf


class SecondOrderBirthTests(unittest.TestCase):
    def model(self,ho=.1):
        previous=np.array([2.,2.,2.]);alpha=np.array([.5,.5,.5]);factor=np.array([.6,.6,.6]);growth=np.array([.1,.4,.8])
        return BirthTransitionKDE(previous,alpha,factor,previous*factor*np.exp(growth),(.1,.3,.1,.08),[1.,2.,3.],ho)

    def test_normalization_and_older_effect(self):
        # Start just inside support so quadrature uses the right-hand limit,
        # not the zero density assigned at the single boundary point.
        m=self.model();q=np.linspace(1.2+1e-10,7.,40001)
        for older in [1.,2.,3.]:
            density=m.density(q,2.,.5,.6,b_older=older)
            self.assertAlmostEqual(np.trapezoid(density,q),1.,delta=2e-4)
        self.assertGreater(m.weights(2.,.5,.6,1.)[0,0],.999)
        self.assertGreater(m.weights(2.,.5,.6,3.)[0,2],.999)

    def test_infinite_bandwidth_recovers_first_order(self):
        m=self.model(np.inf)
        first=BirthTransitionKDE(np.exp(m.logb),np.exp(m.loga),np.exp(m.logf),np.exp(m.logb+m.logf+m.g),(.1,.3,.1,.08))
        q=np.linspace(1.2,5.,100)
        np.testing.assert_array_equal(m.density(q,2.,.5,.6,b_older=1.2),first.density(q,2.,.5,.6))
        b=np.full(1000,2.)
        a=m.sample_fixed_context(b,.5,.6,np.random.default_rng(23),b_older=3.)
        c=first.sample_fixed_context(b,.5,.6,np.random.default_rng(23))
        np.testing.assert_array_equal(a[0],c[0])

    def test_sampling_older_weights_and_exact_fallback(self):
        m=self.model(.2);b,g=m.sample_fixed_context(np.full(60000,2.),.5,.6,np.random.default_rng(129),b_older=2.5)
        w=m.weights(2.,.5,.6,2.5)[0]
        correction=np.exp(-.5*(m.g/m.hg)**2)/np.sqrt(2*np.pi)/normal_sf(-m.g/m.hg)
        self.assertAlmostEqual(g.mean(),np.sum(w*(m.g+m.hg*correction)),delta=.003)
        m=self.model(.00001);_,g=m.sample_fixed_context(np.full(1000,2.),.5,.6,np.random.default_rng(323),b_older=2.5)
        self.assertEqual(m.fallback_draws,1000)
        self.assertAlmostEqual(g.mean(),.8,delta=.01)

    def test_two_seed_replay_and_previous_alpha_divisor(self):
        class Probe:
            logb=np.array([-10.,10.]);logolder=logb;draws=0;fallback_draws=0
            def sample_fixed_context(self,b,a,f,rng,b_older=None):
                # Both generated birth histories must be passed, not observed births.
                growth=.1*np.log(b/b_older)**2+a
                self.draws+=len(b);return b*f*np.exp(growth),growth
        group=pd.DataFrame({'lineage':7,'generation':np.arange(5),'length_birth':[2.,2.5,9.,9.,9.],
            'alpha':[.1,.2,.4,.8,1.6],'time':[1.,1.,1.,1.,1.],'f_effective':[np.nan,.6,.6,.65,.7]})
        b,t,diag=simulate([group],Probe(),3)
        np.testing.assert_array_equal(b[:,0,:2],np.tile([2.,2.5],(3,1)))
        self.assertTrue(np.all(np.isnan(t[:,0,[0,4]])))
        for n in range(2,5):
            expected_g=.1*np.log(b[:,0,n-1]/b[:,0,n-2])**2+group.alpha.iloc[n]
            np.testing.assert_allclose(t[:,0,n-1],expected_g/group.alpha.iloc[n-1])
            np.testing.assert_allclose(b[:,0,n],b[:,0,n-1]*group.f_effective.iloc[n]*np.exp(expected_g))
        tab=dataset([group])
        self.assertEqual(len(tab),3)
        np.testing.assert_array_equal(tab.older_birth,[2.,2.5,9.])
        np.testing.assert_array_equal(tab.alpha,[.4,.8,1.6])


if __name__=='__main__':unittest.main()
