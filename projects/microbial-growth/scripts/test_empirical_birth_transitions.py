import unittest
import numpy as np
import pandas as pd
from fit_empirical_birth_transitions import BirthTransitionKDE,simulate,normal_sf


class BirthTransitionTests(unittest.TestCase):
    def model(self,hb=.15):
        b=np.array([1.8,2.,2.2]);a=np.array([.3,.5,.8]);f=np.array([.5,.6,.7]);g=np.array([.03,.4,.8])
        return BirthTransitionKDE(b,a,f,b*f*np.exp(g),(hb,.3,.1,.1))

    def test_density_normalized_and_supported(self):
        m=self.model();q=np.linspace(1.2,10,40001)
        den=m.density(q,2.,.5,.6)
        self.assertAlmostEqual(np.trapezoid(den,q),1.,delta=1e-4)
        self.assertEqual(float(m.density(1.19,2.,.5,.6)),0.)

    def test_sample_matches_truncated_mixture(self):
        m=self.model();b,g=m.sample_fixed_context(np.full(60000,2.),.5,.6,np.random.default_rng(19))
        w=m.weights(2.,.5,.6)[0]
        correction=np.exp(-.5*(m.g/m.hg)**2)/np.sqrt(2*np.pi)/normal_sf(-m.g/m.hg)
        expected=np.sum(w*(m.g+m.hg*correction))
        self.assertAlmostEqual(g.mean(),expected,delta=.002)
        self.assertTrue(np.all(g>0));np.testing.assert_allclose(b,1.2*np.exp(g))

    def test_dense_fallback(self):
        m=self.model(.0001);b,g=m.sample_fixed_context(np.full(1000,2.1),.5,.6,np.random.default_rng(291))
        self.assertEqual(m.fallback_draws,1000)
        self.assertAlmostEqual(g.mean(),.8,delta=.015)

    def test_indexing_and_terminal_missing(self):
        class Probe:
            logb=np.array([-10.,10.]);draws=0;proposals=0;fallback_draws=0
            def sample_fixed_context(self,b,a,f,rng):
                self.draws+=len(b)
                return b*f*np.exp(a),np.full_like(b,a)
        group=pd.DataFrame({'length_birth':[2.,2.,2.,2.],'alpha':[.2,.4,.8,1.6],'f_effective':[np.nan,.6,.65,.7]})
        b,t,diag=simulate([group],Probe(),3)
        np.testing.assert_allclose(t[:,0,:3],2.)
        self.assertTrue(np.all(np.isnan(t[:,0,3])))
        np.testing.assert_allclose(b[:,0,1:],b[:,0,:-1]*group.f_effective.to_numpy()[1:]*np.exp(group.alpha.to_numpy()[:-1]*t[:,0,:3]))
        self.assertLess(diag['maximum_growth_reconstruction_error'],1e-12)


if __name__=='__main__':unittest.main()
