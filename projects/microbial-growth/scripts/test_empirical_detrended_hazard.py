import unittest
import numpy as np
from fit_empirical_detrended_hazard import means, center, Likelihood, shifted_log_density, simulate
from birth_size_dependent_time_hazard import log_time_density


class DetrendedHazardTests(unittest.TestCase):
    def params(self): return dict(u=.8,omega2=6.,eta=3.,kappa=.7)

    def data(self):
        rng=np.random.default_rng(4); x=rng.normal(1.,.1,(7,6));g=rng.uniform(.2,1.,(7,5))
        return dict(ids=np.arange(7),x=x,g=g,q=np.diff(x,axis=1)-g,a=np.ones_like(g),t=g)

    def test_centered_balance_and_means(self):
        d=self.data();m=means(d);c=center(d,m)
        for v in c.values(): np.testing.assert_allclose(v.mean(0),0,atol=1e-15)
        np.testing.assert_allclose(np.diff(c['x'],axis=1),c['q']+c['g'],atol=1e-15)

    def test_original_hazard_equivalence(self):
        p=self.params();x=np.array([-.2,0,.3]);g=np.array([.3,.7,1.])
        expected=log_time_density(g,np.exp(x),np.ones(3),p['u'],p['omega2'],p['eta'],p['kappa'],1.,order=128)
        np.testing.assert_allclose(Likelihood(x,g,128).scores(p),expected,atol=1e-12)

    def test_shifted_support_and_jacobian(self):
        p=self.params();G=np.array([-.7,-.4,0.,.3]);X=np.zeros(4);mu=.6
        actual=shifted_log_density(G,X,mu,p)
        self.assertTrue(np.isneginf(actual[0]))
        np.testing.assert_allclose(actual[1:],Likelihood(X[1:],G[1:]+mu,128).scores(p),atol=1e-12)
        self.assertTrue(np.isfinite(actual[1]))

    def test_profiled_omega_optimum(self):
        ll=Likelihood(np.zeros(4),np.array([.2,.5,.6,1.]));z=np.array([np.log(.8),np.log(3.),3.7])
        frac,integral,omega=ll.components(z)
        self.assertAlmostEqual(4/omega-integral.sum(),0.,places=12)
        value=ll(z)
        for factor in [.9,1.1]:
            other=np.sum(np.log(omega*factor)+frac-omega*factor*integral)
            self.assertGreater(value,other)

    def test_density_normalization(self):
        g=np.linspace(.000001,6,12001)
        density=np.exp(shifted_log_density(g-.6,np.full_like(g,.2),.6,self.params(),64))
        self.assertAlmostEqual(float(np.trapezoid(density,g)),1.,places=6)

    def test_simulation_preserves_identity(self):
        d=self.data();m=means(d);x,g=simulate(d,m,self.params(),'Detrended',0,12)
        X=x-m['x'];G=g-m['g'];Q=d['q']-m['q']
        np.testing.assert_allclose(np.diff(X,axis=-1),Q+G,atol=1e-15)
        self.assertTrue(np.all(g>0))


if __name__=='__main__': unittest.main()
