import unittest
import numpy as np
import pandas as pd
from fit_generative_glu import FastReplay,simulate,check_recursion,fit_factors,factor_density


class GenerativeTests(unittest.TestCase):
    def setUp(self):
        self.groups=[pd.DataFrame({'lineage':j,'generation':np.arange(n),'length_birth':np.ones(n)*2.5,'alpha':np.linspace(.4,.7,n),'time':np.ones(n),'f_effective':np.r_[np.nan,np.ones(n-1)*.6]}) for j,n in [(1,40),(2,35)]]
        self.rp=FastReplay(self.groups,2.5,.52,replicates=40)
        self.fit={'references':{'b':2.5,'alpha':.52},'hazard':{'b_star':2.5,'u':1.45,'omega2':11.,'eta':8.3,'kappa':-.15},'growth':{'coefficients':[.005,.48],'sigma':.258},'factor':{'mu_log_F':-.5185,'sigma_log_F':.0837}}

    def test_recursion_initial_conditions_and_new_births(self):
        generated=simulate(self.rp,self.fit);control=simulate(self.rp,self.fit,True)
        self.assertLess(check_recursion(self.rp,generated),1e-12)
        np.testing.assert_allclose(generated[0][:,:,0],2.5);np.testing.assert_allclose(generated[1][:,:,0],.4)
        self.assertTrue(np.all(np.isnan(generated[3][:,:,0])))
        self.assertGreater(np.max(abs(generated[0]-control[0])),.1)
        np.testing.assert_array_equal(generated[1],control[1])

    def test_no_recorded_factor_leakage(self):
        before=simulate(self.rp,self.fit)
        self.rp.f[:]=.91
        after=simulate(self.rp,self.fit)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_factor_fit_and_density(self):
        rng=np.random.default_rng(45);f=np.exp(-.52+.08*rng.normal(size=30000))
        p=fit_factors([pd.DataFrame({'f_effective':np.r_[np.nan,f]})])
        self.assertAlmostEqual(p['mu_log_F'],-.52,delta=.002);self.assertAlmostEqual(p['sigma_log_F'],.08,delta=.002)
        grid=np.linspace(.01,2,10001)
        self.assertAlmostEqual(float(np.trapezoid(np.exp(factor_density(grid,p)),grid)),1.,places=6)


if __name__=='__main__':unittest.main()
