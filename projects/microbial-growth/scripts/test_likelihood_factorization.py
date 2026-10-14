import unittest
import numpy as np
import pandas as pd
from check_likelihood_factorization import transitions,kde_scores,cluster_corr,relative_gain,CANDIDATES


class FactorizationTests(unittest.TestCase):
    def test_transition_direction_and_gaps(self):
        g=pd.DataFrame({'lineage':[1]*4,'generation':[0,1,3,4],'time':[1.,2.,3.,4.],'alpha':[.1,.2,.3,.4],'length_birth':[2.,3.,4.,5.]})
        t=transitions([g]);np.testing.assert_array_equal(t.generation,[1,4])
        np.testing.assert_array_equal(t.previous_time,[1.,3.])
        np.testing.assert_array_equal(t.alpha,[.2,.4])

    def test_nested_density_and_normalization(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(100,5));y=rng.normal(size=100)
        grid=np.linspace(-10,10,4001);xq=np.zeros((len(grid),5))
        values=kde_scores(x,y,xq,grid)
        # Independent predictor bandwidths include the previous-alpha-only KDE.
        for j,h in enumerate([.15,.3,.6,1.2,2.4,np.inf]):
            index=CANDIDATES['birth_previous_alpha'].index((np.inf,h))
            np.testing.assert_allclose(values['birth_previous_alpha'][index],values['previous_alpha'][j],atol=1e-12)
        for model in values:
            mass=np.trapezoid(np.exp(values[model][0,2]),grid)
            self.assertAlmostEqual(float(mass),1.,places=4)

    def test_bootstrap(self):
        ids=np.repeat(np.arange(5),10);x=np.arange(50,dtype=float)
        r,lo,hi=cluster_corr(x,2*x,ids)
        self.assertAlmostEqual(r,1.);self.assertAlmostEqual(lo,1.);self.assertAlmostEqual(hi,1.)
        g,lo,hi=relative_gain(np.ones(50),np.ones(50)*.8,ids)
        self.assertAlmostEqual(g,20.);self.assertAlmostEqual(lo,20.);self.assertAlmostEqual(hi,20.)


if __name__=='__main__':unittest.main()
