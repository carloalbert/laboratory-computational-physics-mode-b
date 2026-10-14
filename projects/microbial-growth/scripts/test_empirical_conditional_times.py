import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from empirical_conditional_time_density import ConditionalTimeKDE,normal_sf


class ConditionalKDETests(unittest.TestCase):
    def setUp(self):
        self.kde=ConditionalTimeKDE([2.,2.5,3.,3.5],[.3,.5,.7,.9],[1.8,1.1,.8,.5],.2,.25,.05)

    def test_cdf_approximation(self):
        z=np.linspace(-9,9,1001)
        exact=np.array([.5*math.erfc(v/math.sqrt(2)) for v in z])
        self.assertLess(np.max(abs(normal_sf(z)-exact)),8e-8)

    def test_normalization_and_probability(self):
        grid=np.linspace(0,12,20001)
        f=self.kde.density(grid,2.7,.45)
        self.assertLess(abs(np.trapezoid(f,grid)-1),1e-5)
        self.assertAlmostEqual(float(self.kde.cdf(100,2.7,.45)),1.,places=6)
        self.assertEqual(float(self.kde.cdf(-1,2.7,.45)),0.)
        a=float(self.kde.probability(.8,1.6,2.7,.45))
        b=float(self.kde.cdf(1.6,2.7,.45)-self.kde.cdf(.8,2.7,.45))
        self.assertAlmostEqual(a,b,places=7)
        self.assertEqual(float(self.kde.density(-1,2.7,.45)),0.)

    def test_sampling_and_jacobian(self):
        # Remove alpha conditioning to test exactly the g->t Jacobian.
        kde=ConditionalTimeKDE([2.,2.5],[.5,.6],[1.2,1.],.2,np.inf,.04)
        np.testing.assert_allclose(kde.density([.5,1.],2.4,1.),2*kde.density([1.,2.],2.4,.5))
        draws=kde.sample(np.full(30000,2.4),np.full(30000,.5),np.random.default_rng(142))
        self.assertTrue(np.all(draws>0))
        self.assertLess(abs(draws.mean()-kde.conditional_mean(2.4,.5)[0]),.003)


if __name__=='__main__':unittest.main()
