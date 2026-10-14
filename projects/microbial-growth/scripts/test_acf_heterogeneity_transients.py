import unittest
import numpy as np
from analyze_acf_heterogeneity_transients import covariance_score,acf,summaries


class AcfSourceTests(unittest.TestCase):
    def item(self):
        return {'id':1.,'g':np.array([0.,1.,2.,4.,5.,6.]),'y':np.array([.2,.3,.1,.6,.5,.8])}

    def test_iid_restricted_score(self):
        d=self.item();n=len(d['y']);v=np.var(d['y'],ddof=1)
        expected=-.5*(np.log(n)+(n-1)*np.log(v))
        self.assertAlmostEqual(covariance_score([d],0.)[0],expected,places=12)

    def test_location_scale_and_nested_covariances(self):
        d=self.item();shift=dict(d,y=d['y']+5);scale=dict(d,y=d['y']*3)
        score=covariance_score([d],.6)[0]
        self.assertAlmostEqual(covariance_score([shift],.6)[0],score,places=11)
        self.assertAlmostEqual(covariance_score([scale],.6)[0],score-5*np.log(3),places=11)
        self.assertAlmostEqual(covariance_score([d],.6,.9,1.)[0],score,places=12)
        self.assertAlmostEqual(covariance_score([d],.3,.6,0.)[0],score,places=12)

    def test_acf_excludes_generation_gap(self):
        d=self.item();y=d['y'];c=y-y.mean();pairs=[(0,1),(1,2),(3,4),(4,5)]
        expected=sum(c[i]*c[j] for i,j in pairs)/(c@c)
        self.assertAlmostEqual(acf(y[None],d['g'])[0,1],expected,places=12)
        self.assertTrue(np.isnan(acf(y[None],d['g'])[0,3]))
        np.testing.assert_allclose(acf(y[None],d['g']),acf((3*y+7)[None],d['g']),equal_nan=True)

    def test_irregular_covariance_uses_generation_distance(self):
        d=self.item();_,p=covariance_score([d],.6,return_parameters=True)
        chol=p[0][2];cov=chol@chol.T
        self.assertAlmostEqual(cov[2,3],.6**2,places=12)
        self.assertAlmostEqual(cov[0,5],.6**6,places=12)


if __name__=='__main__':unittest.main()
