import unittest
import numpy as np
from fit_empirical_detrended_ar1 import profile,fit,simulate,intercepts


class EmpiricalARTests(unittest.TestCase):
    def data(self):
        return np.random.default_rng(921).normal(size=(9,7))

    def test_profile_matches_dense_covariance(self):
        x=self.data();x-=x.mean(0);L,n=x.shape;p=.43
        R=p**np.abs(np.arange(n)[:,None]-np.arange(n));W=np.linalg.inv(R)
        for offsets in [False,True]:
            q=np.einsum('li,ij,lj->',x,W,x);ld=np.linalg.slogdet(R)[1]
            if offsets:
                s=W.sum();q-=np.sum((x@W.sum(1))**2)/s;ld+=np.log(s/n)
            df=(L-1)*(n-1 if offsets else n);v=q/df
            score,var,gotdf=profile(x,offsets,[p])
            self.assertEqual(gotdf,df)
            self.assertAlmostEqual(var[0],v,places=12)
            self.assertAlmostEqual(score[0],-.5*((L-1)*ld+df*np.log(v)),places=11)

    def test_arbitrary_generation_mean_invariance(self):
        x=self.data();shift=x+np.arange(x.shape[1])**2
        for offsets in [False,True]:
            np.testing.assert_allclose(profile(x,offsets)[0],profile(shift,offsets)[0],atol=1e-10)

    def test_offset_model_lineage_shift_invariance(self):
        x=self.data();shift=x+np.arange(len(x))[:,None]*2
        np.testing.assert_allclose(profile(x,True)[0],profile(shift,True)[0],atol=1e-10)

    def test_simulation_reestimates_generation_means(self):
        x=self.data();p=dict(phi=.4,stationary_sd=.2,innovation_sd=.2*np.sqrt(1-.4**2),static_lineage_offsets=True)
        y=simulate(x,p,10000);np.testing.assert_allclose(y.mean(1),0,atol=1e-14)
        # Centering across L lineages scales the marginal covariance by 1-1/L.
        noise=y-intercepts(x,p)[None,:,None]
        expected=.2**2*(1-1/len(x))*.4**np.abs(np.arange(7)[:,None]-np.arange(7))
        cov=np.einsum('rli,rlj->ij',noise,noise)/(len(y)*len(x))
        np.testing.assert_allclose(cov,expected,atol=.0008)


if __name__=='__main__':unittest.main()
