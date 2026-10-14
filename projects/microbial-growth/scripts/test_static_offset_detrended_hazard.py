import unittest
import numpy as np
from fit_static_offset_detrended_hazard import simulate
from fit_empirical_detrended_hazard import Likelihood,means
from static_offset_growth_hazard import MarginalLikelihood,pack,unpack


class StaticOffsetTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(56);x=rng.normal(1.,.1,(4,6));g=rng.uniform(.25,.9,(4,5))
        return dict(ids=np.arange(4),x=x,g=g,q=np.diff(x,axis=1)-g,a=np.ones_like(g),t=g)

    def params(self):return dict(u=.6,omega2=12.,eta=8.,kappa=1.,tau_a=.7)

    def test_zero_variance_recovers_shared_likelihood(self):
        d=self.data();X=d['x'][:,:-1]-1;p=self.params();z=pack(p,1e-10)
        got=MarginalLikelihood(X,d['g'],128,41).evaluate(z,gradient=False)[0]
        want=Likelihood(X,d['g'],128).scores(p).reshape(4,5).sum(1)
        np.testing.assert_allclose(got,want,atol=1e-10)

    def test_analytic_gradient(self):
        d=self.data();ll=MarginalLikelihood(d['x'][:,:-1]-1,d['g'],48,61);z=pack(self.params())
        f,g=ll.objective(z);numeric=[]
        for j in range(5):
            dz=np.zeros(5);dz[j]=1e-5;numeric.append((ll.objective(z+dz)[0]-ll.objective(z-dz)[0])/2e-5)
        np.testing.assert_allclose(g,numeric,atol=2e-7,rtol=2e-7)

    def test_posterior_normalization_and_shared_offset(self):
        d=self.data();X=d['x'][:,:-1]-1;z=pack(self.params());ll=MarginalLikelihood(X,d['g'],64,61)
        scores,_,nodes,post,_=ll.evaluate(z,posterior=True)
        np.testing.assert_allclose(post.sum(1),1.,atol=1e-12)
        separate=sum(MarginalLikelihood(X[:,i:i+1],d['g'][:,i:i+1],64,61).evaluate(z,gradient=False)[0] for i in range(5))
        self.assertGreater(np.max(abs(scores-separate)),.01)

    def test_equivalent_lineage_threshold(self):
        d=self.data();X=d['x'][:,:-1]-1;p=self.params();ll=MarginalLikelihood(X,d['g'],128,11)
        _,_,nodes,post,conditional=ll.evaluate(pack(p),posterior=True)
        j=7;changed=dict(p,u=p['u']*np.exp(-nodes[j]/p['eta']))
        want=Likelihood(X,d['g'],128).scores(changed).reshape(4,5).sum(1)
        np.testing.assert_allclose(conditional[:,j],want,atol=1e-11)

    def test_fixed_offset_replay_and_centered_identity(self):
        d=self.data();m=means(d);p=self.params();nodes=np.array([-.2,.4]);post=np.tile([1.,0.],(4,1))
        x,g,a=simulate(d,m,p,'Posterior_offsets',nodes,post,reps=20)
        np.testing.assert_array_equal(a,np.full((20,4),-.2))
        self.assertTrue(np.all(g>0))
        np.testing.assert_allclose(np.diff(x-m['x'],axis=-1),(d['q']-m['q'])+(g-m['g']),atol=1e-14)


if __name__=='__main__':unittest.main()
