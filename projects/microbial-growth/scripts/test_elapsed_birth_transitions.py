import unittest
import numpy as np
import pandas as pd
import itertools
import fit_elapsed_birth_transitions as fit
from fit_elapsed_birth_transitions import BirthTransitionKDE,normal_sf,dataset,simulate


class ElapsedBirthTests(unittest.TestCase):
    def model(self,ht=2.):
        b=np.full(3,2.);a=np.full(3,.5);f=np.full(3,.6);g=np.array([.1,.4,.8])
        return BirthTransitionKDE(b,a,f,b*f*np.exp(g),(.1,.3,.1,.08),[2.,2.,2.],.2,[0.,10.,20.],ht)

    def test_normalization_clock_effect_and_sampling(self):
        m=self.model();q=np.linspace(1.2+1e-10,7.,40001)
        for elapsed in [0.,10.,20.]:
            den=m.density(q,2.,.5,.6,b_older=2.,elapsed=elapsed)
            self.assertAlmostEqual(np.trapezoid(den,q),1.,delta=2e-4)
        self.assertGreater(m.weights(2.,.5,.6,2.,0.)[0,0],.999)
        self.assertGreater(m.weights(2.,.5,.6,2.,20.)[0,2],.999)
        _,g=m.sample_fixed_context(np.full(60000,2.),.5,.6,np.random.default_rng(88),b_older=2.,elapsed=13.)
        w=m.weights(2.,.5,.6,2.,13.)[0];correction=np.exp(-.5*(m.g/m.hg)**2)/np.sqrt(2*np.pi)/normal_sf(-m.g/m.hg)
        self.assertAlmostEqual(g.mean(),np.sum(w*(m.g+m.hg*correction)),delta=.003)

    def test_infinite_clock_bandwidth_exact_reduction(self):
        m=self.model(np.inf);old=BirthTransitionKDE(np.exp(m.logb),np.exp(m.loga),np.exp(m.logf),np.exp(m.logb+m.logf+m.g),(.1,.3,.1,.08),[2.,2.,2.],.2)
        b=np.full(1000,2.);q=np.linspace(1.21,5.,100)
        np.testing.assert_array_equal(m.density(q,2.,.5,.6,b_older=2.,elapsed=100.),old.density(q,2.,.5,.6,b_older=2.))
        a=m.sample_fixed_context(b,.5,.6,np.random.default_rng(23),b_older=2.,elapsed=100.)
        c=old.sample_fixed_context(b,.5,.6,np.random.default_rng(23),b_older=2.)
        np.testing.assert_array_equal(a[0],c[0])

    def test_clock_dense_fallback(self):
        m=self.model(.00001);_,g=m.sample_fixed_context(np.full(1000,2.),.5,.6,np.random.default_rng(981),b_older=2.,elapsed=18.)
        self.assertEqual(m.fallback_draws,1000);self.assertAlmostEqual(g.mean(),.8,delta=.01)

    def group(self):
        return pd.DataFrame({'lineage':7,'generation':np.arange(5),'length_birth':[2.,2.5,9.,9.,9.],
            'alpha':[.1,.2,.4,.8,1.6],'time':[1.,2.,3.,4.,5.],'f_effective':[np.nan,.6,.6,.65,.7]})

    def test_dataset_clock_precedes_target_time(self):
        g=self.group();tab=dataset([g]);np.testing.assert_array_equal(tab.elapsed,[1.,3.,6.])
        g.loc[3:,'time']=1000.;other=dataset([g]);np.testing.assert_array_equal(tab.elapsed,other.elapsed)

    def test_cv_scores_match_density(self):
        m=self.model();train=pd.DataFrame({'previous_birth':np.exp(m.logb),'alpha':np.exp(m.loga),
            'F':np.exp(m.logf),'older_birth':np.exp(m.logolder),'birth':np.exp(m.logb+m.logf+m.g),
            'elapsed':m.elapsed,'g':m.g})
        valid=train.iloc[:2].copy();valid['elapsed']+=1.
        names=['HB','HA','HF','HG','HOLDER','HT','GRID'];saved={name:getattr(fit,name) for name in names}
        try:
            fit.HB=[.1];fit.HA=[.3];fit.HF=[.1];fit.HG=[.08];fit.HOLDER=[.2];fit.HT=[2.,np.inf]
            fit.GRID=list(itertools.product(fit.HB,fit.HA,fit.HF,fit.HG,fit.HOLDER,fit.HT))
            values=fit.cv_scores(train,valid)
            for h,v in zip(fit.GRID,values):
                model=fit.make_model(train,h);den=model.density(valid.birth,valid.previous_birth,valid.alpha,valid.F,b_older=valid.older_birth,elapsed=valid.elapsed)
                self.assertAlmostEqual(v,np.log(den).sum(),places=12)
        finally:
            for name,value in saved.items():setattr(fit,name,value)

    def test_endogenous_clock_no_future_time_leakage(self):
        class Probe:
            elapsed=np.array([0.,100.]);draws=0;fallback_draws=0
            def sample_fixed_context(self,b,a,f,rng,b_older=None,elapsed=None):
                growth=.1+.01*elapsed;self.draws+=len(b)
                return b*f*np.exp(growth),growth
        g=self.group();b,t,s,diag=simulate([g],Probe(),3)
        initial=np.log(2.5/(2.*.6))/.1
        np.testing.assert_allclose(s[:,0,1],initial)
        for n in range(2,5):
            expected=(.1+.01*s[:,0,n-1])/g.alpha.iloc[n-1]
            np.testing.assert_allclose(t[:,0,n-1],expected)
            np.testing.assert_allclose(s[:,0,n],s[:,0,n-1]+expected)
        self.assertTrue(np.all(np.isnan(t[:,0,[0,4]])))
        # After the seeds, no observed duration or observed birth enters replay.
        altered=g.copy();altered['time']=999.;altered.loc[2:,'length_birth']=777.
        b2,t2,s2,_=simulate([altered],Probe(),3)
        np.testing.assert_array_equal(b,b2);np.testing.assert_array_equal(t,t2);np.testing.assert_array_equal(s,s2)


if __name__=='__main__':unittest.main()
