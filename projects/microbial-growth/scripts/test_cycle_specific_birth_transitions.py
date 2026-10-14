import unittest
import itertools
import numpy as np
import pandas as pd
import fit_cycle_specific_birth_transitions as fit


class CycleSpecificTests(unittest.TestCase):
    def group(self,lineage=7,n=6):
        return pd.DataFrame({'lineage':lineage,'generation':np.arange(20,20+n),
            'length_birth':np.linspace(2.,2.8,n),'alpha':np.linspace(.2,.8,n),
            'time':np.arange(1.,n+1),'f_effective':np.r_[np.nan,np.full(n-1,.6)]})

    def test_cycle_index_is_relative_and_resets(self):
        tab=fit.dataset([self.group(7,6),self.group(8,4)])
        np.testing.assert_array_equal(tab.cycle,[1,2,3,4,1,2])
        np.testing.assert_array_equal(tab.elapsed,[1.,3.,6.,10.,1.,3.])

    def test_same_cycle_loo_score_matches_individual_densities(self):
        groups=[self.group(i,3) for i in range(3)]
        groups[1]['length_birth']*=1.2;groups[2]['length_birth']*=.9
        tab=fit.dataset(groups);tab.loc[1,'g']+=.05
        tab.loc[1,'birth']=tab.loc[1,'previous_birth']*tab.loc[1,'F']*np.exp(tab.loc[1,'g'])
        names=['HB_CYCLE','HA','HF','HG','HOLDER','HT','GRID'];saved={k:getattr(fit,k) for k in names}
        try:
            fit.HB_CYCLE=[.1,np.inf];fit.HA=[.3];fit.HF=[.1];fit.HG=[.08];fit.HOLDER=[.2];fit.HT=[2.]
            fit.GRID=list(itertools.product(fit.HB_CYCLE,fit.HA,fit.HF,fit.HG,fit.HOLDER,fit.HT))
            scores=fit.loo_scores(tab)
            for h,score in zip(fit.GRID,scores):
                vals=[]
                for i,row in tab.iterrows():
                    model=fit.make_model(tab.drop(i),h)
                    vals.append(np.log(model.density(row.birth,row.previous_birth,row.alpha,row.F,b_older=row.older_birth,elapsed=row.elapsed)))
                self.assertAlmostEqual(score,np.mean(vals),places=12)
        finally:
            for k,v in saved.items():setattr(fit,k,v)

    def test_tied_scores_prefer_dropping_predictors(self):
        saved=fit.GRID
        try:
            fit.GRID=[(.04,.08,.02,.08,.04,1.),(np.inf,np.inf,np.inf,.08,np.inf,np.inf)]
            self.assertEqual(fit.select_bandwidth(np.array([1.,1.])),1)
        finally:fit.GRID=saved

    def test_routing_uses_generating_cycle_not_lineage(self):
        class Probe:
            elapsed=np.array([0.,100.]);draws=0;fallback_draws=0
            def __init__(self,growth):self.growth=growth;self.draws=0
            def sample_fixed_context(self,b,a,f,rng,b_older=None,elapsed=None):
                self.draws+=len(b);growth=np.full_like(b,self.growth)
                return b*f*np.exp(growth),growth
        groups=[self.group(7,6),self.group(8,4)];models={k:Probe(.1*k) for k in range(1,5)}
        b,t,s,diag=fit.simulate(groups,None,2,cycle_models=models)
        for j,g in enumerate(groups):
            for n in range(2,len(g)):
                np.testing.assert_allclose(t[:,j,n-1],.1*(n-1)/g.alpha.iloc[n-1])
                np.testing.assert_allclose(s[:,j,n]-s[:,j,n-1],t[:,j,n-1])
        self.assertEqual(models[1].draws,4);self.assertEqual(models[4].draws,2)
        self.assertEqual(diag['mixture_draws'],12)


if __name__=='__main__':unittest.main()
