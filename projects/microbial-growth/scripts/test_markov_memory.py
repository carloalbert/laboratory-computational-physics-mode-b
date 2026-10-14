"""Indexing, nesting, and regression checks for the memory diagnostic."""
import unittest
import numpy as np
import pandas as pd
from check_markov_memory import windows,Basis,ridge_path,state_features,interval


class MarkovDiagnosticTests(unittest.TestCase):
    def test_no_gap_crossing(self):
        groups=[]
        for lineage,generations in [(2,[0,1,2,3,5,6,7,8]),(3,[0,1,2])]:
            groups.append(pd.DataFrame({'lineage':lineage,'generation':generations,'time':np.exp(np.array(generations)/10),'alpha':.5,'length_birth':2.}))
        z,ids,gens=windows(groups)
        np.testing.assert_array_equal(gens,[3,8]);np.testing.assert_array_equal(ids,[2,2])
        np.testing.assert_allclose(z[:,:,0],[[0,.1,.2,.3],[.5,.6,.7,.8]])

    def test_state_growth(self):
        z=np.log([[2.,.3,3.],[1.,.7,2.]])
        np.testing.assert_allclose(state_features(z)[:,-1],[.6,.7])

    def test_current_basis_retained(self):
        rng=np.random.default_rng(111);x=rng.normal(size=(250,7))
        base=Basis(x[:,:3],1.,3);extended=Basis(x,1.,3)
        # The first radial block of the extended basis equals baseline's block.
        baseline=base.transform(x[:,:3])[:,9:]
        augmented=extended.transform(x)[:,35:35+baseline.shape[1]]
        np.testing.assert_allclose(baseline,augmented,atol=1e-12)

    def test_ridge_fit(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(300,3));x-=x.mean(0)
        y=np.column_stack([x[:,0]-2*x[:,1]+3,.4*x[:,2]])
        pred=ridge_path(x,y,x)[0]
        self.assertLess(np.max(abs(pred-y)),1e-4)

    def test_cluster_interval(self):
        mean,lo,hi=interval(np.ones(20)*.25,np.repeat(np.arange(4),5))
        self.assertAlmostEqual(mean,.25);self.assertAlmostEqual(lo,.25);self.assertAlmostEqual(hi,.25)


if __name__=='__main__':unittest.main()
