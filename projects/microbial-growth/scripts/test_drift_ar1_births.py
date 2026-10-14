import unittest
import numpy as np
from fit_drift_ar1_births import sufficient, fit, contrast_scores, drift_basis, whiten, simulate


class DriftARTests(unittest.TestCase):
    def items(self):
        return [dict(id=0, g=np.array([0., 1., 3., 4., 5., 8.]), y=np.array([2.1, 2.3, 2., 2.7, 2.6, 2.8])),
                dict(id=1, g=np.arange(7.), y=np.array([2., 1.8, 2.4, 2.5, 2.3, 2.5, 2.9]))]

    def test_saturation_linear_limit(self):
        g = np.arange(30.)
        np.testing.assert_allclose(drift_basis(g, 1e10), g, atol=5e-8)
        np.testing.assert_array_equal(drift_basis(g, None), 0*g)

    def test_whitening_and_contrast_density(self):
        phi = .6; items = self.items(); st = sufficient(items, np.inf, [phi])
        p = dict(phi=phi, slope=.04, stationary_sd=.3)
        for j, item in enumerate(items):
            y = item['y']-.04*item['g']; n = len(y)
            R = phi**np.abs(item['g'][:, None]-item['g'])
            inv = np.linalg.inv(R); one = np.ones(n); s = one@inv@one
            q = y@inv@y-(one@inv@y)**2/s
            expected = -.5*((n-1)*np.log(2*np.pi*.09)+np.linalg.slogdet(R)[1]+np.log(s/n)+q/.09)
            self.assertAlmostEqual(contrast_scores(st, p)[j], expected, places=10)
            self.assertAlmostEqual(whiten(y, item['g'], phi)@whiten(y, item['g'], phi), y@inv@y, places=10)

    def test_reml_matches_dense_design(self):
        phi = .4; items = self.items(); st = sufficient(items, np.inf, [phi]); p = fit(st)
        n = sum(len(d['y']) for d in items); W = np.zeros((n, n)); X = np.zeros((n, 3)); off = 0
        y = np.concatenate([d['y'] for d in items])
        for j, d in enumerate(items):
            stop = off+len(d['y']); sl = slice(off, stop)
            W[sl, sl] = np.linalg.inv(phi**np.abs(d['g'][:, None]-d['g']))
            X[sl, j] = 1; X[sl, 2] = d['g']; off = stop
        beta = np.linalg.solve(X.T@W@X, X.T@W@y); e = y-X@beta
        self.assertAlmostEqual(p['slope'], beta[-1], places=11)
        self.assertAlmostEqual(p['stationary_sd']**2, e@W@e/(n-3), places=11)

    def test_lineage_intercept_invariance(self):
        items = self.items(); moved = [dict(d, y=d['y']+(j+1)*3) for j, d in enumerate(items)]
        a = fit(sufficient(items, np.inf)); b = fit(sufficient(moved, np.inf))
        for key in ['phi', 'slope', 'stationary_sd']:
            self.assertAlmostEqual(a[key], b[key], places=10)

    def test_simulation_covariance_and_drift(self):
        item = self.items()[0]; p = dict(phi=.6, slope=.04, stationary_sd=.3, tau=np.inf)
        y = simulate([item], p, 100000)[0]
        np.testing.assert_allclose(y.mean(0)-y[:, 0].mean(), .04*item['g'], atol=.004)
        expected = .09*.6**np.abs(item['g'][:, None]-item['g'])
        np.testing.assert_allclose(np.cov(y.T), expected, atol=.002)


if __name__=='__main__':
    unittest.main()
