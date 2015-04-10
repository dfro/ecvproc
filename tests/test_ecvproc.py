import unittest
from unittest import TestCase
import numpy as np
import ecvproc

class TestEcvproc(TestCase):
    def test_cv_read(self):
        cap, volt = ecvproc.cv_read('test.cv', 'Cp')
        c_ref = np.array([0.01216187, 0.01013489, 0.00810791, 0.00608093,
                          0.00405396,  0.00202698,  0.01418885,  0.01621583,
                          0.0182428 ,  0.02026978,  0.02229676])
        v_ref = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1. , -1. , -0.9,
                          -0.8, -0.7, -0.6])
        self.assertTrue(np.allclose(cap, c_ref))
        self.assertTrue(np.allclose(volt, v_ref))

    def test_iv_read(self):
        cur, volt = ecvproc.iv_read('test.iv')
        cur_ref = np.array([2.,  1.,  0.,  3.,  4.,  5.])
        v_ref = np.array([-2., -1.5, -1., -1., -0.5,  0.])
        self.assertTrue(np.allclose(cur, cur_ref))
        self.assertTrue(np.allclose(volt, v_ref))

    def test_ep_read(self):
        doping, depth = ecvproc.ep_read('test.ep')
        dop_ref = np.array([1.17e+15, 3.381e+16, -8.4267e+14,
                            1.4746e+17, 1.3958e+15])
        depth_ref = [0.1,  0.86,  0.88,  0.98,  1.0]
        self.assertTrue(np.allclose(depth, depth_ref))
        self.assertTrue(np.allclose(doping, dop_ref))

    def test_log_read(self):
        n, f = ecvproc.log_read('test.log', 'No', 'F1')
        n_ref = np.array([1, 2, 3, 4, 5, 6, 7])
        f_ref = np.array([740., 740., 740., 740., 5555., 5555., 5555.])
        self.assertTrue(np.allclose(n, n_ref))
        self.assertTrue(np.allclose(f, f_ref))

    def test_lin_fit(self):
        volt = np.array([0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
        cap = np.array([ 0.11543373, 0.1259744, 0.14012403,
                         0.16048704, 0.19359813, 0.26199768])
        dop_ref = -1e17
        cap_fit, volt_fit, doping = ecvproc.lin_fit(cap, volt, eps=11.7)
        self.assertTrue(np.allclose(doping, dop_ref, rtol=0.005))