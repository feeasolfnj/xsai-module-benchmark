import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bench_support import BenchError, coding_parameters


class CodingProfileTests(unittest.TestCase):
    def test_dl_default_matches_verified_rfsim_workload(self):
        self.assertEqual(coding_parameters('DL'), dict(mcs=9, symbols=13, dmrs_count=1,
                                                       dmrs_groups=2, tbslbrm=184424))

    def test_ul_keeps_existing_mcs14_profile(self):
        self.assertEqual(coding_parameters('UL'), dict(mcs=14, symbols=13))

    def test_historical_dl_profiles_remain_explicitly_available(self):
        p=coding_parameters('DL', mcs=14, symbols=13, dmrs_count=1,
                            dmrs_groups=1, tbslbrm=950984)
        self.assertEqual((p['mcs'],p['symbols'],p['dmrs_groups']), (14,13,1))
        self.assertEqual(coding_parameters('DL', mcs=9, symbols=12, dmrs_groups=1)['symbols'],12)

    def test_rejects_invalid_or_misapplied_dmrs_settings(self):
        for kwargs in [dict(dmrs_count=0),dict(dmrs_count=3,symbols=2),
                       dict(dmrs_groups=3),dict(tbslbrm=0)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(BenchError):
                coding_parameters('DL',**kwargs)
        for metric in ['UL','llr']:
            with self.assertRaises(BenchError):coding_parameters(metric,dmrs_groups=2)
