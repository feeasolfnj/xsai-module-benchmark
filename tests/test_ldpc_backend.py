import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import check_ldpc_log, make_parser
from bench_support import BenchError, coding_allocation


class LdpcBackendTests(unittest.TestCase):
    def test_ldpc_version_is_optional_and_carries_the_plugin_suffix(self):
        args = make_parser().parse_args(['run', 'UL', '--threads', '1'])
        self.assertIsNone(args.ldpc_version)
        args = make_parser().parse_args(['run', 'UL', '--threads', '1', '--ldpc-version', '_cpu_avx2'])
        self.assertEqual(args.ldpc_version, '_cpu_avx2')

    def test_native_ul_defaults_to_24_rb(self):
        args = make_parser().parse_args(['run', 'UL', '--threads', '1'])
        self.assertIsNone(args.rb)
        self.assertIsNone(args.carrier_rb)
        self.assertEqual(coding_allocation(args.metric, args.rb, args.carrier_rb,
                                          args.ldpc_version), (24, 106))

    def test_cuda_ldpc_and_cpu_ports_default_to_50_rb(self):
        for version in ('_cuda', '_cuda_profile', '_cpu', '_cpu_avx2',
                        '_cpu_scalar', '_cpu_p8'):
            with self.subTest(version=version):
                self.assertEqual(coding_allocation('UL', ldpc_version=version), (50, 106))

    def test_other_plugin_names_keep_the_native_ul_allocation(self):
        for version in ('_custom', '_cpucustom', '_cudacustom'):
            with self.subTest(version=version):
                self.assertEqual(coding_allocation('UL', ldpc_version=version), (24, 106))

    def test_explicit_allocation_overrides_backend_defaults(self):
        for version in (None, '_cuda', '_cpu_avx2'):
            with self.subTest(version=version):
                self.assertEqual(coding_allocation('UL', 12, 24, version), (12, 24))
                self.assertEqual(coding_allocation('UL', 50, None, version), (50, 106))
                self.assertEqual(coding_allocation('UL', None, 52, version),
                                 (24 if version is None else 50, 52))

    def test_dl_defaults_to_50_rb(self):
        self.assertEqual(coding_allocation('DL'), (50, 106))

    def test_plugin_banner_is_required(self):
        self.assertIsNone(check_ldpc_log('Initializing CPU LDPC decoder context\nInitialized CPU LDPC decoder\n', '_cpu'))
        for body in ('', 'loading module libldpc.so\n', 'error loading LDPC library\n'):
            with self.subTest(body=body):
                with self.assertRaises(BenchError) as caught:
                    check_ldpc_log(body, '_cpu')
                self.assertIn('did not print its init banner', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
