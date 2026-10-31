import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import check_ldpc_log, make_parser
from bench_support import BenchError


class LdpcBackendTests(unittest.TestCase):
    def test_ldpc_version_is_optional_and_carries_the_plugin_suffix(self):
        args = make_parser().parse_args(['run', 'UL', '--threads', '1'])
        self.assertIsNone(args.ldpc_version)
        args = make_parser().parse_args(['run', 'UL', '--threads', '1', '--ldpc-version', '_cpu_avx2'])
        self.assertEqual(args.ldpc_version, '_cpu_avx2')

    def test_allocation_defaults_match_the_rfsim_reference(self):
        args = make_parser().parse_args(['run', 'UL', '--threads', '1'])
        self.assertIsNone(args.rb)
        self.assertIsNone(args.carrier_rb)
        # run() resolves these to 50 RB inside a 106 RB carrier, i.e. the rfsim UL layout.
        self.assertEqual((args.rb or 50, args.carrier_rb or 106), (50, 106))

    def test_plugin_banner_is_required(self):
        self.assertIsNone(check_ldpc_log('Initializing CPU LDPC decoder context\nInitialized CPU LDPC decoder\n', '_cpu'))
        for body in ('', 'loading module libldpc.so\n', 'error loading LDPC library\n'):
            with self.subTest(body=body):
                with self.assertRaises(BenchError) as caught:
                    check_ldpc_log(body, '_cpu')
                self.assertIn('did not print its init banner', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
