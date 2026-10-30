import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import check_ov_log, make_parser
from bench_support import BenchError

MODEL = '/srv/models/nrx_oai.onnx'
PROPS = ('{"bridgeAbi":1,"openvinoBuild":"2025.4.0","cpuPluginBuild":"2025.4.0",'
         '"inferenceNumThreads":8,"numStreams":1,"performanceMode":0,'
         '"inferencePrecision":"bf16","enableCpuPinning":0}')


def worker(topology='L', precision='bf16', matches=1):
    return ('NRX_WORKER_PROPS tid=4242 {"backend":"OV_BF16_HINT","tid":4242,'
            f'"topology":"{topology}","propertyOrigin":"CompiledModel.get_property",'
            f'"inferenceNumThreads":8,"numStreams":1,"performanceMode":0,'
            f'"inferencePrecision":"{precision}","enableCpuPinning":0,'
            f'"CPolicyMatches":{matches},"amxProven":false}}')


def log(echo=MODEL, topology='L', props=PROPS, workers=None, extra=''):
    rows = [f'CPU OV receiver: model = {echo}',
            f'CPU OV receiver: topology {"L (shared CompiledModel)" if topology == "L" else "C (per-worker model)"}',
            f'NRX_OV_PROPS {props}',
            worker(topology)]
    rows += workers or []
    return '\n'.join(rows) + '\n' + extra


class OvLauncherTests(unittest.TestCase):
    def test_backend_arguments(self):
        args = make_parser().parse_args(['run', 'neural', '--threads', '1',
                                         '--backend', 'ov', '--ov-model', MODEL, '--ov-topology', 'C'])
        self.assertEqual(args.neural_backend, 'ov')
        self.assertEqual(args.ov_model, MODEL)
        self.assertEqual(args.ov_topology, 'C')
        self.assertIsNone(make_parser().parse_args(['run', 'neural', '--threads', '1']).ov_topology)

    def test_records_the_readback_without_claiming_amx(self):
        result = check_ov_log(log(), MODEL, 'L')
        self.assertEqual(result['props']['inferencePrecision'], 'bf16')
        self.assertEqual(result['props']['inferenceNumThreads'], 8)
        self.assertEqual(result['workers'], 1)
        self.assertFalse(result['amx_proven'])
        self.assertEqual(result['worker']['topology'], 'L')

    def test_each_unproven_run_is_rejected(self):
        cases = [
            ('model echo differs', log(echo='/other/model.onnx'), 'differs from --ov-model'),
            ('wrong topology echo', log(topology='L'), 'did not confirm topology C'),
            ('fp32 precision', log(props=PROPS.replace('"bf16"', '"f32"')), 'not bf16'),
            ('policy mismatch line', log(extra='NRX worker property readback failed or C policy mismatch\n'),
             'C policy mismatch'),
            ('no worker record', log(workers=[]).replace(worker('L'), ''), 'No NRX_WORKER_PROPS'),
            ('unparseable props', log(props='{broken'), 'Unparseable'),
            ('no props at all',
             f'CPU OV receiver: model = {MODEL}\nCPU OV receiver: topology L (shared CompiledModel)\n',
             'did not print'),
        ]
        for name, body, wanted in cases:
            with self.subTest(case=name):
                with self.assertRaises(BenchError) as caught:
                    check_ov_log(body, MODEL, 'C' if name == 'wrong topology echo' else 'L')
                self.assertIn(wanted, str(caught.exception))

    def test_worker_policy_must_hold_for_every_thread(self):
        body = log(workers=[worker(precision='bf16', matches=0)])
        with self.assertRaises(BenchError) as caught:
            check_ov_log(body, MODEL, 'L')
        self.assertIn('CPolicyMatches', str(caught.exception))

    def test_topology_c_is_accepted(self):
        result = check_ov_log(log(topology='C'), MODEL, 'C')
        self.assertEqual(result['worker']['topology'], 'C')


if __name__ == '__main__':
    unittest.main()
