"""Host C harness tests verification modes independently of an installed XSAI."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('gcc') and os.name=='posix', 'POSIX host C compiler required')
class DftProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT)
        tmp=pathlib.Path(cls.tmp.name)
        source=tmp/'fake.c'
        source.write_text('''#include "bench_module.h"
#include <stdlib.h>
#include <string.h>
int latency_verification_status(void){return -1;}
static int setup(size_t n,int inv){return 0;}
static void ref(int16_t*x,int16_t*y,unsigned char scale){memcpy(y,x,8192);}
static void run(int16_t*x,int16_t*y,unsigned char scale){ref(x,y,scale);if(getenv("FAKE_MISMATCH"))y[0]^=1;}
const bench_module selected_module={"fake","test only",setup,run,0};
const bench_module reference_module={"fake-ref","test only",setup,ref,0};
''')
        cls.binary=tmp/'runner'
        subprocess.run(['gcc','-pthread','-UNDEBUG','-I'+str(ROOT/'src'),str(ROOT/'src/bench.c'),str(source),'-o',str(cls.binary)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, mode, mismatch):
        env=os.environ.copy()
        env.update(BENCH_VERIFY=mode,BENCH_INPUT_BANK='2',BENCH_WARMUP='1',BENCH_ITERATIONS='3',BENCH_BATCHES='2')
        if mismatch:env['FAKE_MISMATCH']='1'
        return subprocess.run([str(self.binary),'dft','1'],env=env,text=True,capture_output=True)

    def test_strict_rejects_mismatch(self):
        self.assertEqual(self.run_case('strict',True).returncode,2)

    def test_report_preserves_failure_marker(self):
        result=self.run_case('report',True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('verification=failed',result.stdout)

    def test_skip_does_not_claim_correctness(self):
        result=self.run_case('skip',True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('verification=skipped',result.stdout)

    def test_pass_uses_batch_mean_name(self):
        result=self.run_case('strict',False)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('verification=passed',result.stdout)
        self.assertIn('median_batch_mean_us=',result.stdout)

    def test_slot_iq_is_reproducible_and_verification_is_strict(self):
        if not hasattr(os, 'sched_getaffinity'):
            self.skipTest('Linux affinity required')
        cpus = sorted(os.sched_getaffinity(0))
        from bench_support import records, summary

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'iq14_le_i16.bin'
            env = os.environ.copy()
            env.pop('FAKE_MISMATCH', None)
            env.update(BENCH_INPUT_BANK='14', BENCH_WARMUP='1', BENCH_BATCHES='2',
                       BENCH_SLOT_CPU0=str(cpus[0]), BENCH_SLOT_IQ=str(path), BENCH_VERIFY='strict')
            for workers in (1, 2) if len(cpus) >= 3 else (1,):
                with self.subTest(workers=workers):
                    env['BENCH_SLOT_WORKERS'] = str(workers)
                    if workers == 2:
                        env.update(BENCH_SLOT_CPU1=str(cpus[1]), BENCH_SLOT_COORD_CPU=str(cpus[2]))
                    command = [str(self.binary), 'dft', '20261111']
                    result = subprocess.run(command, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    raw, measured = records(result.stdout, warmup=1)
                    self.assertEqual([r['phase'] for r in raw], ['warmup', 'measurement', 'measurement'])
                    self.assertTrue(all(r['verification'] == 'passed' and r['symbols'] == '14'
                                        and r['workers'] == str(workers) for r in measured))
                    times = summary(measured)['DFT_SLOT']
                    self.assertAlmostEqual(times['amortized_symbol_us'] * 14, times['slot_mean_us'])
                    data = path.read_bytes()
                    self.assertEqual(len(data), 14 * 2048 * 4)
                    self.assertEqual(hashlib.sha256(data).hexdigest(),
                                     'dcd68e88b07bd6f6e34d4dce9677043eb6aeaf61a5b1ee1d7a0a22ca7a883171')
                    failed = subprocess.run(command, env=dict(env, FAKE_MISMATCH='1'),
                                            text=True, capture_output=True)
                    self.assertEqual(failed.returncode, 2)
                    self.assertIn('differs from the serial reference', failed.stderr)
