"""Host C harness tests verification modes independently of an installed XSAI."""
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
        subprocess.run(['gcc','-UNDEBUG','-I'+str(ROOT/'src'),str(ROOT/'src/bench.c'),str(source),'-o',str(cls.binary)],check=True,capture_output=True)

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
