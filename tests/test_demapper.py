import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import make_parser, source_signature
from official_demapper import junit_status


class DemapperLauncherTests(unittest.TestCase):
    def test_entries_and_cache_tracking(self):
        args = make_parser().parse_args(['run', 'demapper', '--threads', '1', '--re', '512'])
        self.assertEqual(args.re, 512)
        self.assertIn('random_demapper.c', source_signature('demapper'))
        args = make_parser().parse_args(['official-demapper-test', '--srk', '/srk'])
        self.assertEqual(args.command, 'official-demapper-test')

    def test_upstream_skips_and_no_tests_are_not_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = pathlib.Path(tmp) / 'junit.xml'
            for body, status in [('', 'failed_or_skipped'), ('<testcase/>', 'passed'),
                                  ('<testcase><skipped/></testcase>', 'failed_or_skipped'),
                                  ('<testcase><failure/></testcase>', 'failed_or_skipped')]:
                p.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
                self.assertEqual(junit_status(p)['status'], status)


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'), 'Linux C compiler required')
class DemapperCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        cls.root = pathlib.Path(cls.tmp.name)
        header = cls.root / 'plugins/neural_demapper/src/nr_demapper_defs.h'
        header.parent.mkdir(parents=True)
        header.write_text('''
#include <stdint.h>
typedef struct { int16_t r, i; } c16_t;
typedef int32_t(demapper_initfunc_t)(void);
typedef int32_t(demapper_shutdownfunc_t)(void);
typedef int(demapper_compute_llrfunc_t)(int32_t*,c16_t*,c16_t*,c16_t*,int16_t*,uint32_t,uint8_t,uint8_t);
''')
        plugin = cls.root / 'plugin.c'
        plugin.write_text('''
#include "plugins/neural_demapper/src/nr_demapper_defs.h"
#include <stdlib.h>
#include <string.h>
static int mode(const char *s) { const char *p=getenv("TEST_DEMAPPER_MODE"); return p&&!strcmp(p,s); }
int demapper_init(void) { return mode("init_error")?-1:0; }
int demapper_shutdown(void) { return mode("shutdown_error")?-1:0; }
/* Intentionally no init_thread: upstream loader permits this symbol to be absent. */
int demapper_compute_llr(int32_t *x,c16_t*a,c16_t*b,c16_t*c,int16_t*y,uint32_t n,uint8_t s,uint8_t q) {
  static int calls;
  if(mode("fallback")) return 0;
  for(unsigned i=0;i<4*n;i++) y[i]=mode("zero")?0:(int16_t)(1+(i%4));
  if(mode("overrun")) y[4*n]=0;
  if(mode("nondeterministic")) y[0]=++calls;
  return 1;
}
''')
        cls.lib = cls.root/'plugin.so'
        subprocess.run(['cc', '-shared', '-fPIC', '-I'+str(cls.root), str(plugin), '-o', str(cls.lib)], check=True)
        cls.exe = cls.root/'demapper'
        subprocess.run(['cc', '-O2', '-I'+str(cls.root), '-I'+str(ROOT/'src'),
                        str(ROOT/'src/random_demapper.c'), '-o', str(cls.exe), '-rdynamic', '-ldl', '-lm'], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def execute(self, re=600, mode='normal', bank=1):
        env = dict(os.environ, BENCH_DEMAPPER_LIB=str(self.lib), BENCH_CYCLES='off',
                   BENCH_WARMUP='2', BENCH_INPUT_BANK=str(bank), TEST_DEMAPPER_MODE=mode)
        return subprocess.run([str(self.exe), '20261022', '0.002', str(re)], env=env, capture_output=True, text=True)

    def test_tail_sizes_and_banks(self):
        for re in (1, 2, 24, 32, 512, 600, 1024):
            p = self.execute(re=re, bank=2)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('verification_scope=repeatability_same_implementation', p.stdout)
            self.assertIn('handled=1', p.stdout)

    def test_invalid_plugin_output_is_never_a_success(self):
        for mode in ('fallback', 'zero', 'overrun', 'nondeterministic', 'init_error', 'shutdown_error'):
            with self.subTest(mode=mode):
                p = self.execute(mode=mode)
                self.assertNotEqual(p.returncode, 0)
                self.assertNotIn('[RESULT]', p.stdout)

    def test_fixed_seed_produces_same_input(self):
        from bench_support import records
        a = records(self.execute().stdout, 0)[1]
        b = records(self.execute().stdout, 0)[1]
        a = next(r for r in a if r['record_type']=='RESULT')
        b = next(r for r in b if r['record_type']=='RESULT')
        self.assertEqual(a['input_hash'], b['input_hash'])
