import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import amx_flags, check_amx_log, make_parser
from bench_support import BenchError


class AmxLauncherTests(unittest.TestCase):
    def test_backend_and_mode_are_explicit(self):
        args = make_parser().parse_args(['run', 'demapper', '--threads', '1',
                                         '--backend', 'amx', '--amx-mode', 'bf16_ref'])
        self.assertEqual(args.neural_backend, 'amx')
        self.assertEqual(args.amx_mode, 'bf16_ref')
        args = make_parser().parse_args(['run', 'demapper', '--threads', '1'])
        self.assertEqual(args.neural_backend, 'native')
        self.assertIsNone(args.amx_mode)
        self.assertEqual(args.demapper_input, 'independent')
        args = make_parser().parse_args(['run', 'demapper', '--threads', '1', '--demapper-input', 'phys'])
        self.assertEqual(args.demapper_input, 'phys')

    def test_cpu_flags_are_read_from_proc_cpuinfo(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = pathlib.Path(tmp) / 'cpuinfo'
            p.write_text('processor\t: 0\nflags\t\t: fpu amx_tile amx_bf16 avx512f\n')
            self.assertEqual(amx_flags(p), dict(amx_tile=True, amx_bf16=True))
            p.write_text('processor\t: 0\nflags\t\t: fpu avx512f\n')
            self.assertEqual(amx_flags(p), dict(amx_tile=False, amx_bf16=False))
            self.assertEqual(amx_flags(pathlib.Path(tmp) / 'missing'),
                             dict(amx_tile=False, amx_bf16=False))

    def test_silent_fallback_is_never_an_amx_result(self):
        good = '[AMX_STATS] source=demapper_cpu_amx_thread_stats mode=1 gate=0 calls=640 handled=640 fallback=0\n'
        stats = check_amx_log(good)
        self.assertEqual((stats['calls'], stats['handled'], stats['fallback']), (640, 640, 0))
        for body, wanted in [
                ('[AMX_STATS] source=demapper_cpu_amx_thread_stats mode=1 gate=0 calls=640 handled=0 fallback=640\n',
                 'P0 result, not AMX'),
                ('[AMX_STATS] unavailable reason=no_symbol\n', 'statistics unavailable'),
                ('[RESULT] calls=1 mean_us=1.0\n', 'did not report'),
                ('[AMX_STATS] source=demapper_cpu_amx_thread_stats mode=1 gate=0 calls=0 handled=0 fallback=0\n',
                 'no calls')]:
            with self.subTest(body=body):
                with self.assertRaises(BenchError) as caught:
                    check_amx_log(body)
                self.assertIn(wanted, str(caught.exception))


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'), 'Linux C compiler required')
class AmxHarnessTests(unittest.TestCase):
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
        stub = cls.root / 'amx_stub.c'
        stub.write_text('''
#include "plugins/neural_demapper/src/nr_demapper_defs.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static uint64_t calls, fallback;
static int enabled(void) { return getenv("STUB_AMX_STATS") != NULL; }
int demapper_init(void) { return 0; }
int demapper_shutdown(void) { return 0; }
int demapper_compute_llr(int32_t *x,c16_t*a,c16_t*b,c16_t*c,int16_t*y,uint32_t n,uint8_t s,uint8_t q) {
  (void)a;(void)b;(void)c;(void)s;(void)q;
  if (getenv("STUB_CHECK_PHYS")) {
    /* Mirrors the AMX adapter guard: |I/Q| <= 4*magnitude, magnitude > 0. */
    for (uint32_t i = 0; i < n; i++) {
      int16_t sr = ((int16_t *)x)[2 * i], si = ((int16_t *)x)[2 * i + 1];
      int16_t mr = ((int16_t *)a)[2 * i], mi = ((int16_t *)a)[2 * i + 1];
      if (mr <= 0 || mi <= 0) return 0;
      if (fabsf((float)sr / mr) > 4.0f || fabsf((float)si / mi) > 4.0f) return 0;
    }
  }
  calls += enabled() ? 1 : 0;
  if (getenv("STUB_AMX_FALLBACK")) fallback += 1;
  for (unsigned i = 0; i < 4 * n; i++) y[i] = (int16_t)(1 + (i % 4));
  return 1;
}
#ifndef NO_AMX_SYMBOL
int demapper_cpu_amx_thread_stats(uint64_t *out, int *gate, int *mode) {
  if (!enabled()) return 0;
  memset(out, 0, 8 * sizeof(uint64_t));
  out[0] = calls; out[1] = calls - fallback; out[2] = fallback;
  if (gate) *gate = 0;
  if (mode) *mode = 1;
  return 1;
}
#endif
''')
        cls.lib = cls.root / 'amx_stub.so'
        cls.nosym = cls.root / 'amx_stub_nosym.so'
        for out, extra in ((cls.lib, []), (cls.nosym, ['-DNO_AMX_SYMBOL'])):
            subprocess.run(['cc', '-shared', '-fPIC', '-I' + str(cls.root), *extra,
                            str(stub), '-o', str(out)], check=True)
        cls.exe = cls.root / 'demapper'
        subprocess.run(['cc', '-O2', '-I' + str(cls.root), '-I' + str(ROOT / 'src'),
                        str(ROOT / 'src/random_demapper.c'), '-o', str(cls.exe),
                        '-rdynamic', '-ldl', '-lm'], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def execute(self, require=True, stats=True, fallback=False, lib=None, protocol=None, check_phys=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('STUB_AMX_', 'STUB_CHECK_', 'BENCH_AMX_', 'BENCH_DEMAPPER_INPUT'))}
        env.update(BENCH_DEMAPPER_LIB=str(lib or self.lib), BENCH_CYCLES='off', BENCH_WARMUP='2')
        if require:
            env['BENCH_AMX_REQUIRED'] = '1'
        if stats:
            env['STUB_AMX_STATS'] = '1'
        if fallback:
            env['STUB_AMX_FALLBACK'] = '1'
        if protocol:
            env['BENCH_DEMAPPER_INPUT'] = protocol
        if check_phys:
            env['STUB_CHECK_PHYS'] = '1'
        return subprocess.run([str(self.exe), '20261022', '0.002', '600'], env=env,
                              capture_output=True, text=True)

    def test_phys_protocol_is_the_only_admissible_one(self):
        good = self.execute(protocol='phys', check_phys=True)
        self.assertEqual(good.returncode, 0, good.stdout[-500:])
        self.assertIn('protocol=phys', good.stdout)
        bad = self.execute(protocol='independent', check_phys=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn('[RESULT]', bad.stdout)

    def test_statistics_are_reported_and_parsed(self):
        p = self.execute()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('[AMX_STATS]', p.stdout)
        stats = check_amx_log(p.stdout)
        self.assertEqual(stats['fallback'], 0)
        self.assertGreater(stats['calls'], 0)

    def test_reported_fallback_invalidates_the_run(self):
        p = self.execute(fallback=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('fallback=', p.stdout)
        with self.assertRaises(BenchError) as caught:
            check_amx_log(p.stdout)
        self.assertIn('not AMX', str(caught.exception))

    def test_missing_symbol_fails_only_when_amx_is_required(self):
        required = self.execute(require=True, lib=self.nosym)
        self.assertNotEqual(required.returncode, 0)
        self.assertNotIn('[RESULT]', required.stdout)
        self.assertIn('does not export demapper_cpu_amx_thread_stats', required.stderr)
        optional = self.execute(require=False, lib=self.nosym)
        self.assertEqual(optional.returncode, 0, optional.stderr)
        self.assertIn('reason=no_symbol', optional.stdout)
        with self.assertRaises(BenchError):
            check_amx_log(optional.stdout)

    def test_unavailable_thread_statistics_fails_a_required_run(self):
        p = self.execute(require=True, stats=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('statistics are unavailable on the measuring thread', p.stderr)


if __name__ == '__main__':
    unittest.main()
