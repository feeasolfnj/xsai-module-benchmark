import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cycle_support import cycle_result

VALID = ('[PMU_STATUS] status=available reason=none errno=0\n'
         '[PMU_CYCLES] windows=2 units=10 cycles=1000 enabled_ns=500 running_ns=500 '
         'cycles_per_unit=100 empty_window_cycles=12 raw=1 scaled=0 inherited=1 exclude_kernel=1 exclude_hv=1')


class CycleResultTests(unittest.TestCase):
    def test_valid_hardware_report_is_not_frequency_converted(self):
        r = cycle_result(VALID)
        self.assertEqual(r['status'], 'available')
        self.assertEqual(r['cycles'], 1000)
        self.assertEqual(r['cycles_per_unit'], 100)
        self.assertFalse(r['instrumentation_overhead_subtracted'])

    def test_unavailable_is_null_not_zero(self):
        r = cycle_result('[PMU_STATUS] status=unavailable reason=hardware_event_unavailable errno=2')
        self.assertEqual(r['status'], 'unavailable')
        self.assertIsNone(r['cycles'])
        self.assertIsNone(json.loads(json.dumps(r))['cycles_per_unit'])

    def test_permission_denial_does_not_become_supported(self):
        r = cycle_result('[PMU_STATUS] status=unavailable reason=permission_denied errno=13')
        self.assertEqual(r['reason'], 'permission_denied')
        self.assertIsNone(r['cycles'])

    def test_missing_or_duplicate_reports_are_not_measurements(self):
        self.assertEqual(cycle_result('[RESULT] mean_us=3')['status'], 'unavailable')
        self.assertEqual(cycle_result(VALID + '\n' + VALID)['status'], 'invalid')
        self.assertEqual(cycle_result(VALID, instrumented=False)['reason'], 'instrumentation_not_built')

    def test_multiplexed_scaled_zero_or_inconsistent_counts_are_invalid(self):
        for bad in (VALID.replace('running_ns=500', 'running_ns=400'),
                    VALID.replace('scaled=0', 'scaled=1'),
                    VALID.replace('cycles=1000', 'cycles=0'),
                    VALID.replace('cycles_per_unit=100 ', 'cycles_per_unit=200 '),
                    VALID.replace('cycles_per_unit=100 ', 'cycles_per_unit=nan '),
                    VALID.replace('exclude_kernel=1', 'exclude_kernel=0')):
            with self.subTest(log=bad):
                r = cycle_result(bad)
                self.assertEqual(r['status'], 'invalid')
                self.assertIsNone(r['cycles'])

    def test_off_does_not_claim_counters(self):
        self.assertEqual(cycle_result(VALID, mode='off')['status'], 'disabled')


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('cc'), 'Linux C compiler required')
class CounterStateTests(unittest.TestCase):
    """Synthetic syscall fixtures test failure policy; these are not hardware results."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.exe = pathlib.Path(cls.temp.name) / 'counter-test'
        source = pathlib.Path(cls.temp.name) / 'counter-test.c'
        source.write_text(r'''
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
static uint64_t count, enabled, running;
static int active;
static long test_syscall(long nr, ...) {
  const char *failure=getenv("TEST_FAILURE");
  if (failure && !strcmp(failure,"permission")) {errno=EACCES;return -1;}
  if (failure && !strcmp(failure,"missing")) {errno=ENOENT;return -1;}
  return 900;
}
static ssize_t test_read(int fd, void *out, size_t n) {
  if(active){count+=100;enabled+=10;running+=getenv("TEST_FAILURE")?5:10;}
  uint64_t data[]={count,enabled,running};memcpy(out,data,sizeof(data));return sizeof(data);
}
static int test_ioctl(int fd, unsigned long request, ...) {
  if(request==PERF_EVENT_IOC_ENABLE)active=1;
  /* Simulate an enabled window completed before the disable/read pair. */
  if(request==PERF_EVENT_IOC_DISABLE){count+=100;enabled+=10;running+=getenv("TEST_FAILURE")?5:10;active=0;}
  return 0;
}
static int test_close(int fd){return 0;}
#define syscall test_syscall
#define read test_read
#define ioctl test_ioctl
#define close test_close
#include "pmu_cycles.h"
int main(void){if(getenv("TEST_NO_WINDOWS"))return 0;pmu_begin();pmu_end(10);return 0;}
'''.replace('#include <stdarg.h>', '#include <stdarg.h>\n#include <string.h>'))
        subprocess.run(['cc', '-I' + str(ROOT / 'src'), str(source), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_counter(self, mode='auto', failure=None, no_windows=False):
        env = os.environ.copy()
        for name in ('TEST_FAILURE', 'TEST_NO_WINDOWS'):
            env.pop(name, None)
        env['BENCH_CYCLES'] = mode
        if failure:
            env['TEST_FAILURE'] = failure
        if no_windows:
            env['TEST_NO_WINDOWS'] = '1'
        return subprocess.run([str(self.exe)], env=env, capture_output=True, text=True)

    def test_available_reports_measured_raw_cycles(self):
        p = self.run_counter()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(cycle_result(p.stdout)['status'], 'available')

    def test_auto_missing_and_permission_keep_program_running(self):
        for failure, reason in [('missing', 'hardware_event_unavailable'), ('permission', 'permission_denied')]:
            p = self.run_counter(failure=failure)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(cycle_result(p.stdout)['reason'], reason)
            self.assertNotIn('[PMU_CYCLES]', p.stdout)

    def test_required_missing_fails_without_abort_core_dump(self):
        p = self.run_counter(mode='required', failure='missing')
        self.assertEqual(p.returncode, 2)
        self.assertIn('Hardware cycles required', p.stderr)

    def test_invalid_counter_discards_partial_measurement(self):
        p = self.run_counter(failure='multiplexed')
        self.assertEqual(p.returncode, 0)
        self.assertEqual(cycle_result(p.stdout)['status'], 'invalid')
        self.assertNotIn('[PMU_CYCLES]', p.stdout)

    def test_off_works_even_if_hardware_is_missing(self):
        p = self.run_counter(mode='off', failure='missing')
        self.assertEqual(p.returncode, 0)
        self.assertEqual(cycle_result(p.stdout)['status'], 'disabled')

    def test_required_without_measurements_fails_cleanly(self):
        p = self.run_counter(mode='required', no_windows=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn('no_valid_measurements', p.stdout)


if __name__ == '__main__':
    unittest.main()
