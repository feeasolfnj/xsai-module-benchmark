import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_support import (BenchError, cpu_list, select_cpus, records, summary,
                           benchmark_lock, fixed_frequency)


class LauncherTests(unittest.TestCase):
    def test_sparse_cpu_set_and_unavailable_cpu(self):
        self.assertEqual(cpu_list('0,2,4-6,2'), [0, 2, 4, 5, 6])
        self.assertEqual(select_cpus('0,2', 8, {0, 2, 4}), [0, 2])
        with self.assertRaisesRegex(BenchError, 'unavailable'):
            select_cpus('0-7', 8, {0, 1})

    def test_bad_cpu_ranges(self):
        for value in ('', '-1', '3-1', '1-2-3', 'a', '1,'):
            with self.subTest(value=value), self.assertRaises(BenchError):
                cpu_list(value)

    def test_warmup_excluded_and_raw_preserved(self):
        log = '\n'.join(f'[CODING_UL] trial={i} wall_us={1000 if i<2 else 100} cpu_us=100' for i in range(5))
        raw, measured = records(log, 2)
        self.assertEqual(len(raw), 5)
        self.assertEqual(len(measured), 3)
        self.assertEqual(summary(measured)['CODING_UL']['wall_us'], 100)
        self.assertEqual(summary(measured)['CODING_UL']['cores'], 1)

    def test_result_and_pmu_are_not_treated_as_codec_trials(self):
        raw, rows = records('[RESULT] mean_us=12 verification=skipped\n[PMU_CYCLES] calls=20', 10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(raw[0]['verification'], 'skipped')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux flock')
    def test_busy_lock_has_actionable_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with benchmark_lock(pathlib.Path(tmp)/'test.lock'):
                with self.assertRaisesRegex(BenchError, 'Another benchmark'):
                    with benchmark_lock(pathlib.Path(tmp)/'test.lock'):
                        self.fail('second lock acquired')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux flock')
    def test_existing_read_only_lock_works(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=pathlib.Path(tmp)/'readonly.lock'
            path.write_text('previous sudo run')
            path.chmod(0o444)
            try:
                with benchmark_lock(path):
                    self.assertEqual(path.read_text(),'previous sudo run')
            finally:
                path.chmod(0o644)

    def test_frequency_preflight_does_not_partially_change_policies(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            for cpu in (0, 1):
                path = root/f'cpu{cpu}'/'cpufreq'
                path.mkdir(parents=True)
                for name, value in {'scaling_governor': 'userspace' if cpu==0 else 'schedutil',
                                    'scaling_available_frequencies':'1800000 2200000',
                                    'scaling_setspeed':'1800000','scaling_cur_freq':'1800000'}.items():
                    (path/name).write_text(value)
            with self.assertRaisesRegex(BenchError, 'userspace'):
                with fixed_frequency([0,1], 2200000, root):
                    self.fail('invalid governor accepted')
            self.assertEqual((root/'cpu0/cpufreq/scaling_setspeed').read_text(), '1800000')

    def test_frequency_is_restored_after_benchmark_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=pathlib.Path(tmp)/'cpu0/cpufreq';path.mkdir(parents=True)
            for name,value in {'scaling_governor':'userspace','scaling_available_frequencies':'1800000 2200000',
                               'scaling_setspeed':'1800000','scaling_cur_freq':'2200000'}.items():
                (path/name).write_text(value)
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                with fixed_frequency([0],2200000,pathlib.Path(tmp)):
                    self.assertEqual((path/'scaling_setspeed').read_text().strip(),'2200000')
                    raise RuntimeError('synthetic failure')
            self.assertEqual((path/'scaling_setspeed').read_text().strip(),'1800000')

    def test_help_does_not_require_linux_or_configuration(self):
        proc=subprocess.run([sys.executable,str(ROOT/'bench.py'),'run','--help'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIn('--input-bank',proc.stdout)

    def test_link_recipes_use_real_gcc_runtime(self):
        import json
        commands=json.loads((ROOT/'templates/commands.json').read_text())
        self.assertFalse(any('-lgcc_s_asneeded' in command for command in commands))


if __name__ == '__main__':
    unittest.main()
