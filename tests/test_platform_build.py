import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from platform_build import source_root, host_arch, compile_flags, module_link, check_library_arch, adapt_statistics, link_command
from bench_support import BenchError


class PlatformBuildTests(unittest.TestCase):
    def test_ninja_linker_shell_wrappers_and_quoted_paths(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            (root / 'CMakeCache.txt').write_text('CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/ninja\n')
            result = subprocess.CompletedProcess([], 0, ': && c++ "source with spaces.o" -o nr_ulsim libPHY_NR.a && :\n', '')
            with patch('subprocess.run', return_value=result):
                self.assertEqual(link_command(root, 'nr_ulsim'), ['c++', 'source with spaces.o', '-o', 'nr_ulsim', 'libPHY_NR.a'])
            result.returncode = 1
            result.stderr = 'unknown target nr_ulsim'
            with patch('subprocess.run', return_value=result), self.assertRaisesRegex(BenchError, 'Cannot read Ninja'):
                link_command(root, 'nr_ulsim')

    def test_optional_statistics_do_not_remove_computation(self):
        text = ('start_meas(&gNB->ulsch_channel_measurement_stats);\n'
                'nr_gnb_measurements(gNB);\n'
                'merge_meas(&gNB->ulsch_resource_extraction_stats, &arr[i].resource_extraction_stats);\n'
                'merge_meas(&gNB->ulsch_channel_compensation_stats, &arr[i].channel_compensation_stats);\n'
                'stop_meas(&gNB->ulsch_channel_measurement_stats);')
        adapted = adapt_statistics(text, '')
        self.assertEqual(adapted.count('(void)0;'), 4)
        self.assertIn('nr_gnb_measurements(gNB);', adapted)
        header = 'ulsch_channel_measurement_stats ulsch_resource_extraction_stats ulsch_channel_compensation_stats'
        self.assertEqual(adapt_statistics(text, header), text)

    def test_tensorrt_skipped_inference_is_not_a_passing_benchmark(self):
        from bench import check_neural_log
        for message in ('TensorRT: failed to deserialize engine',
                        'TensorRT: execution context unavailable; skipping receiver decode',
                        'IExecutionContext::enqueueV3: Error Code 1: Cuda Driver',
                        'TensorRT: Error Code: 4',
                        'TensorRT: no engine available; cannot create execution context'):
            with self.subTest(message=message), self.assertRaises(BenchError):
                check_neural_log(message + '\n[RESULT] verification=passed', 'tensorrt')
        check_neural_log('Initialized TRT receiver\n[RESULT] verification=passed', 'tensorrt')

    def test_architecture_aliases(self):
        self.assertEqual(host_arch('AMD64'), 'x86_64')
        self.assertEqual(host_arch('arm64'), 'aarch64')
        with self.assertRaises(BenchError):
            host_arch('armv7l')

    def test_srk_root_resolves_embedded_oai(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            header = root / 'ext/openairinterface5g/openair1/PHY/defs_gNB.h'
            header.parent.mkdir(parents=True)
            header.write_text('')
            (root / 'ext/openairinterface5g/CMakeLists.txt').write_text('')
            self.assertEqual(source_root(root), root.resolve() / 'ext/openairinterface5g')

    def test_native_compile_flags_with_ccache_and_spaces(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            entry = dict(directory=tmp, file='/source/a.c', arguments=[
                '/usr/bin/ccache', '/usr/bin/cc', '-Irelative path', '-march=armv8-a',
                '-o', 'CMakeFiles/PHY_NR.dir/a.c.o', '-c', '/source/a.c'])
            (root / 'compile_commands.json').write_text(json.dumps([entry]))
            flags = compile_flags(root, 'PHY_NR')
            self.assertEqual(flags, ['-I' + str(root / 'relative path'), '-march=armv8-a'])
            self.assertFalse(any('rv64' in x for x in flags))

    def test_makefile_flags_preserve_local_simd(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = pathlib.Path(tmp)
            path = root / 'CMakeFiles/PHY_NR.dir/flags.make'
            path.parent.mkdir(parents=True)
            path.write_text('C_DEFINES = -DFOO\nC_INCLUDES = -I/source\nC_FLAGS = -mavx2 -O2\n')
            self.assertEqual(compile_flags(root, 'PHY_NR'), ['-DFOO', '-I/source', '-mavx2', '-O2'])

    def test_link_drops_simulator_main_and_unneeded_globals(self):
        cmd = module_link(['c++', '-rdynamic', 'CMakeFiles/nr_ulsim.dir/a/ulsim.c.o',
                           'CMakeFiles/nr_ulsim.dir/softmodem-common.c.o',
                           'CMakeFiles/physim_common.dir/common.c.o', '-o', 'nr_ulsim', 'libPHY_NR.a'],
                          '/bench/module.o', '/bench/module')
        self.assertNotIn('-rdynamic', cmd)
        self.assertNotIn('CMakeFiles/nr_ulsim.dir/a/ulsim.c.o', cmd)
        self.assertIn('CMakeFiles/physim_common.dir/common.c.o', cmd)
        self.assertIn('libPHY_NR.a', cmd)
        self.assertEqual(cmd.count('-o'), 1)

    def test_reject_foreign_elf_before_build(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = pathlib.Path(tmp) / 'libdfts.so'
            data = bytearray(20)
            data[:4] = b'\x7fELF'
            data[5] = 1
            data[18:20] = (243).to_bytes(2, 'little')
            p.write_bytes(data)
            check_library_arch(p, 'riscv64')
            with self.assertRaisesRegex(BenchError, 'does not match'):
                check_library_arch(p, 'aarch64')


if __name__ == '__main__':
    unittest.main()
