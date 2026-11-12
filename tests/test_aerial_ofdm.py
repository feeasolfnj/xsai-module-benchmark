"""Host-side tests for the CPU to Aerial fixture and timing records."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from aerial_ofdm import BINS, POINTS, SYMBOLS, capacity_summary, placement_summary, prepare
from bench_support import BenchError


class AerialOfdmTests(unittest.TestCase):
    def test_prepare_copies_cp_and_normalizes_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            iq = root / 'iq.bin'
            manifest = root / 'metadata.json'
            out = root / 'fixture'
            sample = struct.pack('<hh', 8, -4)
            symbol = sample + bytes((POINTS - 2) * 4) + sample
            iq.write_bytes(symbol * SYMBOLS)
            digest = hashlib.sha256(iq.read_bytes()).hexdigest()
            manifest.write_text(json.dumps({'status': 'passed', 'metric': 'dft',
                                             'input': {'batch14_sha256': digest}}))
            args = SimpleNamespace(iq=iq, oai_manifest=manifest, out=out)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            fixture = json.loads((out / 'fixture.json').read_text())
            self.assertEqual(fixture['slot_samples'], 30720)
            self.assertEqual(fixture['useful_offsets'][:2], [176, 2368])
            slot = (out / 'time_slot_le_f32.bin').read_bytes()
            for offset in fixture['useful_offsets']:
                self.assertEqual(struct.unpack_from('<ffff', slot, 8 * (offset - 1)), (8, -4, 8, -4))
            reference = (out / 'freq_reference_le_f64.bin').read_bytes()
            re, im = struct.unpack_from('<dd', reference, 16 * (BINS // 2))
            self.assertAlmostEqual(re, 16 / math.sqrt(POINTS))
            self.assertAlmostEqual(im, -8 / math.sqrt(POINTS))
            iq.write_bytes(bytes(len(symbol) * SYMBOLS))
            with self.assertRaisesRegex(BenchError, 'SHA-256'):
                prepare(args)

    def test_capacity_and_placement_records(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), samples=2, requests=2, warmup=1)
            trace = """[AERIAL_WAVE] 0 7
[AERIAL_REQUEST] 0 0 4
[AERIAL_REQUEST] 0 1 7
[AERIAL_WAVE] 1 8
[AERIAL_REQUEST] 1 0 5
[AERIAL_REQUEST] 1 1 8
[AERIAL_STEADY] 2 16
"""
            result = capacity_summary(trace, args)
            self.assertEqual(result['sustained_slots_per_second'], 250000)
            self.assertEqual(result['request_response']['p95_us'], 8)
            self.assertTrue((args.out / 'responses.csv').is_file())
            with self.assertRaisesRegex(BenchError, 'per-request'):
                capacity_summary(trace.replace('[AERIAL_REQUEST] 1 1 8\n', ''), args)
            placement = placement_summary('[AERIAL_PLACEMENT] 0 0 1 2 3 4 5 6\n'
                                          '[AERIAL_PLACEMENT] 1 0 1 2 3 4 5 5\n', args)
            self.assertEqual(placement['distinct_sms_per_launch'], {6: 1, 7: 1})
