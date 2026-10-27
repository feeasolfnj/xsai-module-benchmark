import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_support import BenchError, decoder_quality, records


class DecoderQualityTests(unittest.TestCase):
    def test_excludes_warmup_and_checks_crc_and_original_payload(self):
        log = '\n'.join(
            f'[CODING_UL] trial={i} input_mode=encoded_bpsk_awgn decode_success={ok} '
            f'crc_valid={ok} payload_match={ok} uncoded_bit_errors=5 coded_bits=100'
            for i, ok in enumerate([0, 1, 1]))
        _, measured = records(log, warmup=1)
        q = decoder_quality(measured)
        self.assertEqual((q['measured_trials'], q['succeeded'], q['failed']), (2, 2, 0))
        self.assertEqual(q['bler'], 0)
        self.assertEqual(q['uncoded_ber'], 0.05)

    def test_low_snr_failures_and_false_positives_remain_visible(self):
        rows = [dict(record_type='CODING_UL', phase='measurement', input_mode='encoded_bpsk_awgn',
                     decode_success='0', crc_valid='1', payload_match='0',
                     uncoded_bit_errors='50', coded_bits='100')]
        q = decoder_quality(rows)
        self.assertEqual((q['failed'], q['false_positives'], q['bler']), (1, 1, 1))

    def test_legacy_or_inconsistent_records_are_not_quality_results(self):
        base = dict(record_type='CODING_UL', phase='measurement', input_mode='encoded_bpsk_awgn',
                    decode_success='1', crc_valid='0', payload_match='0',
                    uncoded_bit_errors='0', coded_bits='100')
        with self.assertRaises(BenchError):
            decoder_quality([base])
        with self.assertRaises(BenchError):
            decoder_quality([dict(base, input_mode='direct_random_llr')])

    def test_no_measured_trials_cannot_claim_success(self):
        with self.assertRaises(BenchError):
            decoder_quality([])
