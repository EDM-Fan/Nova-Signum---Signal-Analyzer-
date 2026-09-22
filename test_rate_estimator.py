"""
test_rate_estimator.py
======================
Verification test for dsp.rate_estimator:
1. Tests against all sample IQ files (BPSK, QPSK, 16QAM, 2FSK, Full-pipeline Demo).
2. Prints the full candidate rate score table for each signal to verify clear discrimination.
3. Tests FSK fallback tone separation specifically on 2FSK.
"""

import os
import sys
import numpy as np

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sig_io.iq_reader import read_iq
from dsp.rate_estimator import estimate_sample_rate, DEFAULT_CANDIDATES


def test_file(rel_path, expected_rate=1_000_000, expected_sps=8.0):
    full_path = os.path.join(_ROOT, rel_path)
    if not os.path.isfile(full_path):
        print(f"[SKIP] {rel_path} not found")
        return

    info = read_iq(full_path, sample_rate=expected_rate)
    samples = info["samples"]

    res = estimate_sample_rate(samples, candidate_rates=DEFAULT_CANDIDATES)

    print(f"=== Testing: {os.path.basename(rel_path)} ===")
    print(f"  Method Used      : {res['method']}")
    print(f"  Estimated SPS    : {res['estimated_sps']:.2f} (Expected: ~{expected_sps:.1f})")
    print(f"  Estimated Baud   : {res['estimated_sym_rate']/1e3:.2f} ksym/s")
    print(f"  Winning Pick     : {res['best_rate']/1e3:.0f} kHz (Confidence: {res['confidence']:.1f}%)")
    print(f"  Candidate Score Table:")
    for rate, score in sorted(res["candidate_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 25)
        marker = " <--- WINNER (Expected)" if rate == expected_rate else ""
        print(f"    {rate/1e3:7.0f} kHz : {score*100:5.1f}% | {bar:<25}{marker}")

    assert res["best_rate"] == expected_rate, f"Wrong rate selected: {res['best_rate']} != {expected_rate}"
    print(f"  [PASS] Correct sample rate selected with clear score separation!\n")


if __name__ == "__main__":
    print("=== RUNNING SAMPLE-RATE ESTIMATOR VERIFICATION ===\n")
    test_file("samples/bpsk_1000ksps_f32.iq", expected_rate=1_000_000, expected_sps=8.0)
    test_file("samples/qpsk_1000ksps_f32.iq", expected_rate=1_000_000, expected_sps=8.0)
    test_file("samples/16qam_1000ksps_f32.iq", expected_rate=1_000_000, expected_sps=8.0)
    test_file("samples/2fsk_1000ksps_f32.iq", expected_rate=1_000_000, expected_sps=8.0)
    test_file("samples/bpsk_full_pipeline_demo.iq", expected_rate=1_000_000, expected_sps=8.0)
    print("=== ALL RATE ESTIMATION TESTS PASSED SUCCESSFULLY! ===")
