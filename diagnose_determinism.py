import numpy as np
from sig_io.iq_reader import read_iq
from modulation.classifier import predict
from modulation.demodulator import demodulate
from modulation.correlator import find_sync, parse_sync_word

path = "samples/true_diagonal_bpsk_demo.iq"

for run in range(5):
    info = read_iq(path, sample_rate=1_000_000)
    samples = info["samples"]
    label, conf = predict(samples, info["sample_rate"] or 1_000_000)
    bits = demodulate(samples, info["sample_rate"] or 1_000_000, label)
    res = find_sync(bits, sync_pattern=parse_sync_word("0x352EF853"), modulation_type=label)
    print(f"run={run} label={label} conf={conf:.4f} offset={res['offset']} score={res['score']:.6f} payload_len={len(res['payload_bits'])}")