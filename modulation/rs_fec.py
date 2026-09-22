"""
modulation/rs_fec.py
====================
Reed-Solomon (RS) error-correcting code over GF(2^8).

Standard primitive polynomial: p(x) = x^8 + x^4 + x^3 + x^2 + 1 (0x11D = 285)
Primitive element: alpha = 2 (0x02)
Standard citation: CCSDS 131.0-B-3, NASA-GSFC, DVB-S, QR Code (ISO/IEC 18004).

Features:
- Arbitrary RS(n, k) support for n <= 255 (default: RS(64, 48), 2t = 16 parity bytes, t = 8 error correction).
- GF(2^8) log/antilog table-accelerated field arithmetic.
- Systematic polynomial division encoder.
- Berlekamp-Massey error locator polynomial solver.
- Chien search for error root finding.
- Forney algorithm for error magnitude calculation.
- Post-correction syndrome verification for robust failure detection.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Galois Field GF(2^8) Tables & Operations
# ---------------------------------------------------------------------------

PRIMITIVE_POLY = 0x11D  # 285: x^8 + x^4 + x^3 + x^2 + 1

gf_exp = [0] * 512
gf_log = [0] * 256

x = 1
for i in range(255):
    gf_exp[i] = x
    gf_log[x] = i
    x <<= 1
    if x & 0x100:
        x ^= PRIMITIVE_POLY

for i in range(255, 512):
    gf_exp[i] = gf_exp[i - 255]


def gf_mul(x: int, y: int) -> int:
    """Multiply two GF(2^8) elements."""
    if x == 0 or y == 0:
        return 0
    return gf_exp[gf_log[x] + gf_log[y]]


def gf_div(x: int, y: int) -> int:
    """Divide two GF(2^8) elements."""
    if y == 0:
        raise ZeroDivisionError("GF(2^8) division by zero")
    if x == 0:
        return 0
    return gf_exp[(gf_log[x] - gf_log[y]) % 255]


def gf_poly_mul(p: list[int], q: list[int]) -> list[int]:
    """Multiply two polynomials over GF(2^8)."""
    r = [0] * (len(p) + len(q) - 1)
    for j, b in enumerate(q):
        for i, a in enumerate(p):
            r[i + j] ^= gf_mul(a, b)
    return r


def rs_generator_poly(n_sym: int, fcr: int = 0) -> list[int]:
    """
    Construct generator polynomial g(x) = product_{i=0}^{n_sym-1} (x - alpha^(i + fcr)).
    """
    g = [1]
    for i in range(n_sym):
        g = gf_poly_mul(g, [1, gf_exp[i + fcr]])
    return g


# ---------------------------------------------------------------------------
# Reed-Solomon Encoder
# ---------------------------------------------------------------------------

def rs_encode(
    data_bytes: np.ndarray | bytes | list[int],
    n: int = 64,
    k: int = 48,
    fcr: int = 0,
) -> np.ndarray:
    """
    Systematically encode data bytes into an RS(n, k) codeword.

    Parameters
    ----------
    data_bytes : array-like of ints/bytes in range [0, 255]
        Length must be <= k. If len < k, zero-padded on the left or used as shortened code.
    n : int
        Codeword length in bytes (default: 64).
    k : int
        Message length in bytes (default: 48).
    fcr : int
        First consecutive root exponent (default: 0).

    Returns
    -------
    np.ndarray of uint8 (length n)
    """
    msg = list(data_bytes)
    if len(msg) != k:
        raise ValueError(f"Message length must be exactly {k} bytes (got {len(msg)}).")

    n_sym = n - k
    gen = rs_generator_poly(n_sym, fcr)

    out = list(msg) + [0] * n_sym
    for i in range(len(msg)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                out[i + j] ^= gf_mul(gen[j], coef)

    parity = out[len(msg):]
    return np.array(msg + parity, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Reed-Solomon Decoder
# ---------------------------------------------------------------------------

def rs_calc_syndromes(codeword: list[int], n_sym: int, fcr: int = 0) -> list[int]:
    """Calculate syndromes S_0 .. S_{n_sym-1}."""
    syndromes = []
    for i in range(n_sym):
        alpha_i = gf_exp[i + fcr]
        val = 0
        for c in codeword:
            val = gf_mul(val, alpha_i) ^ int(c)
        syndromes.append(val)
    return syndromes


def rs_find_error_locator(synd: list[int], n_sym: int) -> list[int]:
    """Find error locator polynomial Lambda(x) using the Berlekamp-Massey algorithm."""
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1

    for i in range(n_sym):
        d = synd[i]
        for j in range(1, L + 1):
            d ^= gf_mul(C[j], synd[i - j])

        if d == 0:
            m += 1
        elif 2 * L <= i:
            T = list(C)
            scale = gf_div(d, b)
            pad = [0] * m + [gf_mul(x, scale) for x in B]
            if len(pad) > len(C):
                C = C + [0] * (len(pad) - len(C))
            for k in range(len(pad)):
                C[k] ^= pad[k]
            L = i + 1 - L
            B = T
            b = d
            m = 1
        else:
            scale = gf_div(d, b)
            pad = [0] * m + [gf_mul(x, scale) for x in B]
            if len(pad) > len(C):
                C = C + [0] * (len(pad) - len(C))
            for k in range(len(pad)):
                C[k] ^= pad[k]
            m += 1

    while len(C) > 1 and C[-1] == 0:
        C.pop()
    return C


def rs_find_errors(err_loc: list[int], msg_len: int) -> list[int]:
    """Find error positions using Chien search."""
    err_pos = []
    num_errors = len(err_loc) - 1

    for i in range(msg_len):
        neg_p = (255 - (msg_len - 1 - i)) % 255
        val = 0
        for deg, c in enumerate(err_loc):
            val ^= gf_mul(c, gf_exp[(deg * neg_p) % 255])
        if val == 0:
            err_pos.append(i)

    if len(err_pos) != num_errors:
        raise ValueError(
            f"Uncorrectable error: Chien search root mismatch (found {len(err_pos)} roots for degree {num_errors})."
        )
    return err_pos


def rs_forney(synd: list[int], err_loc: list[int], err_pos: list[int], msg_len: int, fcr: int = 0) -> list[int]:
    """Calculate error magnitudes using the Forney algorithm."""
    if fcr not in (0, 1):
        raise NotImplementedError("rs_forney only verified for fcr in {0, 1}; test before using other values.")

    n_sym = len(synd)
    Omega = gf_poly_mul(synd, err_loc)[:n_sym]
    E = [0] * msg_len

    for pos in err_pos:
        p = (msg_len - 1 - pos) % 255
        neg_p = (255 - p) % 255
        X = gf_exp[p]

        # Evaluate Omega(X^-1)
        num = 0
        for deg, c in enumerate(Omega):
            num ^= gf_mul(c, gf_exp[(deg * neg_p) % 255])

        # Multiply by X^(1 - fcr)
        if fcr == 0:
            num = gf_mul(num, X)
        elif fcr != 1:
            num = gf_mul(num, gf_exp[((1 - fcr) * p) % 255])

        # Evaluate Lambda'(X^-1)
        denom = 0
        for deg in range(1, len(err_loc), 2):
            denom ^= gf_mul(err_loc[deg], gf_exp[((deg - 1) * neg_p) % 255])

        if denom == 0:
            raise ValueError("Uncorrectable error: zero derivative denominator in Forney algorithm.")

        err_val = gf_div(num, denom)
        E[pos] = err_val

    return E


def rs_decode(
    codeword_bytes: np.ndarray | bytes | list[int],
    n: int = 64,
    k: int = 48,
    fcr: int = 0,
) -> np.ndarray:
    """
    Decode an RS(n, k) codeword with error correction.

    Parameters
    ----------
    codeword_bytes : array-like of uint8
        Received codeword of length n.
    n : int
        Codeword length (default: 64).
    k : int
        Message length (default: 48).
    fcr : int
        First consecutive root exponent (default: 0).

    Returns
    -------
    np.ndarray of uint8 (length k)
        Decoded and corrected message bytes.

    Raises
    ------
    ValueError:
        If number of errors exceeds error-correcting capacity t = (n - k) // 2
        or cannot be reliably corrected.
    """
    cw = list(codeword_bytes)
    if len(cw) != n:
        raise ValueError(f"Codeword length must be exactly {n} bytes (got {len(cw)}).")

    n_sym = n - k
    synd = rs_calc_syndromes(cw, n_sym, fcr)

    # If all syndromes are zero, no errors
    if max(synd) == 0:
        return np.array(cw[:k], dtype=np.uint8)

    # 1. Error locator polynomial via Berlekamp-Massey
    err_loc = rs_find_error_locator(synd, n_sym)
    max_t = n_sym // 2
    if len(err_loc) - 1 > max_t:
        raise ValueError(
            f"Uncorrectable error: error locator degree {len(err_loc) - 1} exceeds max capacity {max_t}."
        )

    # 2. Chien search for error positions
    err_pos = rs_find_errors(err_loc, len(cw))

    # 3. Forney algorithm for error magnitudes
    E = rs_forney(synd, err_loc, err_pos, len(cw), fcr)

    # 4. Correct errors
    corrected = list(cw)
    for i in range(len(corrected)):
        corrected[i] ^= E[i]

    # 5. Post-correction syndrome verification
    post_synd = rs_calc_syndromes(corrected, n_sym, fcr)
    if max(post_synd) != 0:
        raise ValueError("Uncorrectable error: post-correction syndromes are non-zero.")

    return np.array(corrected[:k], dtype=np.uint8)
