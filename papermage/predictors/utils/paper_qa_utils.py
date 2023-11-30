"""

@soldni

"""

import re
from functools import lru_cache
from hashlib import blake2b
from typing import List

import numpy as np

MIN_WORD_LENGTH = 4
HASH_BITS = 64


def int_to_bin(i: int) -> np.ndarray:
    return np.array(list(map(int, np.binary_repr(i, width=64))), dtype=np.int8)


def bin_to_int(b: np.ndarray) -> int:
    return int("".join(map(str, b)), 2)


def int_to_hex(i: int) -> str:
    return hex(i)[2:]


def hex_to_int(h: str) -> int:
    return int(h, 16)


def generate_tokens(s: str, min_tokens: int = HASH_BITS, min_length: int = MIN_WORD_LENGTH) -> List[str]:
    """Generate tokens and sample a subset if you have enough of minimum length."""
    tokens = [t.strip() for t in re.findall(r" ?[^.\s,!?…。，、।۔،]+", s)]
    lengths = [len(t) for t in tokens]
    for n in range(min_length, -1, -1):
        if sum(lt for lt in lengths if lt > n) >= min_tokens:
            return [t for t in tokens if len(t) > n]
    return tokens


def _hash_token(token: str) -> np.ndarray:
    token_bytes = blake2b(token.encode(), digest_size=8).digest()
    return np.frombuffer(token_bytes, dtype=">u8").astype(np.int64)


@lru_cache(maxsize=2**14)
def hash_token(token: str, hash_bits: int = HASH_BITS):
    # we use blake2b because it's faster than md5, sha1, and sha3
    h = _hash_token(token=token)

    # initialize an array of -1
    bit_array = -1 * np.ones(hash_bits, dtype=np.int64)

    # find the indices where condition h & (1 << np.arange(hash_bits)) is true
    indices = np.where(h & (1 << np.arange(hash_bits, dtype=np.int64)))

    # set those indices to 1
    bit_array[indices] = 1

    # return the bit array
    return bit_array


def create_hash(s: str, hash_bits: int = HASH_BITS) -> int:
    tokens = generate_tokens(s=s, min_tokens=hash_bits)
    v = np.zeros(hash_bits, dtype=np.int64)

    for t in tokens:
        v = v + hash_token(token=t, hash_bits=hash_bits)

    # Get the indices where v[i] is >= 0
    indices, *_ = np.where(v >= 0)

    # Calculate fingerprint using bitwise shift and summation (equivalent to sum(2**indices))
    fingerprint = np.sum(1 << indices.astype(np.uint64))

    return fingerprint.tolist()


def similarity(targets: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(targets.shape) == 1:
        targets = targets.reshape(1, -1)
    if len(queries.shape) == 1:
        queries = queries.reshape(1, -1)
    return 1.0 - np.sum(queries[:, np.newaxis, :] != targets[np.newaxis, :, :], axis=-1) / queries.shape[-1]
