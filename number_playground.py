#!/usr/bin/env python3
"""
Number Theory Playground
Functions: primality (trial + Miller–Rabin), gcd, lcm, sieve, factorization,
Fibonacci, Pascal row. CLI: prime, factor, sieve, gcd, lcm, fib, pascal.
"""

from __future__ import annotations
import argparse
import math
from typing import List, Tuple

# ----------------------------
# Core number theory functions
# ----------------------------

def gcd(a: int, b: int) -> int:
    """Greatest common divisor via Euclid."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """Least common multiple. lcm(0, x) = 0 by convention."""
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)

def _is_trivially_composite(n: int) -> bool:
    if n % 2 == 0 and n != 2:
        return True
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in small_primes:
        if n == p:
            return False
        if n % p == 0:
            return n != p
    return False

def _miller_rabin(n: int) -> bool:
    """
    Deterministic Miller–Rabin for 64-bit integers using known bases.
    Works for all n < 2^64 with these bases.
    """
    if n < 2:
        return False
    # Handle small primes fast
    small_primes = [2, 3, 5, 7, 11, 13, 17]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a: int, d: int, n: int, s: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    # Deterministic bases for 2^64
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if not check(a, d, n, s):
            return False
    return True

def is_prime(n: int, use_miller_rabin: bool = True) -> bool:
    """Primality test: trial division for small n, Miller–Rabin for larger."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Quick composite check with small primes
    if _is_trivially_composite(n):
        return False

    # If requested, do Miller–Rabin for speed on big n
    if use_miller_rabin and n >= 2_000_000:
        return _miller_rabin(n)

    # Otherwise trial divide up to sqrt(n)
    limit = int(math.isqrt(n))
    f = 3
    while f <= limit:
        if n % f == 0:
            return False
        f += 2
    return True

def sieve(n: int) -> List[int]:
    """Sieve of Eratosthenes, returns list of primes <= n."""
    if n < 2:
        return []
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    p = 2
    while p * p <= n:
        if is_prime_arr[p]:
            step = p
            start = p * p
            is_prime_arr[start : n + 1 : step] = [False] * ((n - start) // step + 1)
        p += 1
    return [i for i, flag in enumerate(is_prime_arr) if flag]

def factorize(n: int) -> List[Tuple[int, int]]:
    """
    Prime factorization by trial division.
    Returns list of (prime, exponent), ascending by prime.
    """
    if n == 0:
        return [(0, 1)]
    if n < 0:
        # include -1 factor to record sign
        return [(-1, 1)] + factorize(-n)

    factors = []
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    if count:
        factors.append((2, count))

    f = 3
    while f * f <= n:
        count = 0
        while n % f == 0:
            n //= f
            count += 1
        if count:
            factors.append((f, count))
        f += 2
    if n > 1:
        factors.append((n, 1))
    return factors

def format_factorization(factors: List[Tuple[int, int]]) -> str:
    """Pretty string like 840 = 2^3 * 3 * 5 * 7."""
    if not factors:
        return "1"
    parts = []
    for p, e in factors:
        if e == 1:
            parts.append(f"{p}")
        else:
            parts.append(f"{p}^{e}")
    return " * ".join(parts)

def fibonacci(k: int) -> List[int]:
    """First k Fibonacci numbers starting at 0, 1."""
    if k <= 0:
        return []
    if k == 1:
        return [0]
    seq = [0, 1]
    while len(seq) < k:
        seq.append(seq[-1] + seq[-2])
    return seq

def pascal_row(n: int) -> List[int]:
    """Nth row of Pascal's triangle (0-indexed)."""
    if n < 0:
        return []
    row = [1]
    for k in range(1, n + 1):
        # C(n, k) = C(n, k-1) * (n - k + 1) // k
        row.append(row[-1] * (n - k + 1) // k)
    return row

# -------------
# CLI interface
# -------------

def main():
    parser = argparse.ArgumentParser(
        description="Number Theory Playground CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prime = sub.add_parser("prime", help="Test primality of N")
    p_prime.add_argument("n", type=int)

    p_factor = sub.add_parser("factor", help="Factorize N")
    p_factor.add_argument("n", type=int)

    p_sieve = sub.add_parser("sieve", help="List primes <= N")
    p_sieve.add_argument("n", type=int)

    p_gcd = sub.add_parser("gcd", help="gcd(a, b)")
    p_gcd.add_argument("a", type=int)
    p_gcd.add_argument("b", type=int)

    p_lcm = sub.add_parser("lcm", help="lcm(a, b)")
    p_lcm.add_argument("a", type=int)
    p_lcm.add_argument("b", type=int)

    p_fib = sub.add_parser("fib", help="First k Fibonacci numbers")
    p_fib.add_argument("k", type=int)

    p_pascal = sub.add_parser("pascal", help="Nth row of Pascal's triangle (0-indexed)")
    p_pascal.add_argument("n", type=int)

    args = parser.parse_args()

    if args.cmd == "prime":
        n = args.n
        print(f"{n} is prime" if is_prime(n) else f"{n} is not prime")

    elif args.cmd == "factor":
        n = args.n
        f = factorize(n)
        print(f"{n} = {format_factorization(f)}")

    elif args.cmd == "sieve":
        n = args.n
        primes = sieve(n)
        print(" ".join(map(str, primes)))

    elif args.cmd == "gcd":
        print(gcd(args.a, args.b))

    elif args.cmd == "lcm":
        print(lcm(args.a, args.b))

    elif args.cmd == "fib":
        print(" ".join(map(str, fibonacci(args.k))))

    elif args.cmd == "pascal":
        print(" ".join(map(str, pascal_row(args.n))))

if __name__ == "__main__":
    main()
