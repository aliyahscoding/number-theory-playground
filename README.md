# Number Theory Playground

[![CI](https://github.com/aliyahscoding/number-theory-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/aliyahscoding/number-theory-playground/actions/workflows/ci.yml)


Tiny Python CLI for classic number theory utilities: primality testing, gcd/lcm, sieve of Eratosthenes, prime factorization, Fibonacci sequence, and Pascal’s triangle rows.

## Features
- `prime N` — primality test (trial division with optional deterministic Miller–Rabin for large N)
- `factor N` — prime factorization by trial division
- `sieve N` — list all primes ≤ N (Sieve of Eratosthenes)
- `gcd A B`, `lcm A B`
- `fib K` — first K Fibonacci numbers (starting at 0, 1)
- `pascal N` — Nth row of Pascal’s triangle (0-indexed)

## Quickstart

```bash
# create and activate venv (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Run commands:
python number_playground.py prime 97
python number_playground.py factor 840
python number_playground.py sieve 100
python number_playground.py gcd 462 1071
python number_playground.py lcm 21 6
python number_playground.py fib 12
python number_playground.py pascal 7


Tests
Simple assert-based checks:
python tests/test_basic.py
Passing run prints All tests passed.

Notes
Deterministic Miller–Rabin uses bases valid for all 64-bit integers.
lcm(0, x) = 0 by convention.