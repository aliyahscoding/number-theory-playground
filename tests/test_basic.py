# Run with: python tests\test_basic.py
from number_playground import gcd, lcm, is_prime, sieve, factorize, fibonacci, pascal_row

def assert_equal(a, b):
    if a != b:
        raise AssertionError(f"Expected {b}, got {a}")

# 1) gcd
assert_equal(gcd(54, 24), 6)
assert_equal(gcd(-42, 56), 14)

# 2) lcm
assert_equal(lcm(21, 6), 42)
assert_equal(lcm(0, 7), 0)

# 3) primality
assert_equal(is_prime(97), True)
assert_equal(is_prime(1), False)

# 4) sieve
assert_equal(sieve(30), [2,3,5,7,11,13,17,19,23,29])

# 5) factorization
assert_equal(factorize(840), [(2,3),(3,1),(5,1),(7,1)])

# 6) fibonacci
assert_equal(fibonacci(10), [0,1,1,2,3,5,8,13,21,34])

# 7) Pascal row
assert_equal(pascal_row(0), [1])
assert_equal(pascal_row(5), [1,5,10,10,5,1])

print("All tests passed.")
