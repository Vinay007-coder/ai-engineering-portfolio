# ------------------------------------------------------------------
# 1. Practice: Prime Number Generator (yield vs return)
# ------------------------------------------------------------------

# This example demonstrates why `yield` is essential for creating
# stateful, infinite pipelines.

# If you used `return`, the function would exit after producing
# the first number and forget all previous state.

# With `yield`, the function pauses instead of terminating.
# It keeps its internal memory of what has already been checked
# and continues seamlessly from where it left off.

def generate_primes():
    """An infinite generator that yields prime numbers one by one."""
    yield 2  # The only even prime
    primes_found = [2]
    candidate = 3
    
    while True:
        # Check if the candidate is divisible by any prime found so far
        is_prime = True
        for p in primes_found:
            if p * p > candidate:  # Optimization step
                break
            if candidate % p == 0:
                is_prime = False
                break
                
        if is_prime:
            primes_found.append(candidate)
            yield candidate  # Pauses execution and returns the prime, keeping primes_found in memory
            
        candidate += 2  # Skip even numbers

# --- Consuming the Generator ---
prime_stream = generate_primes()

print("--- First 5 Primes ---")
for _ in range(5):
    print(next(prime_stream))  # Explicitly pulling the next value


# --------------------------------------------------------------
# 2. Practice: Delegating with yield from
# --------------------------------------------------------------

# Instead of manually looping over sub-generators with nested for statements, 
# `yield from` clean-channels data streams by shifting delivery to a sub-generator directly.

def stream_system_logs():
    yield "LOG: System booting up..."
    yield "LOG: Core drivers loaded."

def stream_application_logs():
    yield "LOG: Application server running on port 8000."
    yield "LOG: Database sync operational."

def master_log_aggregator():
    yield "=== STARTING BOOT STRAP ==="
    yield from stream_system_logs()      # Delegates entirely to the first sub-generator
    
    yield "=== STARTING APPLICATION ==="
    yield from stream_application_logs()  # Delegates entirely to the second sub-generator
    
    yield "=== AGGREGATION COMPLETE ==="

# Print combined output
print("\n--- Aggregated Logs ---")
for log_line in master_log_aggregator():
    print(log_line)

