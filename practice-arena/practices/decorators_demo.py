import time
import functools
import random
from typing import Callable, Any

# ======================================================================
# 1. Practice: Basic Decorator with Metadata Preservation
# ======================================================================

def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """A basic decorator that enforces a mock authentication check."""
    @functools.wraps(func)  # Copies __name__ and __doc__ from 'func' to 'wrapper'
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print("[Auth Protection] Intercepted call. Verifying session token...")
        # Simulating a simple authorization payload check
        is_authenticated = True 
        
        if not is_authenticated:
            raise PermissionError("Unauthorized! Valid session token missing.")
            
        print("[Auth Protection] Access granted. Executing target function.")
        return func(*args, **kwargs)
    return wrapper

# --- Application ---
@require_auth
def download_secure_report(report_id: int) -> str:
    """Downloads internal server metrics reports by identifier code."""
    return f"REPORT_DATA_FOR_{report_id}"

# --- Verification Check ---
print(f"Function Identifier: {download_secure_report.__name__}")
print(f"Documentation String: {download_secure_report.__doc__}")
print(f"Result: {download_secure_report(4004)}")


# ------------------------------------------------------------
# Execution Flow of the Authentication Decorator
# ------------------------------------------------------------

# Step 1:
# Python first defines the decorator function:
#
#     require_auth(func)
#
# This decorator is designed to:
#
#     • intercept function calls
#     • perform authentication checks
#     • optionally block execution
#     • execute the original function safely

# ------------------------------------------------------------

# Step 2:
# Python reaches:
#
#     @require_auth
#
# above:
#
#     download_secure_report()

# This is equivalent to:
#
#     download_secure_report =
#         require_auth(download_secure_report)

# The original function is passed into
# the decorator as:
#
#     func

# ------------------------------------------------------------

# Step 3:
# Inside require_auth():
#
# Python defines a nested function:
#
#     wrapper(*args, **kwargs)
#
# This wrapper becomes the NEW function
# that replaces the original function.

# ------------------------------------------------------------

# Step 4:
# functools.wraps(func) executes.
#
# This copies metadata from the original
# function onto the wrapper.
#
# Copied metadata includes:
#
#     • __name__
#     • __doc__
#     • module information
#
# Without functools.wraps:
#
#     download_secure_report.__name__
#
# would incorrectly become:
#
#     "wrapper"

# ------------------------------------------------------------

# Step 5:
# The decorator returns:
#
#     wrapper
#
# Therefore:
#
#     download_secure_report
#
# now points to the wrapper function,
# NOT directly to the original function.

# ------------------------------------------------------------

# Step 6:
# Python executes:
#
#     print(download_secure_report.__name__)
#
# Because functools.wraps copied metadata:
#
# Output:
#
#     "download_secure_report"

# ------------------------------------------------------------

# Step 7:
# Python executes:
#
#     print(download_secure_report.__doc__)
#
# functools.wraps preserved the original
# documentation string successfully.

# ------------------------------------------------------------

# Step 8:
# Python executes:
#
#     download_secure_report(4004)
#
# Since the function was decorated,
# Python actually calls:
#
#     wrapper(4004)

# ------------------------------------------------------------

# Step 9:
# Inside wrapper():
#
#     *args  -> (4004,)
#     **kwargs -> {}

# The wrapper intercepts execution BEFORE
# the original function runs.

# ------------------------------------------------------------

# Step 10:
# Authentication logic executes:
#
#     print("Verifying session token...")
#
# Simulated authorization:
#
#     is_authenticated = True

# ------------------------------------------------------------

# Step 11:
# Since authentication passed:
#
#     print("Access granted...")
#
# The wrapper now executes the original
# protected function using:
#
#     return func(*args, **kwargs)

# Equivalent to:
#
#     return download_secure_report(4004)

# ------------------------------------------------------------

# Step 12:
# The original function runs:
#
#     return f"REPORT_DATA_FOR_{report_id}"

# Returned value:
#
#     "REPORT_DATA_FOR_4004"

# ------------------------------------------------------------

# Step 13:
# The wrapper receives the returned value
# and passes it back to the caller.

# Final printed result:
#
#     REPORT_DATA_FOR_4004

# ------------------------------------------------------------

# Core Concept:
#
# Decorators allow you to inject behavior:
#
#     BEFORE function execution
#     AFTER function execution
#     AROUND function execution
#
# without modifying the original function itself.

# Common production uses:
#
#     • authentication
#     • authorization
#     • logging
#     • rate limiting
#     • caching
#     • timing metrics
#     • retry mechanisms
# ------------------------------------------------------------


# ======================================================================
# 2. Practice: Execution Time Logger Decorator
# ======================================================================


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Measures and logs the exact execution duration of a function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter() # High-resolution timer
        print(f"\n⏱️  [Timer Start] Running: '{func.__name__}'...")
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_duration = end_time - start_time
        print(f"⏱️  [Timer Stop] '{func.__name__}' completed in {execution_duration:.6f} seconds.")
        return result
    return wrapper

# --- Application ---
@log_execution_time
def complex_data_transformation(data_size: int) -> list[int]:
    # Simulate data processing latency
    time.sleep(0.3)
    return [i * 2 for i in range(data_size)]

# --- Running the Code ---
transformed_output = complex_data_transformation(5)


# ======================================================================
# 3. Practice: Decorator Factory with Parameters (@retry(times=3))
# ======================================================================

def retry(times: int, backoff_seconds: float = 0.5) -> Callable[..., Any]:
    """A decorator factory that retries a failing function a specific number of times."""
    # Layer 1: The Factory (captures 'times' and 'backoff_seconds')
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Layer 2: The Decorator (captures the actual function object)
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Layer 3: The Wrapper (contains execution and looping recovery logic)
            attempts = 0
            while attempts < times:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    attempts += 1
                    print(f"⚠️  [Attempt {attempts}/{times} Failed] '{func.__name__}' threw error: {error}")
                    
                    if attempts >= times:
                        print(f"🚨 [Fatal] '{func.__name__}' exceeded max retries. Raising error.")
                        raise error # Out of retries, escalate the error
                        
                    print(f"   Waiting {backoff_seconds}s before retrying...")
                    time.sleep(backoff_seconds)
        return wrapper
    return decorator

# --- Application ---
@retry(times=3, backoff_seconds=0.2)
def unstable_network_call() -> str:
    """Simulates a volatile network request that fails 70% of the time."""
    if random.random() > 0.3:
        raise ConnectionResetError("Remote server closed the channel abruptly.")
    return "SUCCESS_DATA_PAYLOAD"

# --- Running the Code ---
print("\n--- Starting Unstable Network Execution ---")
try:
    response = unstable_network_call()
    print(f"Final Pipeline Result: {response}")
except Exception:
    print("Application safely handled the operation failure scenario.")


# ------------------------------------------------------------
# Execution Flow of the Retry Decorator Factory
# ------------------------------------------------------------

# Step 1:
# Python first defines:
#
#     retry(times, backoff_seconds)
#
# This is NOT the actual decorator yet.
#
# It is a:
#
#     decorator factory
#
# because it accepts configuration values
# and returns a customized decorator.

# ------------------------------------------------------------

# Step 2:
# Python encounters:
#
#     @retry(times=3, backoff_seconds=0.2)
#
# This executes immediately during
# function definition time.

# Equivalent to:
#
#     decorator = retry(times=3, backoff_seconds=0.2)

# ------------------------------------------------------------

# Step 3:
# The factory function runs:
#
#     retry(3, 0.2)
#
# Inside this layer:
#
#     times = 3
#     backoff_seconds = 0.2
#
# These values become permanently captured
# inside a closure.

# The factory then returns:
#
#     decorator

# ------------------------------------------------------------

# Step 4:
# Python now applies the returned decorator
# to the target function:
#
#     unstable_network_call

# Equivalent to:
#
#     unstable_network_call =
#         decorator(unstable_network_call)

# ------------------------------------------------------------

# Step 5:
# Inside decorator():
#
# The original function object is received as:
#
#     func

# Python now defines:
#
#     wrapper(*args, **kwargs)
#
# This wrapper contains:
#
#     • retry logic
#     • exception handling
#     • recovery delays
#     • looping behavior

# ------------------------------------------------------------

# Step 6:
# functools.wraps(func) copies metadata
# from the original function onto wrapper.
#
# Preserves:
#
#     • __name__
#     • __doc__
#     • module info

# ------------------------------------------------------------

# Step 7:
# decorator() returns:
#
#     wrapper
#
# Therefore:
#
#     unstable_network_call
#
# now points to wrapper(),
# NOT the original function directly.

# ------------------------------------------------------------

# Step 8:
# Python executes:
#
#     response = unstable_network_call()
#
# Since decoration occurred,
# Python actually calls:
#
#     wrapper()

# ------------------------------------------------------------

# Step 9:
# Inside wrapper():
#
#     attempts = 0
#
# The retry loop begins:
#
#     while attempts < times

# Since:
#
#     times = 3
#
# Maximum retry count is 3.

# ------------------------------------------------------------

# Step 10:
# wrapper() attempts:
#
#     return func(*args, **kwargs)
#
# Equivalent to:
#
#     unstable_network_call()

# ------------------------------------------------------------

# Step 11:
# Inside unstable_network_call():
#
# Random probability check executes:
#
#     random.random() > 0.3
#
# This creates:
#
#     • 70% failure probability
#     • 30% success probability

# ------------------------------------------------------------

# Step 12A:
# SUCCESS PATH
#
# If the random check passes:
#
#     return "SUCCESS_DATA_PAYLOAD"
#
# wrapper() immediately returns success.
#
# The retry loop ends completely.

# Final Output:
#
#     Final Pipeline Result: SUCCESS_DATA_PAYLOAD

# ------------------------------------------------------------

# Step 12B:
# FAILURE PATH
#
# If the random check fails:
#
#     raise ConnectionResetError(...)
#
# Control jumps into:
#
#     except Exception as error

# ------------------------------------------------------------

# Step 13:
# Retry recovery logic executes:
#
#     attempts += 1
#
# Failure message prints:
#
#     [Attempt X/3 Failed]

# ------------------------------------------------------------

# Step 14:
# Python checks:
#
#     if attempts >= times
#
# Two possibilities exist.

# ------------------------------------------------------------

# Step 15A:
# RETRIES STILL AVAILABLE
#
# If attempts remain:
#
#     print("Waiting before retry...")
#
# Then:
#
#     time.sleep(backoff_seconds)
#
# pauses execution briefly.
#
# Delay value:
#
#     0.2 seconds
#
# The loop then retries the function again.

# ------------------------------------------------------------

# Step 15B:
# MAX RETRIES EXCEEDED
#
# If all retries fail:
#
#     print("[Fatal] exceeded max retries")
#
# Then:
#
#     raise error
#
# re-raises the final exception.

# ------------------------------------------------------------

# Step 16:
# The outer try/except block catches
# the final unhandled failure:
#
#     except Exception:
#
# Preventing the application from crashing.

# Final output:
#
#     Application safely handled
#     the operation failure scenario.

# ------------------------------------------------------------
# Core Architecture Concept
# ------------------------------------------------------------

# This pattern demonstrates:
#
#     • decorator factories
#     • closures
#     • retry recovery systems
#     • exponential/backoff style protection
#     • resilient execution pipelines
#
# Common production use cases:
#
#     • unstable API calls
#     • database reconnection
#     • distributed services
#     • cloud networking
#     • AI inference retries
#     • transient infrastructure failures

# ------------------------------------------------------------
# Three Nested Layers
# ------------------------------------------------------------

# Layer 1:
#
#     retry(...)
#
# Captures configuration values.

# Layer 2:
#
#     decorator(func)
#
# Captures the target function.

# Layer 3:
#
#     wrapper(*args, **kwargs)
#
# Executes runtime retry logic.

# ------------------------------------------------------------