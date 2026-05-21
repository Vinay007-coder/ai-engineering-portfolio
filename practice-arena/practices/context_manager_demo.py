# Remember your core design philosophy: If a resource requires a manual "close", "disconnect", or "stop", it belongs inside a context manager.

# --------------------------------------------------------------
# 1. Practice: Custom @contextmanager using try/yield/finally
# --------------------------------------------------------------

# ------------------------------------------------------------
# Why Function Decorator Context Managers Are Useful
# ------------------------------------------------------------

# The function-decorator context manager pattern
# is useful for wrapping execution logic around
# a protected block of code.

# Common use cases include:
#
#     • measuring execution time
#     • setting temporary state variables
#     • recording diagnostic logs
#     • managing transactions
#     • opening and closing resources safely

# ------------------------------------------------------------

# The critical design rule:
#
#     The `yield` statement must be placed
#     inside a `try/finally` block.

# Why?
#
# Because the code inside the `with` block
# may crash unexpectedly.

# Without `finally`:
#
#     • cleanup code may never execute
#     • resources may leak
#     • sockets/files may remain open
#     • temporary state may become corrupted

# ------------------------------------------------------------

# The `finally` block guarantees cleanup runs:
#
#     • whether the block succeeds
#     • whether an exception occurs
#     • whether execution exits early

# General structure:
#
#     try:
#         yield resource
#
#     finally:
#         cleanup()

# ------------------------------------------------------------

# This creates reliable lifecycle management:
#
#     setup  -> yield control -> guaranteed cleanup
#
# Similar to how production systems safely manage:
#
#     • database sessions
#     • AI inference streams
#     • file handles
#     • network connections
#     • async pipelines
# ------------------------------------------------------------

import time
from contextlib import contextmanager

@contextmanager
def pipeline_performance_tracker(stage_name: str):
    """A functional context manager that monitors pipeline processing performance."""
    print(f"\n[METRICS START] Entering processing stage: '{stage_name}'")
    start_time = time.time()
    
    try:
        # Pause right here, hand control over to the indented 'with' block
        yield 
    finally:
        # This code is completely guaranteed to run, even if the block crashes
        end_time = time.time()
        duration = end_time - start_time
        print(f"[METRICS STOP] Stage '{stage_name}' finished in {duration:.4f} seconds.")

# --- Running the Practice Code ---
with pipeline_performance_tracker("LLM Embedding Generation"):
    print("  [Worker] Batching tokens into numerical vectors...")
    time.sleep(0.4)  # Simulate processing delay
    print("  [Worker] Matrix operations successful.")


# --------------------------------------------------------------
# 2. Practice: Class-Based Context Manager (__enter__ / __exit__)
# --------------------------------------------------------------

# ------------------------------------------------------------
# When to Use a Class-Based Context Manager
# ------------------------------------------------------------

# A class-based context manager is ideal when
# the resource manager needs to maintain
# internal operational state across execution.

# Use a class-based approach when you need:
#
#     • active internal state tracking
#     • persistent communication channels
#     • complex lifecycle coordination
#     • reusable resource management logic
#     • configurable multi-parameter objects

# ------------------------------------------------------------

# Common real-world examples:
#
#     • database connection pools
#     • network socket managers
#     • AI model cache systems
#     • distributed message brokers
#     • streaming APIs
#     • authentication sessions

# ------------------------------------------------------------

# Why classes work well here:
#
#     • state can persist inside `self`
#     • multiple methods can share resources
#     • configuration can be stored cleanly
#     • setup and teardown logic remain organized
#
# Example internal state:
#
#     self.is_active
#     self.connection
#     self.timeout
#     self.buffer
#     self.session_token

# ------------------------------------------------------------

# Unlike lightweight decorator-based managers,
# class-based managers are better suited for:
#
#     • long-running resources
#     • complex cleanup workflows
#     • mutable shared state
#     • advanced error handling
#
# They provide stronger structure for
# production-scale systems and backend services.
# ------------------------------------------------------------

class ModelCacheConnection:
    """Simulates managing a persistent memory channel for caching models."""
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.is_active = False

    def __enter__(self):
        print(f"\n[__enter__] Initializing high-speed connection socket to: {self.endpoint}")
        self.is_active = True
        # Whatever is returned here is bound to the variable after the 'as' keyword
        return self  

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[__exit__] Flushing data buffers for: {self.endpoint}")
        self.is_active = False
        print("[__exit__] Connection channel closed cleanly.")
        
        # If an error happens inside the block, handle it here
        if exc_type is not None:
            print(f"  ⚠️ Intercepted an error within the block: {exc_val}")
            # Return True to swallow the error, or False to let it raise normally
            return False 

    def fetch_cached_weights(self, model_id: str) -> str:
        if not self.is_active:
            raise RuntimeError("Cannot fetch data. Connection is closed.")
        return f"WEIGHT_TENSOR_FOR_{model_id}"

# --- Running the Practice Code ---
with ModelCacheConnection("localhost:6379") as cache:
    print(f"  [Cache Working] State active check: {cache.is_active}")
    tensor_data = cache.fetch_cached_weights("llama-3-8b")
    print(f"  [Cache Working] Data recovered: {tensor_data}")


# ==========================================================
# 3. Structural Comparison: Where to Use Which
# ==========================================================

# ------------------------------------------------------------
# Choosing the Correct Context Manager Pattern
# ------------------------------------------------------------

# To build reliable production systems,
# it is important to understand exactly
# where each context manager pattern fits best.

# ============================================================
# Use Case A:
# Timing & Logging Blocks
# (Use @contextmanager)
# ============================================================

# Scenario:
#
#     • measuring API execution time
#     • timing database lookups
#     • temporarily switching logger levels
#     • enabling debug mode briefly
#
# Example:
#
#     INFO  -> DEBUG -> INFO

# Why this pattern fits:
#
#     • no persistent resource object exists
#     • no long-lived connection is maintained
#     • no complex internal state is required
#
# The goal is simply to wrap
# a block of execution between:
#
#     setup action
#           ↓
#       execute code
#           ↓
#     cleanup action

# This makes the lightweight
# @contextmanager decorator ideal.

# ------------------------------------------------------------

# Typical responsibilities:
#
#     • start timers
#     • stop timers
#     • record metrics
#     • adjust temporary configuration
#     • restore previous settings
#
# These are short-lived operational wrappers,
# not full resource-management systems.

# ============================================================
# Use Case B:
# Pipeline Resource Cleanup
# (Use Class-Based Context Manager)
# ============================================================

# Scenario:
#
#     • opening raw CSV files
#     • connecting to AWS S3 storage
#     • locking file system directories
#     • managing network sockets
#     • handling streaming pipelines
#
# These systems manage real computing assets.

# Why this pattern fits:
#
#     • persistent resources must stay active
#     • internal operational state is required
#     • connections must remain stable
#     • cleanup must be guaranteed safely
#
# The class stores and manages:
#
#     • file descriptors
#     • socket handles
#     • authentication sessions
#     • active stream channels
#     • network clients

# ------------------------------------------------------------

# The class-based structure provides:
#
#     • reusable lifecycle control
#     • organized setup/teardown logic
#     • state persistence via `self`
#     • safer failure recovery
#     • structured cleanup workflows

# ------------------------------------------------------------

# Most importantly:
#
#     Resources are always released correctly,
#     preventing:
#
#         • memory leaks
#         • socket leaks
#         • locked files
#         • dangling connections
#         • corrupted pipeline state

# ============================================================
# Core Production Rule
# ============================================================

# Use @contextmanager when:
#
#     • wrapping behavior temporarily
#     • managing lightweight execution framing
#     • no persistent resource state exists

# Use class-based managers when:
#
#     • managing real system resources
#     • maintaining active internal state
#     • coordinating long-lived connections
#     • handling complex cleanup safely
# ------------------------------------------------------------