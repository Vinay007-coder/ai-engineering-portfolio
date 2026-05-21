import asyncio
import time
import random
import sys
from contextlib import asynccontextmanager

# Goal
# Understand:

# coroutine
# await
# event loop
# non-blocking pause

# ----------------------------
# Practice Code 1
# ----------------------------

# async def say_hello():
#     print("Helloo started")
#     await asyncio.sleep(2)
#     print("Helloo finished")

# async def main():
#     print("Main started")
#     await say_hello()
#     print("Main finished")   


# ------------------------------------------------------------
# Execution Timeline
# ------------------------------------------------------------

# 1. asyncio.run(main())
#    - Python creates a brand-new Event Loop.
#    - The main() coroutine is scheduled to run.

# 2. Inside main()
#    - print("Main started") executes immediately.
#
#      Output:
#      Main started

# 3. await say_hello()
#    - main() pauses at this line.
#    - Control jumps into the say_hello() coroutine.

# 4. Inside say_hello()
#    - print("Helloo started") executes immediately.
#
#      Output:
#      Helloo started

# 5. await asyncio.sleep(2)
#    - say_hello() pauses here.
#    - It tells the Event Loop:
#      "I don't need CPU time for 2 seconds.
#       Run something else if available."

# 6. Event Loop Behavior
#    - Since no other tasks exist,
#      the Event Loop simply waits (idles)
#      for 2 seconds.

# 7. Resuming say_hello()
#    - After 2 seconds, the Event Loop wakes up
#      say_hello().
#
#    - print("Helloo finished") executes.
#
#      Output:
#      Helloo finished

# 8. Returning Control to main()
#    - say_hello() finishes completely.
#    - Control returns to main() exactly where it paused.

# 9. main() Continues
#    - print("Main finished") executes.
#
#      Output:
#      Main finished

# 10. Teardown
#     - main() completes.
#     - asyncio.run() safely shuts down
#       the Event Loop.
 
# -----------------------------
# Practice Code 2
# -----------------------------

# --------------------------------------------
# This is a Sequential Flow
# --------------------------------------------

# async def fetch_user():
#     print("Fetching users")
#     await asyncio.sleep(2)
#     print("Users fetched")
#     return {"name": "Alice", "age": 30}

# async def fetch_orders():
#     print("Fetching orders")
#     await asyncio.sleep(3)
#     print("Orders fetched")
#     return [{"id": 1, "item": "Book"}, {"id": 2, "item": "Pen"}]

# async def main():
#     start_time = time.time()
#     print("Main started")
#     user = await fetch_user()
#     orders = await fetch_orders()
#     print(f"User: {user}")
#     print(f"Orders: {orders}")
#     print(f"\nMain finished in {time.time() - start_time:.2f} seconds")

# ------------------------------------------------------------
# Here's the execution timeline for this code:
# ------------------------------------------------------------

# [ main() starts ]
#        │
#        ▼
# await fetch_user()   --> 1. Program PAUSES here.
#        │                 2. Jumps into fetch_user() and waits 2 seconds.
#        │                 3. fetch_user() finishes completely and returns data.
#        ▼
# await fetch_orders() --> 4. ONLY NOW does this function start.
#                          5. Program PAUSES here again.
#                          6. Jumps into fetch_orders() and waits 3 seconds.
#                          7. fetch_orders() finishes completely and returns data.


# We have two ways to make this Sequential flow into Concurrent flow:
# 1. Using gather() -> Old way and not recommended
# 2. Using TaskGroup -> New way and recommended

# Using gather()
# async def main():
#     start_time = time.time()
#     print("Main started\n")
#     user, orders = await asyncio.gather(fetch_user(), fetch_orders())
#     print(f"\nUser: {user}")
#     print(f"Orders: {orders}")
#     print(f"\nMain finished in {time.time() - start_time:.2f} seconds")

# Using TaskGroup()
# async def main():
#     start_time = time.time()
#     print("Main started\n")
#     async with asyncio.TaskGroup() as tg:
#         user_task = tg.create_task(fetch_user())
#         orders_task = tg.create_task(fetch_orders())
    
#     user = user_task.result()
#     orders = orders_task.result()

#     print(f"\nUser: {user}")
#     print(f"Orders: {orders}")
#     print(f"\nMain finished in {time.time() - start_time:.2f} seconds")


# ---------------------------------------------
# Async Generator
# ---------------------------------------------

# This is streaming. Crtical for AI systems.

# Goal
# Understand:

# yield
# async for
# streaming
# incremental output

# -----------------------------
# Practice Code 3
# -----------------------------


# Simulated ChatGPT backend generating a response token-by-token
# async def chat_gpt_stream():
#     response = ["Artificial", " Intelligence", " is", " changing", " the", " world."]
    
#     for word in response:
#         await asyncio.sleep(0.4) # Simulate the LLM thinking/generating time
#         yield word               # Pause, send this word to the client, preserve state

# async def main():
#     start_time = time.time()
#     print("User asked: What is AI?")
#     print("ChatGPT Bot: ", end="", flush=True)
    
#     # We consume the stream using 'async for'
#     async for chunk in chat_gpt_stream():
#         print(chunk, end="", flush=True)
        
#     print(f"\n\nStream finished in {time.time() - start_time:.2f} seconds.")



# ----------------------------
# Async Context Manager
# ----------------------------
# Goal

# Understand:

# setup/cleanup lifecycle
# resource management
# automatic cleanup

# -----------------------------
# Practice Code 3
# -----------------------------

# class AsyncDatabaseClient:
#     def __init__(self, db_name: str):
#         self.db_name = db_name
#         self.connected = False

#     # 1. Setup phase when entering the block
#     async def __aenter__(self):
#         print(f"[__aenter__] Connecting to database '{self.db_name}'...")
#         await asyncio.sleep(1)  # Simulate network handshake latency
#         self.connected = True
#         print(f"[__aenter__] Successfully connected to '{self.db_name}'!")
#         return self  # This object is assigned to the variable after 'as'

#     # 2. Teardown phase when exiting the block
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         print(f"\n[__aexit__] Closing database connection to '{self.db_name}'...")
#         await asyncio.sleep(0.5)  # Simulate asynchronous disconnection
#         self.connected = False
#         print("[__aexit__] Connection closed cleanly.")
        
#         # If an exception happened inside the block, return False to let it raise,
#         # or True to swallow/ignore the error.
#         return False 

#     # A mock business logic method
#     async def execute_query(self, query: str):
#         if not self.connected:
#             raise RuntimeError("Cannot execute query. Not connected to database.")
#         print(f"[Query] Executing: '{query}'")
#         await asyncio.sleep(0.5)  # Simulate query execution time
#         return {"status": "success", "rows": 3}

# async def main():
#     print("--- Script Started ---")
    
#     # Using the async context manager
#     async with AsyncDatabaseClient(db_name="Production_DB") as db:
#         print("[Block] Inside the context block now.")
#         result = await db.execute_query("SELECT * FROM users LIMIT 3;")
#         print(f"[Block] Query results received: {result}")
#         print("[Block] Reached the end of the context block.")

#     print("\n--- Script Finished ---")


# Step-by-Step Execution Flow

# [ asyncio.run(main()) ]
#            │
#            ▼
# [ Enter async with block ] ──────► Triggers __aenter__()
#                                       │
#                                       ▼
#                                   await asyncio.sleep(1)
#                                   (Pauses here; lets event loop handle other tasks)
#                                       │
#                                       ▼
#                                   Returns 'self' and assigns it to 'db'
#            │
#            ▼
# [ Inside the Block ]      ──────► Runs db.execute_query()
#                                   Prints results
#            │
#            ▼
# [ Exit the Block ]        ──────► Triggers __aexit__() automatically
#                                       │
#                                       ▼
#                                   await asyncio.sleep(0.5)
#                                   (Pauses here to disconnect safely)
#            │
#            ▼
# [ Main Continues ]        ──────► Prints "Script Finished"

# ------------------------------------------------------------
# Execution Timeline
# ------------------------------------------------------------

# 1. Entering the Context
#
#    - The program reaches:
#
#          async with AsyncDatabaseClient() as db
#
#    - Python immediately calls:
#
#          __aenter__()
#
#    - This begins the asynchronous setup phase.

# 2. Async Waiting During Setup
#
#    - Inside __aenter__(), execution reaches:
#
#          await asyncio.sleep(1)
#
#    - The coroutine pauses.
#
#    - Control is returned to the Event Loop for 1 second.
#
#    - During this pause, the Event Loop could run
#      other tasks if they existed.

# 3. Binding the Variable
#
#    - After the 1-second sleep completes,
#      __aenter__() resumes execution.
#
#    - It returns:
#
#          self
#
#    - Python assigns this returned object
#      to the variable:
#
#          db

# 4. Executing the Block Logic
#
#    - The indented block inside async with now runs.
#
#    - The program calls:
#
#          await db.execute_query()
#
#    - Inside execute_query(), execution pauses again
#      at:
#
#          await asyncio.sleep(0.5)
#
#    - The Event Loop handles this async wait smoothly.

# 5. Automatic Cleanup Trigger
#
#    - As soon as the async with block finishes
#      (or even if an exception occurs),
#      Python automatically enters:
#
#          __aexit__()
#
#    - This guarantees cleanup logic always runs.

# 6. Async Waiting During Teardown
#
#    - Inside __aexit__(), execution reaches:
#
#          await asyncio.sleep(0.5)
#
#    - Cleanup pauses asynchronously.
#
#    - The Event Loop manages this wait efficiently.

# 7. Cleanup Completion
#
#    - After the wait completes:
#
#          connection closed
#
#      cleanup finishes successfully.
#
#    - The async context manager fully exits.

# Start the async program


# ------------------------------------------------------------
# Real-World Scenario
# ------------------------------------------------------------

# We will build a:
#
#   Concurrent, Rate-Limited Web Scraper
#   with a Global Timeout
#
# ------------------------------------------------------------
# Features Used
# ------------------------------------------------------------

# 1. Semaphore
#
#    - Limits concurrency.
#
#    - Ensures only a maximum of 2 websites
#      are scraped simultaneously.
#
#    - Prevents:
#         • IP bans
#         • server overload
#         • excessive parallel requests

# 2. Async Context Manager
#
#    - Cleanly opens and closes
#      the network session.
#
#    - Uses:
#
#          __aenter__()
#          __aexit__()
#
#    - Guarantees cleanup even if
#      unexpected errors occur.

# 3. TaskGroup
#
#    - Launches multiple scraping jobs
#      concurrently.
#
#    - Provides structured concurrency.
#
#    - Automatically handles:
#         • task coordination
#         • cancellation propagation
#         • cleanup safety

# 4. Async Generator
#
#    - Streams scraped results
#      incrementally.
#
#    - Returns data token-by-token
#      or result-by-result using:
#
#          yield
#
#    - Avoids waiting for all tasks
#      to finish before sending output.

# 5. Timeout Handling
#
#    - Applies a timeout to
#      individual requests.
#
#    - Prevents hanging operations
#      from blocking the entire system.
#
#    - Example:
#
#          asyncio.timeout(...)

# 6. Cancellation Safety
#
#    - Handles unexpected cancellations
#      gracefully.
#
#    - Ensures:
#         • resources are cleaned up
#         • connections close safely
#         • tasks terminate properly

# ------------------------------------------------------------
# Overall Goal
# ------------------------------------------------------------

# Build a robust asynchronous scraping system that:
#
#    • runs multiple jobs concurrently
#    • respects rate limits
#    • streams live results
#    • handles failures safely
#    • cleans up resources automatically
#    • remains responsive under load


# 1. ASYNC CONTEXT MANAGER (Handles lifecycle of our mock HTTP Session)
@asynccontextmanager
async def mock_http_session(session_name: str):
    print(f"[Session] Initializing network socket pool for '{session_name}'...")
    await asyncio.sleep(0.5)  # Simulate async connection setup
    try:
        yield f"CLIENT_SESSION_{session_name}"
    finally:
        print(f"[Session] Releasing and closing network sockets for '{session_name}'...")
        await asyncio.sleep(0.2)  # Simulate async teardown

# 2. WORKER JOB (Uses Semaphore, Timeout Handling, and handles Cancellation)
async def scrape_website(url: str, semaphore: asyncio.Semaphore):
    # Use semaphore to limit concurrency rate
    async with semaphore:
        print(f"[Scraper] Acquired slot. Starting download for: {url}")
        
        try:
            # Apply a strict 2.0-second timeout to this specific network request
            async with asyncio.timeout(2.0):
                # Simulating dynamic network latency
                # "slow-site.com" will purposefully trigger the 2-second timeout
                delay = 4.0 if "slow-site" in url else random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)
                
                print(f"[Scraper] Finished downloading data from: {url}")
                return {"url": url, "status": 200, "size_kb": random.randint(10, 150)}
                
        except asyncio.TimeoutError:
            print(f"[Warning] Timeout reached! Aborted network call for: {url}")
            return {"url": url, "status": 408, "error": "Timeout"}
            
        except asyncio.CancelledError:
            # 4. CANCELLATION HANDLING (Ensures state integrity if TaskGroup shuts down)
            print(f"[Cleanup] Scraper task for {url} was cancelled mid-flight!")
            raise # Always re-raise CancelledError to let the loop manage closure

# 3. ASYNC GENERATOR (Streams results concurrently using a TaskGroup)
async def concurrent_streamer(urls: list[str]):
    # 6. SEMAPHORE (Allow a maximum of 2 tasks to execute concurrently)
    sem = asyncio.Semaphore(2)
    
    async with mock_http_session("ScrapeSession") as session:
        # 5. TASKGROUP (Manages structured concurrent jobs)
        async with asyncio.TaskGroup() as tg:
            tasks = []
            for url in urls:
                # Schedule them all instantly. Semaphore will throttle their internal execution.
                task = tg.create_task(scrape_website(url, sem))
                tasks.append(task)
            
            # As tasks finish, yield their results back immediately
            for finished_task in tasks:
                # Wait for each task sequentially, but remember they are executing concurrently
                result = await finished_task
                yield result

# Main orchestrator
async def main():
    target_urls = [
        "https://example.com",
        "https://slow-site.com",  # Will timeout
        "https://python.org",
        "https://github.com"
    ]
    
    print("--- Starting Application Streaming Workflow ---\n")
    
    # Consume the async generator stream
    async for stream_chunk in concurrent_streamer(target_urls):
        print(f"STREAM OUTPUT -> Received data chunk: {stream_chunk}\n")
        
    print("--- All Streams Finished Safely ---")

# --------------------------------
# Step-by-Step Flow Explanation
# --------------------------------

#                [ asyncio.run(main()) ]
#                           │
#                           ▼
#             [ Enter mock_http_session ]
#                           │
#                           ▼
#              [ Initialize TaskGroup ]
#         ┌─────────────────┴─────────────────┐
#   [Task 1: example]                 [Task 2: slow-site]   <-- (Only 2 pass Semaphore)
#         │                                   │
#         ▼                                   ▼
#   (Completes in ~1s)                (Hits 2s Timeout)
#         │                                   │
#         ▼                                   ▼
# [Yields Example Data]              [Yields Timeout Error Object]
#         │                                   │
#         ▼                                   ▼
#  [Task 3: python.org starts]        [Task 4: github starts] <-- (Slots freed up)


# ------------------------------------------------------------
# Execution Timeline Breakdown
# ------------------------------------------------------------

# 1. Setup Loop
#
#    - asyncio.run(main()) starts the application.
#
#    - Python creates a fresh Event Loop
#      and schedules main() for execution.

# 2. Context Manager Entrance
#
#    - Before any scraping begins,
#      the code enters:
#
#          mock_http_session
#
#    - The async generator pauses execution
#      at the:
#
#          yield
#
#      boundary while the session
#      remains active.
#
#    - The network/session resources
#      stay open throughout processing.

# 3. Throttling via Semaphore
#
#    - All 4 scraping tasks are added
#      to the TaskGroup immediately.
#
#    - However:
#
#          asyncio.Semaphore(2)
#
#      limits active concurrency to 2 tasks.
#
#    - Tasks 3 and 4 are blocked temporarily
#      and wait in queue.
#
#    - They only proceed after tasks 1 or 2
#      complete and release semaphore slots.

# 4. Timeout Interception
#
#    - example.com completes normally
#      within the allowed timeout window.
#
#    - slow-site.com exceeds:
#
#          asyncio.timeout(2.0)
#
#    - The timeout context manager:
#
#         • interrupts execution
#         • raises TimeoutError
#         • transfers control into:
#
#               except asyncio.TimeoutError
#
#    - The system recovers safely
#      without crashing the application.

# 5. Streaming via Async Generator
#
#    - Instead of collecting all results
#      into a large array first,
#      the application uses:
#
#          yield result
#
#    - Each dataset streams back
#      individually as soon as it is ready.
#
#    - This enables:
#
#         • lower memory usage
#         • real-time output
#         • progressive data delivery

# 6. Automatic Cleanup
#
#    - After all TaskGroup tasks complete,
#      execution exits:
#
#          mock_http_session
#
#    - This automatically triggers:
#
#          finally
#
#      cleanup logic.
#
#    - The simulated session/resources
#      close safely and predictably.

asyncio.run(main())


