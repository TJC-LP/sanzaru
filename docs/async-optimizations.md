# Async Optimizations

## Overview

Sanzaru is fully asynchronous, utilizing `anyio` and `aiofiles` for non-blocking operations. This design enables high throughput under concurrent load and takes full advantage of Python 3.14's free-threading capabilities.

## Key Optimizations

### 1. Async File I/O (`aiofiles`)

**Where:** File read/write operations
**Benefit:** Non-blocking disk I/O allows concurrent operations without thread overhead

**Operations:**
- `create_video()`: Reading reference images
- `download_video()`: Streaming video chunks to disk
- `download_image()`: Writing decoded images
- All operations use `async_safe_open_file()` for centralized security

**Example:**
```python
async with async_safe_open_file(reference_file, "rb", "reference image") as f:
    file_content = await f.read()
    video = await client.videos.create(..., input_reference=file_content)
```

### 2. CPU-Bound Operations in Thread Pools (`anyio.to_thread`)

**Where:** Computationally expensive operations
**Benefit:** Offloads CPU work to thread pool, keeping event loop responsive

**Operations:**
- `prepare_reference_image()`: PIL image processing (resize, crop, convert, save)
- `download_image()`: Base64 decoding and PIL dimension reading

**Example:**
```python
def _process_image():
    img = Image.open(input_path)
    img = img.convert("RGB")
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    img.save(output_path, "PNG")
    return img.size

# Run in thread pool - multiple preps can execute concurrently
original_size = await anyio.to_thread.run_sync(_process_image)
```

### 3. Async Streaming Downloads

**Where:** `download_video()`
**Benefit:** True async streaming with no blocking, efficient memory usage

**Pattern:**
```python
async with client.with_streaming_response.videos.download_content(video_id, variant=variant) as response:
    async with async_safe_open_file(out_path, "wb", "video file") as f:
        async for chunk in response.response.aiter_bytes():
            await f.write(chunk)
```

## Performance Benchmarks

Run `python docs/benchmarks/async_benchmark.py` to see performance metrics.

**Typical speedup factors (from benchmark):**
- **Image preparations**: 8-10x speedup for concurrent resize operations
- **File writes**: 10-11x speedup for concurrent I/O
- **Base64 decoding**: 7-8x speedup for concurrent CPU work
- **Mixed workloads**: All operations run truly concurrently

These speedups are measured against sequential execution. Under real-world concurrent load (e.g., multiple MCP clients), the server maintains responsiveness by not blocking the event loop.

## Concurrency Patterns

### Multiple Image Preparations
```python
# All operations run concurrently without blocking each other
results = await asyncio.gather(
    prepare_reference_image("img1.png", "1280x720"),
    prepare_reference_image("img2.png", "1280x720"),
    prepare_reference_image("img3.png", "1280x720"),
)
```

### Parallel Downloads
```python
# Download multiple videos simultaneously
results = await asyncio.gather(
    download_video(video_id_1),
    download_video(video_id_2),
    download_video(video_id_3),
)
```

### Mixed Operations
```python
# Different operation types run concurrently
results = await asyncio.gather(
    create_image("prompt1"),
    create_video("prompt2"),
    download_video(completed_video_id),
    prepare_reference_image("ref.png", "1280x720"),
)
```

## Python 3.14 Free-Threading

With Python 3.14's experimental free-threading mode (PEP 703), the server can utilize true parallelism:

**Benefits:**
- CPU-bound operations (PIL, base64) execute in parallel across cores
- Even higher throughput for compute-heavy workloads
- Better utilization of multi-core systems

**To enable:**
```bash
# Install Python 3.14t (free-threading build)
python3.14t -m venv .venv
source .venv/bin/activate
uv sync
```

**Note:** The async optimizations provide immediate benefits on standard Python 3.10+, with additional gains on 3.14t.

## Architecture Benefits

1. **Responsive Server**: Long operations don't freeze other requests
2. **Scalable**: Multiple clients can process simultaneously
3. **Efficient Resource Usage**: No unnecessary blocking or thread creation
4. **Future-Proof**: Scales better as API usage grows

## Implementation Details

### Security Layer
All async file operations use centralized security utilities:

**`async_safe_open_file()`**
- Async context manager wrapping `aiofiles`
- Standardized error handling
- Optional symlink checking
- Path traversal protection via `validate_safe_path()`

**Thread Safety**
- PIL operations are isolated in thread pools
- Each operation gets its own thread from the pool
- No shared state between concurrent operations

### Error Handling
All async operations maintain the same error handling guarantees as sync code:
- File not found → ValueError with context
- Permission denied → ValueError with action description
- Path traversal → ValueError with security context
- Consistent error messages across sync and async paths

## Migration Notes

### What Changed
- Added `anyio>=4.0.0` and `aiofiles>=24.0.0` dependencies
- Created `async_safe_open_file()` for async file operations
- Wrapped PIL operations in `anyio.to_thread.run_sync()`
- Switched to streaming downloads with `with_streaming_response`

### What Stayed the Same
- All public APIs unchanged
- Security guarantees maintained
- Error messages consistent
- Test coverage maintained (90 tests pass)

### Backward Compatibility
✅ Fully backward compatible - all existing tool calls work identically

## See Also
- [Async Optimization Spec](roadmap/async-optimization-spec.md) - Original implementation plan
- [Benchmark Script](benchmarks/async_benchmark.py) - Performance measurement tool
- [CLAUDE.md](../CLAUDE.md) - Full project architecture
