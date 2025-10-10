# Async Optimization Specification

✅ **IMPLEMENTATION COMPLETE**

## Overview

This document outlines a plan to make the Sora MCP server fully asynchronous using `anyio` and `aiofiles`. Currently, several operations block the event loop, which limits throughput under heavy load. With Python 3.14's free-threading support, making these operations truly async will maximize concurrency.

## Current Blocking Operations

### File I/O (Needs `aiofiles`)

**`create_video`**
```python
with open(reference_file, "rb") as f:
    video = await client.videos.create(..., input_reference=f)
```
- Blocks on reading reference image from disk
- Affects: Videos with reference images only

**`download_video`**
```python
content.write_to_file(str(out_path))
```
- OpenAI SDK method that performs synchronous write
- Affects: All video downloads
- Note: May need custom async wrapper since this is SDK code

**`prepare_reference_image`**
```python
img = Image.open(input_path)
# ... processing ...
img.save(output_path, "PNG")
```
- Blocks on reading and writing image files
- Affects: All image resize operations

**`download_image`**
```python
with open(output_path, "wb") as f:
    f.write(image_bytes)
```
- Blocks on writing decoded image to disk
- Affects: All image downloads

### CPU-Bound Operations (Needs `anyio.to_thread`)

**`prepare_reference_image`**
```python
img = Image.open(input_path)
img = img.convert("RGB")
img = img.resize(...)
img = img.crop(...)
img.thumbnail(...)
img.save(...)
```
- All PIL operations are CPU-intensive
- Blocks event loop during resize/crop/convert operations
- Affects: All image resize operations

**`download_image`**
```python
image_bytes = base64.b64decode(image_base64)
img = Image.open(output_path)  # for dimension checking
```
- Base64 decoding can be expensive for large images
- PIL dimension reading is I/O + CPU
- Affects: All image downloads

## Proposed Solution

### Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
  "anyio>=4.0.0",
  "aiofiles>=24.0.0",
  # ... existing deps
]
```

### Refactoring Strategy

#### 1. `create_video` - Async Reference Image Reading

**Current:**
```python
with open(reference_file, "rb") as f:
    video = await client.videos.create(..., input_reference=f)
```

**Proposed:**
```python
async with aiofiles.open(reference_file, "rb") as f:
    file_content = await f.read()
    # OpenAI SDK accepts bytes or file-like, so pass bytes
    video = await client.videos.create(..., input_reference=file_content)
```

**Impact:** Non-blocking file reads for reference images

---

#### 2. `download_video` - Async Video Writing

**Current:**
```python
content.write_to_file(str(out_path))
```

**Proposed:**
```python
# Option A: If SDK returns bytes
video_bytes = await content.read()  # or similar
async with aiofiles.open(out_path, "wb") as f:
    await f.write(video_bytes)

# Option B: Wrap SDK method in thread
await anyio.to_thread.run_sync(content.write_to_file, str(out_path))
```

**Impact:** Non-blocking video file writes
**Note:** Need to check OpenAI SDK's `download_content()` return type

---

#### 3. `prepare_reference_image` - Fully Async Image Processing

**Current:** All synchronous PIL operations

**Proposed:**
```python
async def prepare_reference_image(...) -> PrepareResult:
    reference_image_path = get_path("reference")
    # ... validation (still sync, fast) ...

    # Wrap entire PIL workflow in thread pool
    def _process_image() -> tuple[tuple[int, int], str]:
        """Synchronous image processing in worker thread."""
        # Read image
        img = Image.open(input_path)
        original_size = img.size

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize based on mode (crop/pad/rescale)
        if resize_mode == "crop":
            # ... crop logic ...
        elif resize_mode == "pad":
            # ... pad logic ...
        else:
            # ... rescale logic ...

        # Write output
        img.save(output_path, "PNG")

        return original_size, output_filename

    # Run in thread pool
    original_size, final_filename = await anyio.to_thread.run_sync(_process_image)

    logger.info(...)
    return {...}
```

**Impact:**
- Non-blocking during expensive resize/crop operations
- Multiple image preps can run concurrently
- Biggest performance gain under load

---

#### 4. `download_image` - Async Base64 + File Write

**Current:**
```python
image_bytes = base64.b64decode(image_base64)
with open(output_path, "wb") as f:
    f.write(image_bytes)
img = Image.open(output_path)
size = img.size
```

**Proposed:**
```python
# Decode in thread pool (CPU-bound for large images)
image_bytes = await anyio.to_thread.run_sync(base64.b64decode, image_base64)

# Write with aiofiles
async with aiofiles.open(output_path, "wb") as f:
    await f.write(image_bytes)

# Get dimensions in thread pool
def _get_dimensions():
    img = Image.open(output_path)
    return img.size, img.format.lower() if img.format else "unknown"

size, output_format = await anyio.to_thread.run_sync(_get_dimensions)
```

**Impact:** Non-blocking base64 decode and file writes

---

## Testing Strategy

### Load Test Scenarios

**Scenario 1: Multiple concurrent image preparations**
```python
# Should not block each other
tasks = [
    prepare_reference_image("img1.png", "1280x720"),
    prepare_reference_image("img2.png", "1280x720"),
    prepare_reference_image("img3.png", "1280x720"),
]
results = await asyncio.gather(*tasks)
```

**Scenario 2: Parallel downloads**
```python
# Download multiple videos simultaneously
tasks = [
    download_video(video_id_1),
    download_video(video_id_2),
    download_video(video_id_3),
]
results = await asyncio.gather(*tasks)
```

**Scenario 3: Mixed operations**
```python
# Image gen + video creation + downloads all at once
tasks = [
    create_image("prompt1"),
    create_video("prompt2"),
    download_video(completed_video_id),
    prepare_reference_image("ref.png", "1280x720"),
]
results = await asyncio.gather(*tasks)
```

### Performance Metrics

Before/after measurements:
- Time to process 10 concurrent image resizes
- Time to download 5 videos in parallel
- Server responsiveness under mixed load
- CPU utilization with Python 3.14 free-threading

## Implementation Checklist

- [x] Add `anyio>=4.0.0` to dependencies
- [x] Add `aiofiles>=24.0.0` to dependencies
- [x] Refactor `create_video` reference image reading
- [x] Refactor `download_video` file writing
- [x] Refactor `prepare_reference_image` with thread pool
- [x] Refactor `download_image` base64 decode + file write
- [x] Update tests/smoke tests for concurrent operations
- [x] Benchmark before/after performance
- [x] Update documentation with concurrency notes

## Benefits

1. **True async under heavy load** - Multiple operations won't block each other
2. **Better Python 3.14 utilization** - Free-threading can parallelize CPU work
3. **Improved throughput** - Multiple clients can process simultaneously
4. **Responsive server** - Long operations don't freeze other requests
5. **Future-proof** - Scales better as API usage grows

## Risks & Considerations

1. **Complexity increase** - More moving parts, harder to debug
2. **Thread safety** - PIL operations must be isolated (already handled by `to_thread`)
3. **OpenAI SDK compatibility** - May need workarounds for SDK methods
4. **Testing burden** - Need concurrent test scenarios
5. **Diminishing returns** - Network latency to OpenAI API may dominate anyway

## Recommendation

**Priority:** Medium-High

The biggest wins will be:
1. **`prepare_reference_image`** - High CPU usage, frequently called
2. **`download_image`** - Multiple users downloading images simultaneously
3. **`download_video`** - Large video files benefit from async writes

Start with `prepare_reference_image` as it has the most blocking CPU work and is easiest to isolate.

## Implementation Results

**Status:** ✅ Fully implemented and verified in production

**Documentation:**
- See [`docs/async-optimizations.md`](../async-optimizations.md) for complete technical details and patterns
- See [`docs/stress-test-results.md`](../stress-test-results.md) for real-world stress test data

**Stress Test Results:**
- **32 concurrent downloads** completed successfully (4 agents × 8 downloads each)
- **100% success rate** - all downloads completed at identical timestamps
- **246MB total data** transferred without blocking
- Identical file sizes and MD5 checksums across all operations

**Benchmark Performance:**
- **8-10x speedup** for concurrent image processing operations
- **10-11x speedup** for concurrent file I/O operations
- **7-8x speedup** for concurrent base64 decoding
- Run benchmark: `python docs/benchmarks/async_benchmark.py`

**Key Achievement:** The server now handles heavy concurrent load with true non-blocking async architecture, proven under real-world stress testing conditions.
