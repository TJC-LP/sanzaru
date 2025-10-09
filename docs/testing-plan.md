# Testing Plan

## Overview

This document outlines testing strategies for the Sora MCP server, including edge cases, security scenarios, and manual testing procedures.

## Unit Test Coverage (Future)

### Path Validation (`get_path()`)

**Happy Path:**
- [ ] Valid SORA_VIDEO_PATH returns correct path
- [ ] Valid SORA_REFERENCE_PATH returns correct path
- [ ] Caching works (second call doesn't re-validate)

**Error Cases:**
- [ ] Missing env var raises RuntimeError with clear message
- [ ] Empty string env var raises RuntimeError
- [ ] Whitespace-only env var raises RuntimeError
- [ ] Non-existent directory raises RuntimeError
- [ ] File (not directory) raises RuntimeError
- [ ] Symlink path raises RuntimeError with security message

**Edge Cases:**
- [ ] Env var with leading/trailing whitespace is handled
- [ ] Relative paths get resolved to absolute
- [ ] Paths with special characters work correctly

### Security Tests

**Path Traversal Protection:**
- [ ] `../` in filename rejected in `create_video`
- [ ] `../` in filename rejected in `download_video`
- [ ] `../` in filename rejected in `prepare_reference_image`
- [ ] `../` in filename rejected in `download_image`
- [ ] Absolute paths in filenames rejected
- [ ] Symlinks in user-provided filenames (separate from env var symlinks)

**Reference Image Validation:**
- [ ] Invalid file extension rejected in `create_video`
- [ ] Non-existent reference file raises clear error
- [ ] Path traversal via reference filename blocked

### File Operations

**Download Operations:**
- [ ] `download_video` creates file with correct name
- [ ] `download_video` with custom filename works
- [ ] `download_video` rejects path traversal in custom filename
- [ ] `download_image` creates file with correct dimensions
- [ ] `download_image` handles all output formats (png, jpeg, webp)

**Image Preparation:**
- [ ] `prepare_reference_image` crop mode preserves aspect ratio
- [ ] `prepare_reference_image` pad mode adds letterboxing
- [ ] `prepare_reference_image` rescale mode stretches image
- [ ] Output filename auto-generation works correctly
- [ ] Custom output filename is respected

## Integration Tests (Manual)

### Basic Workflow Tests

**Test 1: Video Creation (No Reference)**
```bash
# Prerequisites: OPENAI_API_KEY, SORA_VIDEO_PATH set
1. Start server: uv run sora-mcp-server
2. Call create_video(prompt="test video", model="sora-2", seconds="4")
3. Poll get_video_status until completed
4. Call download_video
5. Verify: Video file exists in SORA_VIDEO_PATH
```

**Test 2: Video Creation (With Reference)**
```bash
# Prerequisites: Reference image in SORA_REFERENCE_PATH
1. Call list_videos_references to find available images
2. Call create_video with input_reference_filename
3. Poll until completed
4. Download and verify
```

**Test 3: Image Generation → Video**
```bash
1. Call create_image(prompt="test image")
2. Poll get_image_status until completed
3. Call download_image with custom filename
4. Verify: Image exists in SORA_REFERENCE_PATH
5. Call prepare_reference_image to resize
6. Call create_video using prepared image
7. Poll and download final video
```

### Error Handling Tests

**Test 4: Missing Environment Variables**
```bash
# Unset SORA_VIDEO_PATH
1. Call download_video
2. Expected: RuntimeError with message "SORA_VIDEO_PATH environment variable is not set"
```

**Test 5: Invalid Directory**
```bash
# Set SORA_VIDEO_PATH to non-existent directory
1. Call download_video
2. Expected: RuntimeError with "does not exist" message
```

**Test 6: Symlink Rejection**
```bash
# Create symlink: ln -s /tmp/videos sora-videos-link
# Set SORA_VIDEO_PATH=sora-videos-link
1. Start server or call get_path("video")
2. Expected: RuntimeError with "cannot be a symbolic link" message
```

**Test 7: Path Traversal Attempt**
```bash
1. Call download_video(video_id, filename="../../../etc/passwd.mp4")
2. Expected: ValueError with "path traversal detected" message
```

**Test 8: Whitespace in Env Var**
```bash
# Set SORA_VIDEO_PATH="  /path/to/videos  " (with spaces)
1. Call download_video
2. Expected: Works correctly (whitespace stripped)
```

**Test 9: Directory Deleted Mid-Session**
```bash
1. Start server successfully
2. Delete SORA_VIDEO_PATH directory externally
3. Call download_video
4. Expected: File operation fails (limitation of caching)
5. Note: Documented limitation - server restart required if dirs deleted
```

### Concurrent Operation Tests

**Test 10: Parallel Image Preparations**
```bash
# Using Python script or concurrent MCP client
1. Call prepare_reference_image 5 times in parallel
2. Verify: All complete without blocking each other
3. Verify: All output files created correctly
```

**Test 11: Parallel Downloads**
```bash
1. Have 3 completed videos
2. Call download_video 3 times in parallel
3. Verify: All download successfully
4. Check: No file corruption or conflicts
```

## Security Test Scenarios

### Symlink Attack Vectors

**Scenario 1: Symlink in env var**
```bash
mkdir /tmp/evil
ln -s /tmp/evil ~/.sora-videos-link
export SORA_VIDEO_PATH=~/.sora-videos-link
# Expected: Rejected at get_path() call
```

**Scenario 2: Symlink in reference directory**
```bash
# Symlink to sensitive file inside SORA_REFERENCE_PATH
cd sora-references
ln -s /etc/passwd evil.png
# Try to use in create_video
# Current behavior: Would follow symlink (potential issue)
# Consider: Add symlink check for user filenames too
```

**Scenario 3: Path traversal via filename**
```bash
download_video(video_id, filename="../../sensitive/file.mp4")
# Expected: ValueError "path traversal detected"
```

## Performance Tests

### Cache Effectiveness
```bash
# Measure get_path() performance
1. First call: ~X ms (validation + disk I/O)
2. Second call: <0.1ms (cached)
3. Verify: No repeated env var lookups
```

### Concurrent Load
```bash
# With Python 3.14 free-threading
1. 10 concurrent create_image calls
2. 10 concurrent prepare_reference_image calls
3. 10 concurrent downloads
4. Measure: Total time, CPU usage, memory usage
```

## Test Automation Strategy (Future)

### pytest Structure
```
tests/
├── unit/
│   ├── test_path_validation.py
│   ├── test_security.py
│   └── test_file_operations.py
├── integration/
│   ├── test_video_workflow.py
│   ├── test_image_workflow.py
│   └── test_concurrent_operations.py
└── conftest.py  # Fixtures for mock env vars, temp dirs
```

### Mock Strategy
- Mock `os.getenv()` for env var tests
- Use `pytest-asyncio` for async tests
- Create temp directories with `pytest.fixtures` for file tests
- Mock OpenAI API responses to avoid API costs

## Known Limitations

1. **Directory Deletion Mid-Session:**
   - `@lru_cache` caches validated paths
   - If directory deleted after validation, cached path is stale
   - File operations will fail with generic errors (not "directory missing")
   - **Mitigation:** Document that server restart required if directories are deleted
   - **Alternative:** Remove caching, validate every time (performance trade-off)

2. **Symlinks in User Filenames:**
   - Current code doesn't check if user-provided filenames are symlinks
   - Only checks env var paths
   - User could create symlink in SORA_REFERENCE_PATH pointing elsewhere
   - **Risk:** Low (still protected by path.startswith() check)
   - **Future:** Consider adding `is_symlink()` check for user filenames

3. **OpenAI SDK write_to_file():**
   - Blocking operation from SDK
   - Can't make async without wrapping in thread pool
   - **Future:** Address in async optimization work

## Checklist for Manual Testing (Pre-Release)

- [ ] Test with missing OPENAI_API_KEY
- [ ] Test with missing SORA_VIDEO_PATH
- [ ] Test with missing SORA_REFERENCE_PATH
- [ ] Test with whitespace in env vars
- [ ] Test with empty env vars
- [ ] Test with symlink paths (should fail)
- [ ] Test with non-existent directories
- [ ] Test with file instead of directory
- [ ] Test path traversal in all filename parameters
- [ ] Test video creation without reference
- [ ] Test video creation with reference
- [ ] Test image generation and download
- [ ] Test image preparation with all resize modes
- [ ] Test custom filenames in downloads
- [ ] Test concurrent operations (5+ parallel calls)
- [ ] Test with Python 3.14 free-threading
- [ ] Test with `uv run sora-mcp-server`
- [ ] Test with `mcp run`
- [ ] Test Claude Desktop integration

## Recommendations

**Priority 1 (Before Merge):**
- ✅ Add symlink rejection to `get_path()`
- ✅ Add whitespace/empty validation to env vars
- ✅ Document caching limitation for deleted directories

**Priority 2 (Post-Merge):**
- Consider symlink check for user filenames
- Add pytest test suite
- Add CI/CD with automated tests

**Priority 3 (Future):**
- Async optimization (see async-optimization-spec.md)
- Load testing framework
- Integration tests with real API (optional)
