# Image Input Feature Specification

## Overview

This document specifies a feature to add reference image input support to the `create_image` tool, enabling image-to-image generation workflows with the OpenAI Responses API.

## Motivation

Currently, `create_image` only supports text-to-image generation. The OpenAI Responses API supports passing reference images to enable:

1. **Image editing**: Modify existing images based on text prompts
2. **Multi-image composition**: Combine multiple images into new scenes
3. **Inpainting**: Edit specific regions using alpha-channel masks
4. **High-fidelity preservation**: Maintain faces, logos, and fine details from input images

This unlocks creative workflows like:
- Generate reference image → Create variations with different styles
- Combine product photos → Generate composed marketing imagery
- Iterative refinement with both image and text inputs
- Precise region editing with masks

## Use Cases

### Use Case 1: Single Image Editing
**Scenario:** User has a photo of a pool and wants to add a flamingo.

```python
# 1. User already has lounge.png in REFERENCE_IMAGE_PATH
list_reference_images(pattern="lounge*")  # Verify it exists

# 2. Edit the image
create_image(
    prompt="add a flamingo swimming in the pool",
    input_images=["lounge.png"]
)
```

**Result:** New image with flamingo added to the pool scene.

### Use Case 2: Multi-Image Composition
**Scenario:** Create a gift basket containing multiple products from individual product photos.

```python
# 1. List available product images
list_reference_images(pattern="product_*")
# → ["product_lotion.png", "product_soap.png", "product_bomb.jpg", "product_incense.png"]

# 2. Compose into gift basket scene
create_image(
    prompt="Generate a photorealistic gift basket labeled 'Relax & Unwind' containing all these products",
    input_images=["product_lotion.png", "product_soap.png", "product_bomb.jpg", "product_incense.png"]
)
```

**Result:** New composed image showing all products arranged in a gift basket.

### Use Case 3: Logo/Face Preservation
**Scenario:** Add a company logo to a person's shirt while preserving both the face and logo details.

```python
create_image(
    prompt="Add the logo to the woman's shirt, stamped into the fabric",
    input_images=["woman.jpg", "company_logo.png"],
    input_fidelity="high"  # Preserve face and logo details precisely
)
```

**Result:** Image with logo seamlessly integrated, face and logo remain crisp.

### Use Case 4: Masked Inpainting
**Scenario:** Edit only a specific region (the pool area) without affecting the rest of the scene.

```python
# 1. User creates mask in image editor with alpha channel
#    - Transparent pixels = edit this area (pool)
#    - Black pixels = keep original (rest of scene)
#    - Save as pool_mask.png

# 2. Apply masked edit
create_image(
    prompt="add a flamingo swimming",
    input_images=["lounge.png"],
    mask_filename="pool_mask.png"
)
```

**Result:** Only the masked pool region is edited; surrounding lounge area unchanged.

### Use Case 5: Image-to-Video Pipeline
**Scenario:** Generate a reference image, then animate it with Sora.

```python
# 1. Generate base image
resp = create_image(
    prompt="futuristic pilot in orange suit in mech cockpit",
    size="1536x1024"
)

# 2. Wait for completion
get_image_status(resp["id"])  # Poll until completed

# 3. Download to reference path
download_image(resp["id"], filename="pilot.png")

# 4. Add variations using the base image
resp2 = create_image(
    prompt="same scene but at night with blue lighting",
    input_images=["pilot.png"]
)

# 5. Resize for Sora if needed
prepare_reference_image("pilot_night.png", "1280x720", resize_mode="crop")

# 6. Animate with Sora
create_video(
    prompt="The pilot looks up and smiles",
    input_reference_filename="pilot_night_1280x720.png",
    size="1280x720"
)
```

## Architecture Decision: Base64 vs File IDs

### Recommendation: **Base64 Encoding (Stateless)**

**Rationale:**
- ✅ **Aligns with server philosophy**: Maintains stateless architecture
- ✅ **Simpler mental model**: All files stay in `REFERENCE_IMAGE_PATH`, no OpenAI file lifecycle tracking
- ✅ **No cleanup burden**: No need to delete files from OpenAI storage after use
- ✅ **Consistent with existing patterns**: Mirrors how `create_video` handles reference images
- ✅ **One-shot operations**: Image editing typically not reused across requests

**Trade-offs Accepted:**
- ❌ Larger API payloads (~33% overhead from base64 encoding)
- ❌ Extra encoding step per request
- ❌ Cannot reuse uploaded images across multiple API calls

**Exception: Masks Use File IDs**
- Masks require the Files API (cannot use base64 per OpenAI docs)
- Small exception for inpainting feature only
- Mask files are typically small (<1MB), so upload overhead is minimal

### Alternative: File IDs (Stateful) - Not Recommended

Could be implemented later if needed for:
- High-frequency reuse of same reference images
- Batch processing workflows with repeated image references
- Cost optimization for very large images used repeatedly

**Why Not Now:**
- Adds complexity: file upload lifecycle management
- Requires cleanup: delete files after use or risk quota issues
- State leakage: files persist beyond single tool call
- Inconsistent: mixing stateless and stateful paradigms

## API Design

### New Parameters

```python
async def create_image(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,

    # NEW PARAMETERS
    input_images: list[str] | None = None,
    input_fidelity: Literal["low", "high"] | None = None,
    mask_filename: str | None = None,
) -> ImageResponse:
```

### Parameter Specifications

#### `input_images: list[str] | None`

**Purpose:** List of reference image filenames from `REFERENCE_IMAGE_PATH`.

**Behavior:**
- Each filename validated with `validate_safe_path(reference_path, filename)`
- Files read and encoded as base64 data URLs
- Order matters: first image receives highest detail preservation
- Supported formats: JPEG, PNG, WEBP
- When provided, `prompt` semantics change (describes edits, not generation)

**Examples:**
- Single image: `["lounge.png"]`
- Multiple images: `["lotion.jpg", "soap.png", "bomb.jpg", "incense.png"]`

**Security:**
- Path traversal protection: `../`, absolute paths rejected
- Symlink detection: symlinks rejected
- Extension validation: only `.jpg`, `.jpeg`, `.png`, `.webp` allowed

**Error Cases:**
- File not found → `ValueError: "Image file not found: {filename}"`
- Unsupported format → `ValueError: "Unsupported image format: {filename} (use JPEG, PNG, WEBP)"`
- Path traversal → `ValueError: "Invalid filename: path traversal detected"`

#### `input_fidelity: Literal["low", "high"] | None`

**Purpose:** Controls precision of input image detail preservation.

**Values:**
- `"low"`: Standard preservation (default if not specified)
- `"high"`: Maximum preservation of faces, logos, textures, fine details

**Behavior:**
- Only relevant when `input_images` is provided
- Ignored if `input_images` is `None`
- First image receives richest preservation regardless of setting
- High fidelity consumes more input image tokens (higher cost)

**Use When:**
- Faces must remain recognizable
- Logos/text must stay crisp and readable
- Fine textures are important (fabric patterns, wood grain)
- Brand elements must be preserved exactly

**Token Impact:**
High fidelity uses more input image tokens. See [OpenAI vision tokens guide](https://platform.openai.com/docs/guides/vision/calculating-costs) for cost implications.

#### `mask_filename: str | None`

**Purpose:** PNG image with alpha channel defining inpainting region.

**Requirements:**
- Must be PNG format with alpha channel
- Dimensions should match first image in `input_images`
- Transparent pixels = edit this region
- Black/opaque pixels = preserve original content

**Behavior:**
- Only works when `input_images` is provided
- Applied to first image only (if multiple images provided)
- Mask uploaded to OpenAI Files API, file ID passed to tool config
- OpenAI uses mask as guidance (not pixel-perfect boundary)

**Mask Creation:**
Users can create masks using:
- Photoshop: Layer mask with transparent regions
- GIMP: Alpha channel layer
- Python PIL: Convert grayscale to RGBA with alpha channel

**Error Cases:**
- Mask without input images → `ValueError: "mask_filename requires input_images parameter"`
- Non-PNG format → `ValueError: "Mask must be PNG format with alpha channel"`
- File not found → `ValueError: "Mask file not found: {mask_filename}"`

### Behavior Changes

#### Current Behavior: Text-Only Generation

```python
create_image(prompt="a sunset over mountains")
```

**Flow:**
1. `input` parameter = simple string prompt
2. Model generates image from scratch
3. No reference images involved

#### New Behavior: Image + Text Editing

```python
create_image(
    prompt="add a flamingo to the pool",
    input_images=["lounge.png"]
)
```

**Flow:**
1. Load `lounge.png` from `REFERENCE_IMAGE_PATH`
2. Encode as base64 data URL
3. `input` parameter = structured list:
   ```python
   [
       {
           "role": "user",
           "content": [
               {"type": "input_text", "text": "add a flamingo to the pool"},
               {"type": "input_image", "image_url": "data:image/png;base64,..."}
           ]
       }
   ]
   ```
4. Model edits the provided image based on prompt

**Key Difference:**
- Prompt describes **changes to make**, not what's already in the image
- Model sees the image context and applies modifications

#### New Behavior: Multi-Image Composition

```python
create_image(
    prompt="gift basket containing all these products",
    input_images=["lotion.png", "soap.png", "bomb.jpg"]
)
```

**Flow:**
1. Load all 3 images from `REFERENCE_IMAGE_PATH`
2. Encode each as base64 data URL
3. `input` parameter includes all images in order
4. First image gets richest detail preservation
5. Model synthesizes new image using all references

### Tool Configuration Structure

The `input_fidelity` parameter is simply passed as a key/value pair in the `tools` array sent to the Responses API.

**Without input_fidelity:**
```python
tools=[
    {
        "type": "image_generation",
        "size": "1024x1024",
        "quality": "high",
        "output_format": "png"
    }
]
```

**With input_fidelity="high":**
```python
tools=[
    {
        "type": "image_generation",
        "size": "1024x1024",
        "quality": "high",
        "output_format": "png",
        "input_fidelity": "high"  # Simple key/value addition
    }
]
```

**Complete API call example:**
```python
response = await client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Add the logo to the woman's top"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,..."},
                {"type": "input_image", "image_url": "data:image/png;base64,..."}
            ]
        }
    ],
    tools=[
        {
            "type": "image_generation",
            "input_fidelity": "high"  # Preserves faces/logos precisely
        }
    ],
    background=True,
)
```

**Key implementation detail:** In `create_image()`, it's a one-line addition to the tool config:
```python
if input_fidelity is not None:
    tool_config["input_fidelity"] = input_fidelity
```

## Implementation Details

### File Structure

**Modified Files:**
1. `src/sora_mcp_server/tools/image.py` - Core implementation (~120 lines added)
2. `src/sora_mcp_server/descriptions.py` - Enhanced `CREATE_IMAGE` description
3. `src/sora_mcp_server/server.py` - Add parameters to MCP tool registration
4. `README.md` - Document new workflows
5. `CLAUDE.md` - Update with image editing patterns

**New Test Files:**
6. `tests/unit/test_image_encoding.py` - Test base64 encoding helpers
7. `tests/integration/test_image_input.py` - Integration tests for image inputs

### Implementation Pseudocode

#### Helper Function: Encode Image

```python
async def _encode_image_base64(image_path: Path) -> str:
    """Read image file and encode as base64 string.

    Args:
        image_path: Absolute path to validated image file

    Returns:
        Base64-encoded string (not data URL, just the base64 part)

    Raises:
        ValueError: If file cannot be read (with context)
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")
    except FileNotFoundError as e:
        raise ValueError(f"Image file not found: {image_path.name}") from e
    except PermissionError as e:
        raise ValueError(f"Permission denied reading image: {image_path.name}") from e
    except OSError as e:
        raise ValueError(f"Error reading image file: {e}") from e
```

#### Helper Function: Get MIME Type

```python
def _get_mime_type(image_path: Path) -> str:
    """Get MIME type from file extension.

    Args:
        image_path: Path with validated extension

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png")
    """
    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")  # Default to jpeg
```

#### Helper Function: Upload Mask File

```python
async def _upload_mask_file(mask_path: Path) -> str:
    """Upload mask image to OpenAI Files API.

    Args:
        mask_path: Absolute path to validated PNG mask with alpha channel

    Returns:
        OpenAI file ID string

    Raises:
        ValueError: If upload fails
    """
    client = get_client()

    try:
        with open(mask_path, "rb") as f:
            file_obj = await client.files.create(
                file=f,
                purpose="vision"
            )
        return file_obj.id
    except Exception as e:
        raise ValueError(f"Failed to upload mask file: {e}") from e
```

#### Main Function: Updated `create_image()`

```python
async def create_image(
    prompt: str,
    model: str = "gpt-5",
    size: Literal["auto", "1024x1024", "1024x1536", "1536x1024"] | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    output_format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["transparent", "opaque", "auto"] | None = None,
    previous_response_id: str | None = None,
    input_images: list[str] | None = None,
    input_fidelity: Literal["low", "high"] | None = None,
    mask_filename: str | None = None,
) -> ImageResponse:
    """Create image generation job with optional input images.

    [Full docstring with parameter descriptions]
    """
    client = get_client()
    reference_path = get_path("reference")

    # Validate mask requires input images
    if mask_filename and not input_images:
        raise ValueError("mask_filename requires input_images parameter")

    # Build tool configuration
    tool_config: dict = {"type": "image_generation"}

    if size is not None:
        tool_config["size"] = size
    if quality is not None:
        tool_config["quality"] = quality
    if output_format is not None:
        tool_config["output_format"] = output_format
    if background is not None:
        tool_config["background"] = background
    if input_fidelity is not None:
        tool_config["input_fidelity"] = input_fidelity

    # Handle mask upload if provided
    if mask_filename:
        mask_path = validate_safe_path(reference_path, mask_filename)
        check_not_symlink(mask_path, "mask image")

        # Validate PNG format
        if mask_path.suffix.lower() != ".png":
            raise ValueError("Mask must be PNG format with alpha channel")

        # Upload to Files API
        mask_file_id = await _upload_mask_file(mask_path)
        tool_config["input_image_mask"] = {"file_id": mask_file_id}

        logger.info("Uploaded mask %s as file_id %s", mask_filename, mask_file_id)

    # Build input parameter
    if input_images:
        # Structured input with images
        content_items: list[dict] = [{"type": "input_text", "text": prompt}]

        for filename in input_images:
            # Validate filename and construct path
            img_path = validate_safe_path(reference_path, filename)
            check_not_symlink(img_path, "reference image")

            # Validate file extension
            ext = img_path.suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                raise ValueError(
                    f"Unsupported image format: {filename} (use JPEG, PNG, WEBP)"
                )

            # Encode to base64
            base64_data = await _encode_image_base64(img_path)
            mime_type = _get_mime_type(img_path)

            # Add to content items
            content_items.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_data}"
            })

        input_param = [{"role": "user", "content": content_items}]

        logger.info(
            "Creating image with %d reference image(s)%s",
            len(input_images),
            f" (fidelity={input_fidelity})" if input_fidelity else ""
        )
    else:
        # Simple text-only input (existing behavior)
        input_param = prompt
        logger.info("Creating image from text prompt only")

    # Create response
    prev_resp_param: str | Omit = omit if previous_response_id is None else previous_response_id
    response = await client.responses.create(
        model=model,
        input=input_param,
        tools=[tool_config],
        previous_response_id=prev_resp_param,
        background=True,
    )

    logger.info(
        "Started image generation %s (%s)%s",
        response.id,
        response.status,
        f" from {previous_response_id}" if previous_response_id else "",
    )

    return {
        "id": response.id,
        "status": str(response.status) if response.status else "unknown",
        "created_at": response.created_at,
    }
```

### Security Considerations

**Path Traversal Protection:**
- All filenames validated with `validate_safe_path(base_path, filename)`
- Rejects `../`, `./`, absolute paths
- Ensures resolved path stays within `REFERENCE_IMAGE_PATH`

**Symlink Protection:**
- All file paths checked with `check_not_symlink()`
- Prevents symlink attacks to read arbitrary system files

**Extension Validation:**
- Only `.jpg`, `.jpeg`, `.png`, `.webp` allowed
- Prevents execution of malicious file types
- Mask specifically requires `.png`

**File Size Limits:**
- Rely on OpenAI API limits (typically 20MB per image)
- No explicit server-side size validation (keeps implementation simple)
- API will return error if image too large

**Error Message Safety:**
- Never expose full system paths in errors
- Use filename only: `"Image file not found: cat.png"` not `"/Users/..."`
- Prevents information disclosure about filesystem structure

## Testing Strategy

### Unit Tests

**File:** `tests/unit/test_image_encoding.py`

```python
"""Unit tests for image encoding utilities."""

class TestImageEncoding:
    def test_encode_image_base64_success(self, tmp_path):
        """Test successful base64 encoding of image file."""

    def test_encode_image_base64_file_not_found(self):
        """Test error handling when image file doesn't exist."""

    def test_encode_image_base64_permission_error(self, tmp_path):
        """Test error handling when permission denied."""

    def test_get_mime_type_jpg(self):
        """Test MIME type detection for JPEG."""
        assert _get_mime_type(Path("test.jpg")) == "image/jpeg"

    def test_get_mime_type_png(self):
        """Test MIME type detection for PNG."""
        assert _get_mime_type(Path("test.png")) == "image/png"

    def test_get_mime_type_webp(self):
        """Test MIME type detection for WebP."""
        assert _get_mime_type(Path("test.webp")) == "image/webp"

    def test_validate_image_extension_valid(self):
        """Test validation accepts supported formats."""

    def test_validate_image_extension_invalid(self):
        """Test validation rejects unsupported formats."""
```

### Integration Tests

**File:** `tests/integration/test_image_input.py`

```python
"""Integration tests for image input functionality."""

class TestImageInput:
    @pytest.mark.integration
    async def test_create_image_with_single_input(self, mocker, tmp_reference_path, sample_image):
        """Test creating image with single reference image."""

    @pytest.mark.integration
    async def test_create_image_with_multiple_inputs(self, mocker, tmp_reference_path):
        """Test creating image with multiple reference images."""

    @pytest.mark.integration
    async def test_create_image_with_high_fidelity(self, mocker, tmp_reference_path, sample_image):
        """Test high fidelity parameter is passed to tool config."""

    @pytest.mark.integration
    async def test_create_image_with_mask(self, mocker, tmp_reference_path, sample_image):
        """Test masked inpainting with alpha channel."""

    @pytest.mark.integration
    async def test_create_image_mask_without_images_error(self, tmp_reference_path):
        """Test error when mask provided without input images."""

    @pytest.mark.integration
    async def test_create_image_invalid_filename_path_traversal(self, tmp_reference_path):
        """Test path traversal protection for input images."""

    @pytest.mark.integration
    async def test_create_image_unsupported_format(self, tmp_reference_path):
        """Test error for unsupported image format (e.g., .gif)."""

    @pytest.mark.integration
    async def test_create_image_symlink_rejected(self, tmp_reference_path, sample_image):
        """Test symlink detection rejects symlinked images."""
```

### Manual Testing Checklist

**Basic Functionality:**
- [ ] Text-only generation still works (existing behavior preserved)
- [ ] Single image input generates expected result
- [ ] Multiple image inputs compose correctly
- [ ] High fidelity preserves face/logo details
- [ ] Mask-based inpainting edits only masked region

**Error Handling:**
- [ ] Missing image file returns clear error
- [ ] Unsupported format (.gif, .bmp) rejected
- [ ] Path traversal attempts blocked (`../../../etc/passwd.png`)
- [ ] Symlink to system file rejected
- [ ] Mask without input images fails with helpful message
- [ ] Non-PNG mask format rejected

**Integration:**
- [ ] Works with `previous_response_id` for iterative refinement
- [ ] Images downloaded with `download_image()` can be reused as inputs
- [ ] `list_reference_images()` finds all usable inputs
- [ ] Prepared images from `prepare_reference_image()` work as inputs

## Documentation Updates

### README.md

**New Section: Image Editing with Reference Images**

```markdown
## Image Editing

The `create_image` tool supports using reference images to create new images based on existing ones.

### Single Image Editing

Modify an existing image with a text prompt:

```python
create_image(
    prompt="add a flamingo swimming in the pool",
    input_images=["lounge.png"]
)
```

### Multi-Image Composition

Combine multiple images into a new scene:

```python
create_image(
    prompt="photorealistic gift basket containing all these products",
    input_images=["lotion.png", "soap.png", "bomb.jpg", "incense.png"]
)
```

### High-Fidelity Preservation

Preserve faces, logos, and fine details:

```python
create_image(
    prompt="add this logo to the woman's shirt",
    input_images=["woman.jpg", "logo.png"],
    input_fidelity="high"
)
```

### Masked Inpainting

Edit specific regions using an alpha-channel mask:

```python
create_image(
    prompt="add a flamingo",
    input_images=["pool.png"],
    mask_filename="pool_mask.png"
)
```

**Mask Requirements:**
- PNG format with alpha channel
- Transparent pixels = edit this region
- Black pixels = preserve original

### Combined Workflow: Generate → Edit → Animate

```python
# 1. Generate base image
resp = create_image("futuristic pilot in mech cockpit", size="1536x1024")
get_image_status(resp["id"])  # Poll until completed
download_image(resp["id"], "pilot.png")

# 2. Create night variation
resp2 = create_image(
    prompt="same scene at night with blue lighting",
    input_images=["pilot.png"]
)
get_image_status(resp2["id"])
download_image(resp2["id"], "pilot_night.png")

# 3. Prepare for Sora
prepare_reference_image("pilot_night.png", "1280x720", resize_mode="crop")

# 4. Animate
create_video(
    prompt="The pilot looks up and smiles",
    input_reference_filename="pilot_night_1280x720.png",
    size="1280x720"
)
```
```

### CLAUDE.md

**Update Typical Workflows Section:**

```markdown
## Typical Workflows

### Image Editing with Reference Images

**Single image editing:**
```python
create_image(
    prompt="add a flamingo to the pool",
    input_images=["lounge.png"]
)
```

**Multi-image composition:**
```python
create_image(
    prompt="gift basket with all these items",
    input_images=["lotion.png", "soap.png", "bomb.jpg"]
)
```

**High-fidelity logo/face preservation:**
```python
create_image(
    prompt="add logo to shirt",
    input_images=["person.jpg", "logo.png"],
    input_fidelity="high"
)
```

**Masked region editing:**
```python
create_image(
    prompt="add flamingo",
    input_images=["pool.png"],
    mask_filename="pool_mask.png"
)
```

### Generate Reference Image → Animate with Sora

[Existing workflow from README]
```

## Open Questions

### 1. File ID Support for Regular Images

**Question:** Should we support File IDs as an alternative to base64 for input images?

**Option A (Current Spec):** Base64 only for images, File IDs only for masks
- Simpler implementation
- Fully stateless except for mask upload
- Users don't need to think about file lifecycle

**Option B:** Support both base64 and File IDs
- `input_images` accepts filenames (base64) OR file IDs
- New parameter: `input_file_ids: list[str] | None`
- Users can optimize for reuse

**Recommendation:** Start with Option A, add File ID support later if users request it.

### 2. Async Image Encoding

**Question:** Should base64 encoding use thread pool for non-blocking I/O?

**Option A (Current Spec):** Synchronous encoding
- Simpler implementation
- Acceptable for typical image sizes (<10MB)
- ~10-50ms overhead per image

**Option B:** Async with `anyio.to_thread.run_sync()`
- Non-blocking for large images
- Aligns with `async-optimization-spec.md`
- Requires `anyio` dependency

**Recommendation:** Start synchronous, optimize later if profiling shows bottleneck.

### 3. Input Image Limit

**Question:** Should we enforce a maximum number of input images?

**Option A:** No limit (trust OpenAI API)
- Simpler implementation
- API will error if too many

**Option B:** Enforce 4-image limit
- Matches OpenAI documentation examples
- Clear error message before API call

**Option C:** Configurable limit via environment variable
- Most flexible

**Recommendation:** Start with Option A (no limit), document OpenAI's limits.

### 4. Mask Dimension Validation

**Question:** Should we validate mask dimensions match first input image?

**Option A:** Client-side validation
- Open first image with PIL, get dimensions
- Compare with mask dimensions
- Fail fast with clear error

**Option B:** Let OpenAI API handle it
- Simpler implementation
- API returns error if mismatch

**Recommendation:** Option B (API validation), document requirement in description.

### 5. Mask File Cleanup

**Question:** Should we delete uploaded mask files from OpenAI after use?

**Context:** Masks are uploaded to Files API with `purpose="vision"`. These files may accumulate over time.

**Option A:** No cleanup
- Simpler implementation
- Users manage their Files API quota
- Document that masks persist

**Option B:** Track and delete after response completes
- Cleaner, prevents quota issues
- Adds complexity (need to poll response completion)
- Requires stateful tracking

**Option C:** Background cleanup job
- Delete old vision files periodically
- Separate concern from tool logic

**Recommendation:** Start with Option A (no cleanup), document in description. Add cleanup later if users hit quota issues.

## Cost Implications

### Input Image Tokens

Using `input_images` consumes additional input image tokens based on:
- Image dimensions (higher resolution = more tokens)
- Input fidelity setting (`high` = more tokens than `low`)

**Rough Token Estimates (per image):**
- 512x512 image: ~85 tokens (low), ~170 tokens (high)
- 1024x1024 image: ~340 tokens (low), ~680 tokens (high)
- 1536x1024 image: ~510 tokens (low), ~1020 tokens (high)

See [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision/calculating-costs) for exact calculations.

### File Upload Costs

- Mask file uploads: No direct cost (storage quota only)
- Files API storage: May count against account quota
- Consider cleanup strategy if heavy usage

### Recommendations for Users

**Cost Optimization Tips:**
1. Use smaller images when high fidelity not needed
2. Resize images before using as reference (with `prepare_reference_image`)
3. Use `input_fidelity="low"` unless faces/logos critical
4. Batch workflows to minimize API calls

## Environment Variable Rename: `SORA_REFERENCE_PATH` → `REFERENCE_IMAGE_PATH`

### Motivation

The current `REFERENCE_IMAGE_PATH` environment variable name better reflects that this directory will be used for:
1. **User-provided reference images** for `create_image` tool
2. **Downloaded generated images** from `download_image`
3. **Reference images for Sora videos** (existing use case)

The new name `REFERENCE_IMAGE_PATH` better reflects that this is a general-purpose directory for all reference images, not just Sora-specific content.

### Changes Required

**Total Impact:** 64+ occurrences across 23 files
**Complexity:** Medium (mostly mechanical string replacements)
**Breaking Change:** Yes - users must update their `.env` files

#### Critical Code Changes (7 locations)

1. **`src/sora_mcp_server/config.py`** (lines 70-71)
   ```python
   # OLD
   path_str = os.getenv("REFERENCE_IMAGE_PATH")
   env_var = "REFERENCE_IMAGE_PATH"

   # NEW (Already updated)
   path_str = os.getenv("REFERENCE_IMAGE_PATH")
   env_var = "REFERENCE_IMAGE_PATH"
   ```

2. **`setup.sh`** (line 102)
   ```bash
   # OLD (Already updated)
   REFERENCE_IMAGE_PATH="$REFERENCE_PATH"

   # NEW (Already updated)
   REFERENCE_IMAGE_PATH="$REFERENCE_PATH"
   ```

3. **`tests/unit/test_config.py`** (4 locations)
   - Update mock env var patches: `{"REFERENCE_IMAGE_PATH": ...}`
   - Update error message assertions: `"REFERENCE_IMAGE_PATH"`

#### Documentation Updates (52+ locations)

- **Tool descriptions** (5 files in `descriptions.py`): Update all mentions
- **User documentation** (README.md, CLAUDE.md, AGENTS.md): ~25 references
- **Specifications** (testing-plan.md, this file): ~15 references
- **Comments and docstrings** (tools/*.py): ~10 references

#### Configuration Files (3 locations)

- **`.env.example`**: Update example env var name
- **`.gitignore`**: Update to `reference-images/` directory pattern
- **`setup.sh`**: Update generated `.env` file and default path

### Design Decisions

#### Default Directory Name: Change to `reference-images/`

**Decision:** Change the default directory name to `reference-images/` for semantic consistency.

**Rationale:**
- Better semantic clarity: "reference-images" clearly describes the directory purpose
- Consistent with the new `REFERENCE_IMAGE_PATH` environment variable name
- More descriptive than previous naming which implied Sora-only usage
- Provides full semantic consistency across the codebase

**Example `.env` after change:**
```bash
REFERENCE_IMAGE_PATH="/absolute/path/to/reference-images"
```

#### Internal API: Keep `get_path("reference")`

**Decision:** Keep the internal API parameter as `"reference"` (not `"reference_image"`).

**Rationale:**
- Shorter, cleaner API
- Less code churn (12+ call sites unchanged)
- Internal abstraction doesn't need to match env var name exactly

### Migration Guide for Users

**Breaking Change Notice:**
```markdown
## Breaking Change: Environment Variable Renamed

**OLD:** `SORA_REFERENCE_PATH`
**NEW:** `REFERENCE_IMAGE_PATH`

### Action Required

1. **Update your `.env` file:**
   ```diff
   - SORA_REFERENCE_PATH="/path/to/reference-images"
   + REFERENCE_IMAGE_PATH="/path/to/reference-images"
   ```

2. **No directory rename needed:**
   - You can keep your `reference-images/` directory
   - Only the environment variable name changed

3. **Restart the MCP server** after updating `.env`

### Why This Change?

Better reflects that this directory is for all reference images (including
user-provided images for the new image input feature), not just Sora-specific content.
```

### Implementation Checklist

**Phase 1: Code Changes**
- [ ] Update `config.py` lines 70-71 (env var name)
- [ ] Update `setup.sh` line 102 (generated .env file)
- [ ] Update all 5 tool descriptions in `descriptions.py`
- [ ] Update all docstrings in tools/*.py (4 files, ~10 locations)
- [ ] Update test mocks in `test_config.py` (4 locations)
- [ ] Update test assertions in `test_config.py` (1 error message)

**Phase 2: Configuration**
- [ ] Update `.env.example` line 3
- [ ] Update `.gitignore` to use `reference-images/` pattern
- [ ] Test `setup.sh` generates correct `.env`

**Phase 3: Documentation**
- [ ] Update README.md (~7 occurrences)
- [ ] Update CLAUDE.md (~3 occurrences)
- [ ] Update AGENTS.md (~4 occurrences)
- [ ] Update docs/testing-plan.md (~7 occurrences)
- [ ] Update this spec (~7+ occurrences)
- [ ] Add migration section to README

**Phase 4: Testing**
- [ ] Run full test suite: `pytest`
- [ ] Verify `get_path("reference")` still works
- [ ] Test setup.sh generates correct .env
- [ ] Test error messages display new env var name
- [ ] Manual smoke test with actual server

### Testing Requirements

```bash
# Automated tests
pytest tests/unit/test_config.py::TestGetPath::test_get_path_reference_valid
pytest tests/unit/test_config.py::TestGetPath::test_missing_reference_path_error
pytest tests/integration -v

# Manual verification
./setup.sh  # Verify .env contains REFERENCE_IMAGE_PATH
uv run sora-mcp-server  # Test server starts correctly
```

### Risk Assessment

**Low Risk:**
- Documentation updates (can't break functionality)
- Docstring updates (don't affect runtime)

**Medium Risk:**
- Env var name change in `config.py` (core functionality)
  - **Mitigation:** Comprehensive test coverage exists
- Setup script change (affects new installations)
  - **Mitigation:** Test script manually before release

**User Impact:**
- **Breaking change** for existing installations
  - **Mitigation:** Clear migration docs, prominent release notes

### Estimated Effort

- **Code changes:** 2 hours
- **Testing:** 1 hour
- **Total:** 3 hours

### Recommendation

✅ **Implement alongside image input feature in the same PR**

**Rationale:**
1. Both changes relate to reference images
2. Semantic improvement makes more sense with image input feature
3. Single migration point for users (update once for both features)
4. Clear narrative: "Expanding reference images beyond Sora"

## Future Enhancements

### Phase 2: File ID Support

**Motivation:** Allow reusing uploaded images across multiple requests.

**Changes:**
- New parameter: `input_file_ids: list[str] | None`
- New tool: `upload_reference_image(filename) -> file_id`
- New tool: `delete_reference_file(file_id)`
- Track uploaded files in user's Files API storage

### Phase 3: Image URL Support

**Motivation:** Use images from external URLs without downloading first.

**Changes:**
- Accept URLs in `input_images`: `["https://example.com/image.jpg"]`
- Auto-detect URL vs filename
- Security: whitelist allowed domains or require user confirmation

### Phase 4: Async Optimization

**Motivation:** Non-blocking image encoding for large files.

**Changes:**
- Use `anyio.to_thread.run_sync()` for base64 encoding
- Use `aiofiles` for async file reads
- Aligns with `async-optimization-spec.md`

## Success Metrics

**Feature Adoption:**
- % of `create_image` calls using `input_images` parameter
- Average number of input images per call
- Usage of `input_fidelity="high"` vs default

**Quality Metrics:**
- User satisfaction with edited images
- Error rate for image input operations
- Most common error types

**Performance Metrics:**
- Average encoding time per image
- Total request latency with 1, 2, 4 input images
- API token consumption patterns

## References

- [OpenAI Image Generation Guide](https://platform.openai.com/docs/guides/image-generation)
- [OpenAI Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI Files API Reference](https://platform.openai.com/docs/api-reference/files)
- [Vision Tokens Calculation](https://platform.openai.com/docs/guides/vision/calculating-costs)

## Appendix: Creating Alpha-Channel Masks

### Using Photoshop

1. Open image in Photoshop
2. Create new layer mask
3. Paint with black (keep) or white (edit)
4. File → Export → PNG (ensure "Transparency" checked)

### Using GIMP

1. Open image in GIMP
2. Layer → Mask → Add Layer Mask
3. Paint with black (keep) or white (edit)
4. File → Export As → PNG (ensure "Save background color" unchecked)

### Using Python PIL

```python
from PIL import Image

# Load grayscale mask (white = edit, black = keep)
mask_gray = Image.open("mask_gray.png").convert("L")

# Convert to RGBA
mask_rgba = mask_gray.convert("RGBA")

# Use grayscale values as alpha channel
mask_rgba.putalpha(mask_gray)

# Save as PNG with alpha
mask_rgba.save("mask_alpha.png", "PNG")
```

### Mask Best Practices

- Use soft edges (feathering) for natural blending
- Test mask on small region first
- Keep mask file size reasonable (<5MB)
- Name masks descriptively: `pool_region_mask.png`
