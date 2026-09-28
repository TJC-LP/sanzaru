# Reference Images Guide

Reference images are start frames (and optional end frames) for image-to-video on Higgsfield,
and inputs for `create_image` / `edit_image`. The video model animates from the image, keeping
its subject, style, composition and lighting.

> **0.13.0:** video moved from OpenAI Sora (removed 2026-09-24) to Higgsfield. Reference images no
> longer have to match a fixed set of video sizes; instead, **the image decides the frame shape**.

## Supported formats

- **JPEG** (.jpg, .jpeg), **PNG** (.png), **WEBP** (.webp) — and GIF for Higgsfield uploads
- Prepared images are saved as PNG for best quality

## Where they live

Reference images live in the reference directory: `SANZARU_MEDIA_PATH/images` (or `IMAGE_PATH`).
The server only reads files there (path-traversal and symlink protection).

```python
list_reference_images()                                   # everything
list_reference_images(pattern="dog*", file_type="png")    # filter
list_reference_images(sort_by="modified", order="desc", limit=10)
```

When a video job uses a reference image, sanzaru uploads it to Higgsfield through a presigned URL
(your API key is never sent to the storage host). A completed job's `hf_…` id can be passed
instead of a filename to reuse that job's output directly.

## Framing: the image decides the shape

Image-to-video has **no `aspect_ratio` parameter** — Seedance 2.5 frames the output from the start
image. To choose the shape, crop the reference first:

```python
prepare_reference_image("sunset.jpg", aspect_ratio="16:9")     # → sunset_1280x720.png
prepare_reference_image("portrait.png", aspect_ratio="9:16")   # → portrait_720x1280.png
prepare_reference_image("logo.png", size="1024x1024", resize_mode="pad")
```

`aspect_ratio` takes `16:9`, `4:3`, `1:1`, `3:4`, `9:16` or `21:9` and targets Seedance's 720p frame
for that shape; `size="WxH"` targets an exact size (64–4096 px per edge). Pass exactly one.

For an **end frame** (`end_image`), prepare it to the same shape as the start frame.

## Resize modes

| Mode | What it does | Use for |
|---|---|---|
| `crop` (default) | Scale to cover the target, center-crop the excess — no distortion, may lose edges | Photos, scenes |
| `pad` | Scale to fit inside, add black bars — no distortion, whole image kept | Logos, graphics, text that must stay visible |
| `rescale` | Stretch/squash to the exact size — may distort | Abstract art where distortion is fine |

Check the result before paying for a render: `inspect_image("sunset_1280x720.png")` shows you
whether the crop kept the subject.

## Prompting with a reference image

The image already carries the character, setting, framing, style and lighting. **Describe only
the motion and the camera.**

❌ Re-describing the image:
```python
create_video(prompt="A beautiful sunset over the ocean with orange and pink clouds...",
             reference_image="sunset_1280x720.png")
```

✅ Describing what happens:
```python
create_video(prompt="The sun sinks below the horizon as the clouds drift left. Slow push-in.",
             reference_image="sunset_1280x720.png")
```

See the [video prompting guide](video-prompting-guide.md) for more.

## Complete workflows

### Animate an existing image

```python
images = list_reference_images(pattern="sunset*")
prepare_reference_image("sunset.jpg", aspect_ratio="16:9")
job = create_video(prompt="The sun sinks as clouds drift across the sky",
                   reference_image="sunset_1280x720.png", duration=5, max_cost_usd=3)
wait_for([job.id], download=True)          # waits server-side and saves the mp4
```

### Generate the reference first

```python
generate_image(prompt="futuristic pilot in mech cockpit", size="1536x1024", filename="pilot.png")
prepare_reference_image("pilot.png", aspect_ratio="16:9")
job = create_video(prompt="The pilot looks up and smiles", reference_image="pilot_1280x720.png",
                   duration=5, resolution="480p", max_cost_usd=2)
wait_for([job.id], download=True)
extend_video("He reaches for the throttle.", source_video=job.id, duration=5, max_cost_usd=3)
```

### Refine, then animate

```python
base = create_image(prompt="a cyberpunk character, full body")
refined = create_image(prompt="add neon details and a city background", previous_response_id=base.id)
wait_for([refined.id], download=True)
# then prepare + create_video as above
```

## Troubleshooting

- **"File not found"** — the file must be in the reference directory; check `list_reference_images()`.
- **`aspect_ratio` refused with a reference image** — framing follows the image; crop it with
  `prepare_reference_image(aspect_ratio=...)` instead.
- **The video ignores the reference** — simplify the prompt to motion only; re-describing the scene
  competes with the image.
- **The image looks distorted** — `rescale` stretches; use `crop` or `pad`.
