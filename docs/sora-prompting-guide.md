# Sora 2 Prompting Guide

> Source: [OpenAI Cookbook - Sora 2 Prompting Guide](https://cookbook.openai.com/examples/sora/sora2_prompting_guide)

## Key Principles for Sora 2 Video Generation

### 1. Prompt Approach
- **Think like a cinematographer briefing**: Approach prompting as if you're briefing a cinematographer who hasn't seen your storyboard
- **Balance control and creativity**: Find the sweet spot between detailed control and creative freedom
- **Embrace variation**: Expect different results with the same prompt - this is a feature, not a bug
- **Iterate and collaborate**: Work with the model through iteration

### 2. API Parameters

Critical parameters to set explicitly:

- **Model**: `sora-2` (faster, cheaper) or `sora-2-pro` (higher quality)
- **Size**: Supported resolutions:
  - `1280x720` (landscape)
  - `720x1280` (portrait)
  - `1024x1792` (tall portrait)
  - `1792x1024` (wide landscape)
- **Seconds**: Clip length - `4`, `8`, or `12` seconds

### 3. Prompt Anatomy

Effective prompts should include:

1. **Describe the shot like a storyboard**
2. **State camera framing** (wide, close-up, etc.)
3. **Note depth of field** (shallow, deep focus)
4. **Describe action in beats** (step-by-step progression)
5. **Set lighting and palette** (golden hour, neon-lit, etc.)

### 4. Visual Cues and Style

- **Style is powerful**: Use specific aesthetic descriptors
  - Examples: "1970s film", "cinema verité", "noir aesthetic"
- **Be clear and precise**: Concrete visual details work better than abstract concepts
- **Reference visual media**: Film styles, photography techniques, art movements

### 5. Motion and Timing

**Keep motion simple:**
- Describe **one clear camera move** (dolly-in, pan left, etc.)
- Describe **one clear subject action**
- Use **beats or counts** for precise timing
- Avoid complex multi-part movements

### 6. Lighting and Color

**Lighting:**
- Describe light quality (soft, harsh, diffused)
- Specify light sources (natural light, neon, candlelight)
- Note light direction and tone

**Color:**
- Use 3-5 color references for consistency
- Anchor colors to specific elements
- Consider color temperature (warm/cool tones)

### 7. Advanced Techniques

- **Image input**: Use reference images for more control (see `input_reference_filename`)
- **Remix functionality**: Build on existing videos with new prompts
- **Iterative refinement**: Make small, controlled changes between iterations

## Prompt Structure Template

```
[Prose scene description]

Cinematography:
Camera shot: [framing and angle]
Mood: [overall tone]

Actions:
- [Specific action 1]
- [Specific action 2]

Lighting: [light description]
Color palette: [color references]

Dialogue: [Optional, brief dialogue]
```

## Example Prompt

```
Style: 1970s romantic drama, shot on 35mm film

At golden hour, a brick tenement rooftop transforms into a small stage.
Laundry lines sway, catching warm sunlight.

Cinematography:
Camera: medium-wide shot, slow dolly-in
Depth of field: shallow focus on subjects
Mood: nostalgic, intimate

Actions:
- A guitarist strums opening chords
- Singer steps forward, begins verse
- Wind catches hanging sheets

Lighting: Soft golden hour backlight, warm amber tones
Color palette: Burnt orange, dusty rose, faded denim, brick red

Dialogue: "This is our moment..."
```

## Tips for Success

### Do's ✓
- Be specific about camera angles and movements
- Use clear, concrete visual descriptions
- Specify lighting conditions and quality
- Describe actions sequentially
- Reference specific film styles or aesthetics
- Keep motion simple and focused

### Don'ts ✗
- Avoid vague or abstract descriptions
- Don't overcomplicate with too many simultaneous actions
- Avoid conflicting style references
- Don't expect pixel-perfect consistency across generations
- Avoid extremely complex camera choreography

## Working with Reference Images

When using `input_reference_filename`:
- Reference image dimensions must match target video size
- Use `sora_prepare_reference` to resize images automatically
- Reference images provide strong visual anchors
- Prompt should describe how to animate/transform the reference

## Model Selection

### sora-2
- **Best for**: Quick iterations, testing concepts, prototyping
- **Speed**: Faster generation
- **Cost**: More economical
- **Quality**: High quality, suitable for most use cases

### sora-2-pro
- **Best for**: Final production, complex scenes, maximum fidelity
- **Speed**: Slower generation (especially for 12s clips)
- **Cost**: Higher cost
- **Quality**: Maximum visual quality and coherence

## Common Prompting Patterns

### Product Demo
```
Professional product demonstration with smooth camera movement.
Close-up dolly shot of [product] on clean background.
Soft studio lighting, shallow depth of field.
Minimal motion, elegant reveal.
```

### Cinematic Scene
```
Cinematic [genre] scene, shot on ARRI Alexa.
[Detailed scene description with environment and subjects].
Camera: [specific shot type and movement]
Lighting: [specific lighting setup]
Color grading: [color palette and mood]
```

### Animation Style
```
Style: [specific animation style]
[Scene description]
Smooth, fluid motion. [Specific animation characteristics].
[Color palette]. [Mood and atmosphere].
```

## Iteration Strategy

1. **Start broad**: Begin with a general concept
2. **Refine specifics**: Add camera, lighting, and color details
3. **Control motion**: Specify precise movements
4. **Polish style**: Add aesthetic refinements
5. **Use remix**: Build on successful generations

## Summary

Effective Sora prompting combines:
- Clear cinematographic direction
- Specific visual details
- Simple, focused motion
- Thoughtful lighting and color
- Iterative refinement
- Appropriate model selection

Think of each prompt as a creative collaboration between your vision and the model's capabilities.
