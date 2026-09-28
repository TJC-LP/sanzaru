You are generating a multi-voice podcast using Sanzaru's `generate_podcast` tool.

## Overview

The `generate_podcast` tool takes a structured PodcastScript and produces a single
downloadable mp3 (or wav) file. It generates each spoken segment via TTS, then stitches
them together with configurable silence gaps.

## Choosing the engine

`generate_podcast` is the podcast tool even when you start from just a topic: you write the
script, then render it. Do not reach for `simulate_podcast` — it is experimental, the most
expensive tool in sanzaru, and meant only for an explicitly requested unscripted conversation.

**Recommended for any multi-voice script: ElevenLabs `eleven_v4`, `render_mode: "dialogue"`,
`verify: true`.** The model paces the turns itself and performs inline direction tags —
`[laughs]`, `[sighs]`, stacked `[whispers] [nervously]` — which v3 dropped in a blind listen.
Use `"model": "eleven_v4_turbo"` for drafts (half the character cost, roughly half the render time).

- Set `"provider": "elevenlabs"` on each speaker and give an ElevenLabs voice **id** (table below).
- Put delivery in the text as tags. `instructions` is ignored by ElevenLabs, and `speed` is refused
  on v4 (the API ignores it), so omit both.
- Keep each run of consecutive turns under 2000 characters; longer scripts are split into several
  dialogue requests automatically, at turn boundaries.
- Fall back to OpenAI voices (the guide below) when no `ELEVENLABS_API_KEY` is configured, or use
  `render_mode: "segments"` when you need exact pauses or per-speaker `voice_settings`.

ElevenLabs premade voices that pair well (global ids, available on every account):

| Voice | id | Character |
|---|---|---|
| George | `JBFqnCBsd6RMkjVDRZzb` | warm British storyteller |
| Laura | `FGY2WhTYpPnrIDTdsKH5` | quirky enthusiast |
| Alice | `Xb7hH8MSUJpSbSDYk0k2` | clear British educator |
| Roger | `CwhRBWXzGAHq8TQ4Fs17` | laid-back, casual |
| Matilda | `XrExE9yKIg1WjnnlVkGX` | professional |

```json
{
  "title": "show_name_ep01",
  "speakers": [
    {"id": "host", "name": "George", "voice": "JBFqnCBsd6RMkjVDRZzb", "provider": "elevenlabs"},
    {"id": "cohost", "name": "Laura", "voice": "FGY2WhTYpPnrIDTdsKH5", "provider": "elevenlabs"}
  ],
  "segments": [
    {"speaker": "host", "text": "[warmly] Welcome back. Today we're asking whether Haskell belongs in private equity."},
    {"speaker": "cohost", "text": "[laughs] Most people hear Haskell and think ivory tower."}
  ],
  "config": {"render_mode": "dialogue"}
}
```

Call it with `verify: true`.

## Step-by-Step Workflow

1. **Clarify the topic and format** — confirm episode topic, number of speakers, desired
   tone, and approximate length before writing the script.

2. **Write the PodcastScript** — produce the full script object (see schema below).
   Aim for natural conversational exchanges. 1–2 sentences per turn feels realistic.

3. **Estimate duration** — count words in all segments, divide by (150 × avg_speed).
   A typical 10-minute podcast needs ~1500 words of spoken content.

4. **Call generate_podcast** — pass the script. Generation takes time proportional to
   total word count (sequential TTS). Expect ~30–90 seconds for a 5–10 minute podcast.

5. **Deliver the file**:
   - Always: the output_file is written to the audio directory — tell the user the filename.
   - Desktop: call `view_media(media_type="audio", filename=<output_file>)` to open the
     player, then tell the user the filename as backup.
   - Mobile / unknown: just report the filename and instruct the user to find it in their
     configured audio directory. Do NOT attempt view_media on mobile.

6. **Optional: share the transcript** — the result includes a full formatted transcript.
   Offer to share or save it as show notes.

---

## Voice Pairing Guide (OpenAI)

Voice contrast is critical for listener comprehension. Pair voices with clear tonal
differences so listeners can distinguish speakers without visual cues.

### Recommended Pairings

| Format | Speaker A | Speaker B | Notes |
|--------|-----------|-----------|-------|
| News / analysis | **ash** | **nova** | Ash: authoritative anchor; Nova: energetic reporter |
| Technical deep-dive | **onyx** | **coral** | Onyx: measured, deep; Coral: enthusiastic |
| Interview | **fable** | **alloy** | Fable: warm interviewer; Alloy: neutral subject |
| Debate | **ash** | **shimmer** | Strong contrast between confident and smooth |
| Narrative / documentary | **onyx** (narrator) | **fable** (subject) | Narrator separate from subject |
| 3-way panel | **ash** + **nova** + **echo** | — | Good spread across pitch and energy |

### Voice Characteristics

| Voice | Character | Best For |
|-------|-----------|----------|
| ash | Authoritative, confident, measured | Hosts, anchors, lead speakers |
| nova | Youthful, friendly, reactive | Co-hosts, interviewees, guests |
| onyx | Deep, professional, deliberate | Narrators, senior speakers |
| alloy | Neutral, balanced | All-purpose, interview subjects |
| coral | Bright, energetic | Enthusiastic guests, segments requiring energy |
| fable | Warm, engaging, storytelling | Narrative voiceover, interview moderators |
| shimmer | Smooth, polished | Intros, sign-offs, guest experts |
| ballad | Expressive, lyrical | Emotional or creative segments |
| echo | Resonant, distinctive | Differentiated third voice in panel discussions |
| sage | Calm, soothing | Closing segments, thoughtful commentary |

### Speed Guidelines

- **0.9–0.95**: Slightly slower — gravitas, news anchor, documentary feel
- **1.0**: Normal pace — standard conversational tempo
- **1.05–1.1**: Slightly faster — energetic co-host, reactive commentary
- **Avoid >1.2 or <0.85** unless intentional (speed changes are noticeable)

---

## Silence / Pacing Guide

| Situation | Recommended pause_after |
|-----------|------------------------|
| Normal conversational turn | 500–700ms (use default_pause_ms) |
| Question followed by answer | 700–900ms (slightly longer) |
| Section transition | 1200–1500ms (explicit pause_after) |
| Dramatic pause after punchline | 1000–1200ms (explicit pause_after) |
| Rapid back-and-forth banter | 300–500ms |

**Config recommendations:**
```json
{
  "default_pause_ms": 600,
  "intro_silence_ms": 500,
  "outro_silence_ms": 1000,
  "normalize_loudness": true,
  "output_format": "mp3",
  "output_bitrate": "192k"
}
```

---

## Script Writing Tips

### Conversational rhythm
- Keep individual turns short (1–4 sentences). Long monologues lose the multi-voice feel.
- Interject reactions: "Right.", "Exactly.", "Wait — let me push back on that."
- Use verbal handoffs: "So Sam, what did you find?" — this marks clear speaker transitions.

### Intro / outro template
```
HOST: Welcome to [show name]. I'm [name], and joining me is [co-host].
COHOST: Great to be here. Today we're talking about [topic].
HOST: Let's get into it.
...
HOST: That's all for today. Thanks for listening to [show name].
COHOST: See you next time.
```

### Marking section transitions
To signal a section break (longer pause), set a larger `pause_after` (1200–1500ms)
on the last segment before the transition:
```json
{"speaker": "host", "text": "Now let's turn to the risks.", "pause_after": 1200}
```

---

## Full Script Template

```json
{
  "title": "show_name_ep01",
  "description": "Optional show notes / context for this episode.",
  "speakers": [
    {
      "id": "host",
      "name": "Alex",
      "voice": "ash",
      "speed": 1.0,
      "instructions": "Confident, authoritative podcast host. Measured pacing.",
      "role": "host"
    },
    {
      "id": "cohost",
      "name": "Sam",
      "voice": "nova",
      "speed": 1.05,
      "instructions": "Curious, energetic co-host. Slightly faster. Reactive.",
      "role": "cohost"
    }
  ],
  "segments": [
    {
      "speaker": "host",
      "text": "Welcome back to Type Safe Returns. Today we are asking whether Haskell has a place in private equity."
    },
    {
      "speaker": "cohost",
      "text": "This is a great question because most people hear Haskell and think ivory tower academics.",
      "pause_after": 700
    },
    {
      "speaker": "host",
      "text": "Right. But the firms using it in production have a completely different story to tell.",
      "pause_after": 1200
    },
    {
      "speaker": "cohost",
      "text": "So what makes Haskell attractive for financial modeling specifically?"
    },
    {
      "speaker": "host",
      "text": "Three things: correctness guarantees, composability, and the type system catching domain errors at compile time."
    }
  ],
  "config": {
    "default_pause_ms": 600,
    "intro_silence_ms": 500,
    "outro_silence_ms": 1000,
    "normalize_loudness": true,
    "output_format": "mp3",
    "output_bitrate": "192k"
  }
}
```

---

## Length Guidelines

| Target runtime | Approximate word count | Segments |
|----------------|----------------------|----------|
| 2 minutes | ~300 words | 6–10 short turns |
| 5 minutes | ~750 words | 15–25 turns |
| 10 minutes | ~1500 words | 30–50 turns |
| 20 minutes | ~3000 words | 60–100 turns |

---

## Known Limitations

- **No background music**: audio beds and transitions are not supported.
- **`instructions` is OpenAI-only**: OpenAI speakers get it; ElevenLabs speakers ignore it —
  direct them with inline audio tags (`[whispers]`, `[laughs]`) in the segment text instead.
- **Dialogue runs retry as a whole**: in `render_mode: "dialogue"`, a batched run is one
  request, so a verify failure re-renders (and re-bills) every turn in it. Runs cap at 2000
  characters; a turn left alone after a split renders as an ordinary segment.
- **Peak normalization only**: `normalize_loudness: true` uses pydub peak normalization,
  not LUFS-based normalization. Volume will be consistent but not calibrated to -16 LUFS.
- **Memory**: every segment is held in memory until stitching — roughly 2x the final file
  size at peak, which matters past about an hour of audio.

---

Now write the PodcastScript for the user's request and call `generate_podcast`.
