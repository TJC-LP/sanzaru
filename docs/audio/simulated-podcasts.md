# Simulated podcasts

`generate_podcast` speaks a script you wrote. `simulate_podcast` records a conversation that
did not exist until the models had it.

Each host is a `gpt-realtime` session with a persona. A producer gives one host the floor,
collects its audio, and plays those exact frames into every other host's ears — so the models
respond to delivery and timing, not to a transcript. The audio *is* the performance; there is
no TTS step and no script.

```
sanzaru podcast rundown  PREMISE    # plan only — cheap, editable
sanzaru podcast simulate BRIEF      # record it
sanzaru podcast generate SCRIPT     # the scripted path, unchanged
```

---

## Why it is built this way

Three findings from live recordings shaped every structural decision here. They are worth
reading before changing anything.

### 1. The producer is load-bearing, not polish

Two agents alternating on a topic brief, with no producer, drift immediately. Turns ran
**30.8 seconds** against an instruction that said "two or three sentences", and the
conversation slid into mutual agreement.

The same agents with a hard length rule, a `max_output_tokens` backstop, and a per-turn
steering note held **9–17 second turns** and stayed on brief.

So `producer.py` does three jobs, and all three matter:

- **Floor control** — exactly one agent is asked to respond at a time.
- **Coverage** — talking points are walked deliberately across the act, so a brief with four
  points does not produce an act about the first one.
- **Landing** — the last turns are steered to a conclusion and, for a non-final act, to the
  handoff the next act was briefed to pick up.

Steering notes are system messages. The audience never hears them.

**`max_turns` is a budget, not a stop.** It was a hard cap, which made act length a function of
how long the model's turns happen to run: full-size `gpt-realtime` turns measured ~10s against
the mini's ~17.5s under the same 15s rule, so full-model acts landed about a third short of
their targets. An act now borrows up to **1.5x** its planned turns (bounded by `MAX_ACT_TURNS`)
to reach `target_seconds`, and the closing turn is cued once one more *measured* average turn
would reach the target — before the overshoot, because the closing turn still has to fit.
Extension turns carry an anti-recap steer, since an act that outlives its talking points drifts
into restatement. A single-turn act (`max_turns: 1`) never extends: one turn is a shape you
asked for, not an estimate. An act that spends every extension turn and still lands short
reports `stop_reason: "max_turns"` — that value is the undershoot signal, not a routine ending.

Measured on the same episode family: 6:28 against a 7–9 minute target became **11:36 against an
11:15 plan, with all five acts on their marks**.

### 2. `max_output_tokens` counts the transcript too

Audio output runs a very steady **20 tokens per second** of speech — three independently
recorded acts came in at 19.96, 19.99 and 20.05. It is tempting to size the per-turn cap
from that alone.

Doing so truncated **17 of 29 turns** mid-sentence. The Realtime API returns a text
transcript alongside the audio and bills it separately, at another 9–17 tok/s. The real
output rate is 29–37 tok/s, and the spread is transcript-driven, so the cap needs headroom.

`turn_token_cap()` budgets both rates and multiplies by `TURN_TOKEN_HEADROOM`. That constant
is tuned, not guessed:

| headroom | effect |
| --- | --- |
| 1.0 | the cap does the prompt's job — 17/29 turns clipped mid-sentence |
| 1.5 | **current** — prompt stays in charge, cap only stops a genuine runaway |
| 2.0 | nothing clips, but turns stretch to ~22s against a 15s rule |

Truncated turns are counted and reported per act (`truncated_turns`), so this can't silently
regress again.

### 3. Acts drift *forward*, and it reads as repetition

Acts are recorded in parallel and cannot hear each other, so a rundown tells each act what
came before (`prior_context`). The first three-act QC run came back with every act on-brief,
no missed points — and every act flagged:

> Act 1 leaks into audio-spec territory, Act 2 leaks into MP3-provider territory, and Act 3
> therefore repeats material listeners have already heard.

Act 1 had no idea sample rates belonged to act 2. The fix is symmetric: `upcoming` tells each
act what *later* acts own. It is derived automatically from the following two acts, so it
works for hand-edited rundowns too, and an explicitly written one always wins.

With `upcoming` in place, the same rundown recorded clean: `repeats_earlier` and `off_brief`
false on every act.

---

## Directing it yourself

Everything above describes what the built-in producer does *by default*. The caller of this
tool is usually itself an agent, and it is a better producer than a set of f-strings — it
knows which point deserves dwelling on and when someone should follow their own thought.

It cannot steer live: the tool blocks while acts record in parallel, and a round-trip per turn
would cost more than the recording. So direction is declarative, set on the act before
recording. Three fields, all optional, all replacing the default rather than adding to it:

| field | what it replaces |
| --- | --- |
| `direction` | nothing — free text added to every host's instructions for this act |
| `turn_notes` | the generated steering note, per turn index (`""` = say nothing) |
| `speaking_order` | round-robin; cycles if shorter than the act |

Setting `turn_notes` without `speaking_order` also **pins who opens the act**. Acts otherwise
rotate their starting host, which silently reassigned index-keyed notes to the other host on odd
acts — that is how a mandated reveal went missing in a live run.

```json
{
  "id": "act1",
  "title": "Cowardice",
  "topic": "whether saving partial work is discipline or an admission you expect to fail",
  "direction": "Let this get genuinely tense. Nobody concedes. Interrupt.",
  "max_turns": 5,
  "speaking_order": ["rory", "avery", "avery", "rory"],
  "turn_notes": {
    "0": "Open by calling checkpointing cowardice. One sentence. Do not greet anyone.",
    "2": "Do not let Rory answer yet — press your own point harder with a concrete example.",
    "4": "Refuse to resolve it. End mid-disagreement. One sentence."
  }
}
```

Recorded, that produces Rory → Avery → Avery → Rory → Rory…, with Avery's second turn reaching
for a concrete example instead of handing back, and the act ending unresolved. `turn_notes` is
the strongest lever in the tool: it is the difference between "move onto the next point" and
"object to what they just said."

A note on the **last planned turn** (`max_turns - 1`, turn 4 here) takes over the closing, so
landing the act becomes your job. It follows the closing turn rather than firing on that index:
if the act extends toward `target_seconds`, turn 4 becomes mid-act and the note waits for the
real landing. Any other note the closing turn was carrying is superseded, and logged so you can
see it happened.

The workflow this is built for:

```bash
sanzaru podcast simulate @rundown.json --dry-run   # returns the full rundown
# edit direction / turn_notes / speaking_order
sanzaru podcast simulate @rundown.json --max-cost 2.00 -o ep.mp3
```

One interaction to watch: the cold-open instruction ("welcome listeners once") lives in the
act *instructions*, so telling turn 0 not to greet can just move the greeting to turn 1. If
you want no greeting at all, say so in `direction`, which every host sees.

What is *not* delegable, and shouldn't be: one speaker at a time, the PCM broadcast, the token
cap, the seconds/turn budgets, the cost ceiling. Those are correctness and money, not taste.

---

## What is checked before anything runs

Recording costs money, so the rundown and brief are pydantic-validated up front and the bounds
live in the schema — which means they also reach the MCP tool description, where an agent can
read the rule instead of discovering it by spending.

**Rejected** (exit 2, with the field named):

| | why it matters |
| --- | --- |
| duplicate act ids | two acts share one checkpoint filename; a resumed run silently loses one |
| duplicate host ids | two hosts collapse into one speaker, breaking stems and `speaking_order` |
| `speaking_order` naming an unknown host | previously surfaced four acts into a paid run |
| `turn_notes` past the extended ceiling (1.5x `max_turns`) | the note would never fire, silently |
| a separator in an id, `run_id`, or `filename` | these reach filenames; storage sanitizes too, but late and confusingly |
| out-of-range budgets | `acts` 1–24, `target_minutes` ≤ 240, `target_seconds` ≤ 2400/act, `max_cost_usd` > 0 |

**Warned about, but allowed** — these go to stderr and are worth reading:

- an unknown voice (OpenAI ships them faster than we do, so this must not hard-fail)
- a model with no known price, especially with `--max-cost` set: the ceiling cannot fire
- an act with too few turns to fill its `target_seconds` *even extended* — it will stop on the
  turn cap short of its target, and its last talking point never gets air. This is the exact
  calibration bug the first live run hit.

---

## Acts, and why an episode is chunked

Not an optimization. Three independent forces require it:

1. **A hard limit.** A Realtime WebSocket closes at 60 minutes. A long episode cannot be one
   session at any price.
2. **Cost.** Context grows within a session. A 30-minute episode (~138 turns) costs roughly
   **992k input tokens as one session versus 192k across six acts**. Honest caveat: prompt
   caching already flattens the *uncached* term either way, so most of that saving is in
   cached tokens, which are the cheap ones.
3. **The execution model.** Acts record in parallel, which is what makes a blocking tool
   viable at all.

| acts | turns/act | total input | wall clock |
| --- | --- | --- | --- |
| 1 | 138 | 992k | 6.6 min |
| 6 | 23 | 192k | **1.1 min** |
| 12 | 12 | 112k | 0.5 min |

Measured: a 3-act, 7-minute episode records in **~30 seconds** of wall clock, plus ~25s for
QC and mixdown.

### Concurrency

Each act opens one session per host, so a 6-act two-host episode wants 12 concurrent
sessions. The account-level ceiling is not published, so `SANZARU_REALTIME_MAX_SESSIONS`
defaults to a conservative **6** (3 parallel two-host acts). Raise it once you know yours:

```bash
export SANZARU_REALTIME_MAX_SESSIONS=12
```

---

## Cost

This is the most expensive thing sanzaru does. Measured on a 7-minute, 3-act, two-host
episode: **$0.21 on `gpt-realtime-2.1-mini`**. The full model is roughly 3× that.

Always dry-run first. It plans the episode and projects turns, duration, tokens and dollars
without recording anything:

```bash
sanzaru podcast simulate @rundown.json --dry-run
```

```
sanzaru: dry run — 'The Hard Part Isn't the Model': 3 acts, up to 27 turns
sanzaru:   act1: The Model Is the Easy Part to Demo — 120s, up to 9 turns
sanzaru: projected ~6 min audio, 16,593 input / 10,800 output tokens
sanzaru: projected cost ~$0.20 (estimate, not a quote)
sanzaru: nothing was recorded; drop --dry-run to record
```

Then set a ceiling on the real run. `--max-cost` is checked after **every turn** across every
parallel act, not once per act:

```bash
sanzaru podcast simulate @rundown.json --max-cost 2.00
```

Crossing it exits **6** with the run id, what was spent, which acts are safe on disk, and a
`resume` command that carries a **raised** `--max-cost`. That last part matters: the ceiling is
restored from the manifest and a resumed run replays the spend of the acts it reads back, so a
bare `--resume` would abort at the same total after paying to re-record the same acts. It does
not get the chance to — a resume whose projected total cannot fit under the restored ceiling
refuses before opening a session, and names the ceiling that would work.

Prices live in `audio/realtime/pricing.py` (list prices captured 2026-08-05). They go stale;
override without waiting for a release:

```bash
# text_in,cached_text_in,audio_in,cached_audio_in,audio_out,text_out — USD per 1M tokens
export SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1=4,0.4,32,0.4,64,24
```

A model with no known price is reported in `cost.unpriced_models` rather than silently
charged at zero.

---

## Checkpointing and resume

Every act is written to the audio directory the moment it finishes, with a JSON sidecar
holding its turns, durations and usage. A crash, an interrupt, or a cost abort never throws
away audio that was already paid for.

```
simrun_<run_id>.json                 # rundown + settings — makes resume self-sufficient
<slug>_<run_id>_act1.mp3 / .json     # audio + turns/usage sidecar
<slug>_<run_id>.mp3                  # the finished episode
<slug>_<run_id>_stem_<host>.mp3      # optional per-host tracks
```

The run id is printed on stderr **before** recording starts, because that is the only moment
it matters:

```
sanzaru: run d33730ea — resume with: sanzaru podcast simulate --resume d33730ea
```

**stderr is the only place it appears**, which strands paid audio for any harness that parses
only stdout. Name the run yourself instead — `--run-id ep1`, or a top-level `"run_id"` in the
rundown JSON (the flag wins) — and `--resume ep1` is predictable even if the shell died before
anyone read a line. Recording over an id that already exists is refused rather than silently
overwriting that run's manifest and checkpoints; resume it, or pick another id.

Resume needs nothing else — the manifest carries the rundown:

```bash
sanzaru podcast simulate --resume d33730ea
```

Measured: resuming a 3-act episode with one act missing re-recorded only that act, in 31
seconds. A checkpoint that is corrupt, truncated, or holds no decodable audio is re-recorded
rather than treated as fatal — one bad act must not take the whole resume down with it.

Three things this depends on, all of them deliberate:

- **`-o` moves the episode, never the bookkeeping.** `-o ./out/ep1.mp3` repoints the whole
  audio path type for that one invocation, but the resume command it prints carries no `-o`.
  So the manifest and the act checkpoints are always written through the *default* backend
  (the media dir), and only the episode and stems follow `-o`. The two flags are documented
  as one workflow; they have to compose.
- **A bare `--resume` reinstates the run's settings.** Anything you pass this time wins;
  everything else — including `--max-cost` — is restored from the manifest. Following the
  ceiling abort's own hint must not re-run uncapped, and must not re-run into the same wall
  either: that is why the abort prints a resume command with a higher `--max-cost`, and why a
  resume that provably cannot finish under the restored ceiling stops before recording. An act
  checkpoint is also written under a shield, so a sibling act's abort cannot cancel it halfway
  and leave an mp3 with no sidecar for the next resume to discard.
- Checkpoints are **not** deleted after a successful run — the storage backend has no delete
  operation. They are safe to remove by hand once you have the episode.

A resumed act is decoded from its mp3 checkpoint, so it carries one extra generation of
encoding. Acts recorded in the same run go to the mixer as raw PCM with no round-trip.

### Turn timeouts

Nothing in the Realtime protocol bounds a turn, so a session that stops answering would hold
its slot forever inside a blocking tool. Each turn runs under `anyio.fail_after`; the bound
defaults to 6x `--turn-seconds` (never under a minute) and a breach surfaces as a
`RealtimeAPIError`. That fails the run — it is not retried, and it cancels the acts recording
alongside it — but acts that already finished keep their checkpoints, and the failure envelope
carries the `run_id` and a `resume` command, so the recovery path is the same as any other
interrupt. Override the bound with `SANZARU_REALTIME_TURN_TIMEOUT` (seconds) or `turn_timeout_s`
in the brief.

---

## Quality control

A scripted podcast has a script to check against. A simulated one does not, which inverts the
problem: the conversation is the output. Two transcripts make it tractable.

- **Intended** — the Realtime API returns `output_audio_transcript` per turn, free. What the
  model meant to say.
- **Rendered** — `gpt-transcribe` on the finished audio. What a listener actually hears.

A word-level similarity score between them catches dropped audio with no model judgement at
all. Normal transcription disagreement lands at 0.83–0.86; below 0.80 is worth a listen.

A judge model then reads each act's rendered transcript against its brief for what a diff
cannot see: skipped talking points, drift, and acts repeating each other.

```
sanzaru: qc warn: act2 — see result.qc for why (--qc-retry re-records just those)
```

`--qc-retry` re-records only the flagged acts, once. That is cheap precisely because acts are
independent — the same property that makes checkpointing work.

One flag is worth handling by hand: a `tail_truncated` act was cut off by
`max_output_tokens`, so re-recording at the same cap is likely to reproduce it and bill you
twice. Raise `--turn-tokens` before retrying that one.

**A retry is not automatically the better take.** QC verdicts disagree run-to-run on the same
material, and a live retry has lost a mandated figure the first take had. So the take being
replaced is preserved beside it as `<slug>_<run>_<act>_take1.mp3` (`_take2`, … on later
retries), with its `.json` sidecar. Those names are outside what `--resume` reads, so the run
still holds exactly one truth while you stay free to assemble the best cut per act — which is
how a real episode ended up with two acts from the retry and one from the preserved take.

QC runs per act, not per episode: acts are already on disk, a 30-minute episode would exceed
the 25MB upload limit, and per-act verdicts are what make selective retry possible. It costs
about $0.005 per minute of audio.

> **Note:** `gpt-live-transcribe` is *not* a substitute for `gpt-transcribe` here. It is
> realtime-streaming only; `/v1/audio/transcriptions` rejects it with a 404. It remains
> interesting for live transcription *during* capture — a possible follow-up, not this.

---

## Audio path

Realtime speaks PCM16 mono at 24kHz in both directions, so one agent's output frames go
straight into another's input buffer with no transcoding, and straight to the mixer with no
mp3 round-trip.

`podcast._stitch_audio` takes a `decode` callable for exactly this reason: the scripted path
passes mp3 (the TTS contract), the simulated path passes `mixdown.pcm_to_segment`. Everything
downstream — frame-rate pinning to 44.1kHz, normalization, bitrate — is shared.

Turns inside an act are butted together with no inserted silence: the models pace themselves,
and adding gaps makes it sound scripted. Acts are separated by `--act-gap` (default 700ms).

`normalize_loudness` defaults to **false** here, unlike scripted podcasts. Acts recorded by
the same models are already level, and per-act peak normalization can *introduce* jumps.

### Stems

`--stems` writes one time-aligned track per host: the full episode length, with silence
wherever that host is not speaking, so every stem lines up with the master sample-for-sample
and drops onto an editor timeline. Stems are rendered and encoded one at a time — each is a
full-length copy of the episode in raw PCM.

---

## Memory

Audio is held in memory as raw PCM: roughly **48 KB per second**, so ~86 MB for a 30-minute
episode, plus a transient copy during mixdown and one more per stem. Fine at podcast lengths;
worth watching past an hour.

---

## Module map

```
src/sanzaru/audio/realtime/
├── types.py      # rundown/act/turn/usage values, PCM16 helpers
├── agent.py      # one persona on one connection: configure / speak / hear / steer
├── producer.py   # floor control, coverage steering, act budgets, prompts
├── rundown.py    # pre-production: premise → parallel-recordable acts
├── budget.py     # shared cost ceiling, charged every turn
├── pricing.py    # token→dollars, and the measured rates a dry run projects from
├── mixdown.py    # PCM→AudioSegment, stems, checkpoint decode
└── qc.py         # transcribe the rendered audio, judge it against the rundown

src/sanzaru/tools/simulate_podcast.py   # the tool: parallel acts, checkpoints, resume
src/sanzaru/cli/podcast.py              # rundown / simulate / generate
```

Every `openai.realtime` import is function-local or `TYPE_CHECKING`-only, so `sanzaru --help`
never pays for the SDK (guarded by `tests/cli/test_root.py`).

Tests run against a fake connection object — the agent only ever touches five methods and
async iteration — so floor control, budgets, checkpointing and resume are all covered without
an SDK, a websocket, or a cent of spend.
