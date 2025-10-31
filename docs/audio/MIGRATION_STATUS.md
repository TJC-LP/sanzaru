# Audio Feature Migration Status

**Start Date:** 2025-10-31
**Source:** mcp-server-whisper v1.1.0
**Target:** sanzaru v0.2.0

## Progress

- [x] Phase 0: Foundation (in progress)
- [ ] Phase 1: Parallel Migration
  - [ ] Track A: Audio Domain Logic
  - [ ] Track B: Audio Services
  - [ ] Track C: Infrastructure & Tools
  - [ ] Track D: Tests & Documentation
- [ ] Phase 2: Integration

## Architecture Decision

Making ALL features truly optional:
- Base package: Core MCP + OpenAI client only
- `sanzaru[video]`: Sora video generation
- `sanzaru[audio]`: Whisper, GPT-4o audio, TTS
- `sanzaru[image]`: GPT Vision with Pillow
- `sanzaru[all]`: All features

Users can install exactly what they need!

## Notes

Foundation scaffolding in progress.

## Attribution

This migration incorporates code from [mcp-server-whisper](https://github.com/arcaputo3/mcp-server-whisper) by Richie Caputo, licensed under MIT.
