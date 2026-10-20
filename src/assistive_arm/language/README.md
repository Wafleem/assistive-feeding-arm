# 5. Language layer

**Owner:** stream 5. **Runs on:** AGX Orin (faster-whisper, local LLM); dashboard TTS in the browser.
**Goal:** turn what the person says into a validated `FeedIntent`, keep the meal
conversation, and talk back.

## Examples to support
"feed me the strawberries" · "a grape please" · "next bite" · "a sip of water" ·
"slower" · "smaller bite" · "wait" · "stop" · "I'm done" · "what's on the plate?" · "yes" / "no"

## Design: one router, interchangeable backends

```
utterance ──► 1 safety rules ──(stop / wait / no / I'm done)──► intent, immediately; NO model is called
                 │ otherwise
                 ▼
             2 LLM backends, in the configured order (each: timeout, JSON, schema check)
                 local (AGX, default) ──fail──► anthropic ──fail──► openai ──fail──┐
                 ▼ valid FeedIntent                                                │
             3 rule parser (always available, offline) ◄───────────────────────────┘
```

- `IntentRouter.route(utterance, state) -> RouteResult(intent, source, latency, errors)` (`router.py`).
- Backends implement one interface (`backends/base.py`): `available()`, `complete(system, user, json_schema, timeout) -> str`.
  - `LocalLLMBackend` (**default**): a small quantised model on the AGX Orin via Ollama or llama.cpp.
  - `AnthropicBackend`: Claude through the Claude API. Optional.
  - `OpenAIBackend`: the OpenAI API. Optional.
  - `FakeBackend`: canned replies for tests and offline demos.
- **Choosing backends:** set `ASSIST_LLM_BACKENDS=local,anthropic,openai` (fallback order).
  The default is `local`; the rule parser is always last.
- **Safety words can't be overridden:** "stop", "wait", "no" and "I'm done" are handled by the rule layer
  *before* any model and returned directly. A model's output is only ever used after it passes
  `FeedIntent.from_dict`.
- **Cloud backends are optional extras:** `pip install -e ".[llm-cloud]"`. They are imported lazily, so
  `make test` and offline runs never need network access or keys. Tests use `FakeBackend`.

## Subscriptions are not API access

ChatGPT Plus/Pro and Claude Pro/Max are consumer subscriptions for the chat apps.
They are **not** API access, and nothing here may drive them programmatically
(no browser automation, no scraping, no session tokens). The cloud backends need
**API keys** from the Claude Console / OpenAI platform:
- Put keys in `.env` (copy `.env.example`); `.env` is git-ignored.
- Read keys only from environment variables.
- Never commit, print or log a key.

## Cost, latency, privacy

| | Local (AGX) | Cloud API (Claude / OpenAI) |
|---|---|---|
| Privacy | Speech and intents never leave our hardware ("nothing you say leaves the table") | Utterances are sent to the provider |
| Works without venue internet | yes | no |
| Latency | predictable; target < 1 s for a short JSON reply on the AGX (measure) | network-dependent, often ~0.5-2 s; the router's per-backend timeout (1.5 s) caps the wait |
| Understanding odd phrasing | good for this small schema; weaker on long, rambling requests | strongest |
| Cost | none per call | tiny: each routing call is a few hundred input tokens and a short JSON reply; a whole demo day is a few hundred calls, i.e. cents to low dollars. **Ask the organisers for API credits.** |

Suggested setup: `local` for the demo (privacy story, no network risk), and
`local,anthropic` while developing, to compare accuracy on the recorded utterances.

## Interfaces
- `FeedIntent` / `Action` (`schema.py`, alias `Intent`): the only thing orchestration accepts
- `rules.parse(text)`: the safety path **and** the final fallback
- `IntentRouter`, `RouteResult` (`router.py`); backends in `backends/`
- `DialogueState.apply(intent) -> reply text`; `SpeechToText`, `TextToSpeech` (`speech.py`)

## Plan
1. Schema validation + `from_dict` + `json_schema` (the structured-output schema every backend gets).
2. The rule parser for every example above; safety words first (the pattern is in git tag v0-wheelchair-skills).
3. `IntentRouter.route`: safety rules → backends in order (timeout, JSON, validation) → rules.
4. `LocalLLMBackend` on the AGX: pick a small instruct model, constrain its output to the schema, measure latency.
5. Cloud backends (optional): Anthropic and OpenAI, same contract, short timeouts, no retries (the router falls back).
6. faster-whisper push-to-talk; browser TTS for replies and confirmations.
7. Dialogue: remember preferences (pace, bite size), what's left, pending questions.

## TODO checklist
- [ ] schema validation + `from_dict` + `json_schema` → `tests/test_language.py`
- [ ] `rules.parse` for every example
- [ ] `IntentRouter.route` (fallback + safety-word specs in `tests/test_language.py`)
- [ ] `LocalLLMBackend` on the AGX; accuracy and latency table on 100 recorded utterances
- [ ] `AnthropicBackend`, `OpenAIBackend` (optional)
- [ ] Speech in/out; measure "stop" end-of-speech → arm halted (< 300 ms target)

## Done when
≥ 95% correct intents on 100 recorded utterances with the default (local) backend.
"Stop" never depends on a model. `make test` passes with no network access and no keys.
