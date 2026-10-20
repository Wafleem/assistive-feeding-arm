# 5. Language layer

**Owner:** stream 5. **Runs on:** AGX Orin (faster-whisper, local LLM); dashboard TTS in the browser.
**Goal:** turn what the person says into a validated `FeedIntent`, keep the meal
conversation, and talk back.

## Examples to support
"feed me the strawberries" · "a grape please" · "next bite" · "a sip of water" ·
"slower" · "smaller bite" · "wait" · "stop" · "I'm done" · "what's on the plate?" · "yes" / "no"

## Interfaces
- `FeedIntent` / `Action` (`schema.py`): the only thing orchestration accepts
- `rules.parse(text)`: offline fallback **and** the safety path (stop/wait words first)
- `Router.route(text, state)`: LLM → JSON → `FeedIntent.from_dict`, fallback to rules
- `DialogueState.apply(intent) -> reply text`; `SpeechToText`, `TextToSpeech`

## Plan
1. Schema + validation + JSON schema for structured output.
2. Rule parser for every example above (pattern from git tag v0-wheelchair-skills).
3. Local LLM on the AGX (small quantised model via Ollama/llama.cpp, OpenAI-compatible
   API), prompt with the schema + plate contents + state; 1.5 s timeout → rules.
4. faster-whisper push-to-talk; browser TTS for replies and confirmations.
5. Dialogue: remember preferences (pace, bite size), what's left, pending questions.

## TODO checklist
- [ ] schema validation + `from_dict` + `json_schema` → `tests/test_language.py`
- [ ] `rules.parse` for every example
- [ ] Router with the local LLM; latency table
- [ ] Speech in/out; measure "stop" end-of-speech → arm halted (< 300 ms target)

## Done when
≥ 95% correct intents on 100 recorded utterances; "stop" never depends on the LLM.
