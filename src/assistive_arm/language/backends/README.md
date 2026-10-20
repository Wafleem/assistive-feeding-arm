# `language/backends`: one interface, interchangeable LLMs

| File | Backend | Status |
|---|---|---|
| `base.py` | `LLMBackend` protocol (`available`, `complete(system, user, json_schema, timeout)`), `BackendError` | contract |
| `local.py` | `LocalLLMBackend`: Ollama / llama.cpp on the AGX Orin (**default**) | TODO |
| `anthropic_backend.py` | `AnthropicBackend`: Claude API (optional extra `llm-cloud`) | TODO, notes in the docstring |
| `openai_backend.py` | `OpenAIBackend`: OpenAI API (optional extra `llm-cloud`) | TODO |
| `fake.py` | `FakeBackend`: canned replies for tests | works |
| `__init__.py` | `REGISTRY`, `backends_from_env()` (`ASSIST_LLM_BACKENDS`) | TODO |

## Rules
- A backend only returns raw text. Validation, fallback and safety words belong to `IntentRouter`.
- Cloud SDKs are imported inside the methods (lazy). Tests never need network access or keys.
- Keys come only from environment variables. Subscriptions are not API access. See `../README.md`.
