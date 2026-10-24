# bowl-contents (Viam module, runs on the Orin)

CLIP (`openai/clip-vit-base-patch32`, Hugging Face `transformers`) on the Orin GPU says what food
is in a bowl crop. The dashboard looks at each fixed bowl, crops it from the gripper camera and
sends the crops here; the answer decides which bowl to scoop from ("cereal" / "mms" / "empty").

- Runs with the depth module's Python (`depth-anything-3/viam-env`: CUDA torch + transformers +
  viam SDK), so nothing is installed for it. Weights download to the HF cache on first load.
- Config: local module `executable_path: <repo>/modules/bowl-contents/run.sh`, service
  `{"api": "rdk:service:vision", "model": "assistive-arm:bowl-contents:clip", "name": "bowl-contents"}`.
  Optional attributes: `model_id`, `device` (auto/cuda/cpu), `labels` ({name: [prompts]}).
- `do_command({"command": "classify", "images_b64": [...]})`, `do_command({"command": "status"})`.
