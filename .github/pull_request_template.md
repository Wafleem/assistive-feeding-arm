## What and why

<!-- One or two sentences. Link the issue. -->

**Section:** 1 mouth / 2 food / 3 grasping / 4 feeding & safety / 5 language / 6 orchestration / 7 sim / 8 props / shared core

## Checklist

- [ ] `make test` passes locally
- [ ] Touches a frozen contract (a section's `types.py`/`schema.py`/`states.py`, `robot_iface/base.py`)? If yes, the affected sections are tagged as reviewers
- [ ] Changes force/speed/standoff/keep-out numbers (`motion/safety.py`, `feeding/config.py`)? If yes, the safety lead reviewed it
- [ ] Removed `@todo` from the spec tests this PR implements
- [ ] Behaviour changed? Sim eval table pasted below (head contacts and false claims must stay 0)
- [ ] No secrets (Viam keys, API keys), model weights or recorded data committed

## Eval (if behaviour changed)

```
paste scripts/eval_feeding.py output (once it exists)
```
