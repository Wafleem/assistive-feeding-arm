## What and why

<!-- One or two sentences. Link the issue. -->

**Workstream:** A voice / B perception / C skills / D sim / E hardware / F VLA / P props / S safety

## Checklist

- [ ] `make test` passes locally
- [ ] Touches a frozen contract (`intent/schema.py`, `robot_iface/base.py`, `perception/types.py`, `skills/base.py`, `planning/safety.py`, eval world checks)? If yes, the affected streams are tagged as reviewers
- [ ] Changes force/speed/keep-out numbers? If yes, the safety lead reviewed it
- [ ] Behaviour changed? `make eval` table pasted below (world ok / FALSE / user hit must not get worse)
- [ ] No secrets (Viam keys, API keys), model weights or recorded data committed

## Eval (if behaviour changed)

```
paste scripts/eval_sim.py output
```
