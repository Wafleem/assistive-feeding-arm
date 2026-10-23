# `assistive_arm` package map

```
assistive_arm/
├── geometry.py          shared core: Pose, rotations, interpolation, arcs (SI units, base frame)
├── robot_iface/         shared core: RobotInterface + FakeRobot / SimRobot / ViamRobot
├── vision/              shared core: camera types, deprojection, detector wrappers, Locator
├── motion/              shared core: static safety envelope, guarded move primitives
│
├── mouth_perception/    section 1: detect, track (Kalman), predict, ready gate, retreat triggers
├── food_perception/     section 2: menu vocabulary, open-vocab food detection → 3D items, VLM hook
├── grasping/            section 3: grasp planning (PCA, open models), acquisition strategies, VLA runner
├── feeding/             section 4: delivery modes, face keep-out zone, approach, release, retreat
├── language/            section 5: intent schema, rules, provider-agnostic LLM router, dialogue, speech
├── orchestration/       section 6: bite-cycle state machine (composes the other sections)
├── sim/                 section 7: feeding-scene behaviour (head scripts) and metrics
└── pick_place/          section 9: taught (hard-coded) pick and place on the SO-101 bench arm
```

Each folder has its own README (plan, interfaces, TODO checklist, done-when).
The rules for how these pieces may depend on each other are in [`AGENTS.md`](../../AGENTS.md).
