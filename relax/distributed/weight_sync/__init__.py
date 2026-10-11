# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-independent parts of sparse delta weight sync.

- :mod:`.wire`: bucket and control-message format;
- :mod:`.store`: immutable version packages, epochs, ``HEAD``, planning and
  retention;
- :mod:`.tcp`: push (in-job) and pull transport of packages;
- :mod:`.publisher`: the training side, which writes and publishes packages
  and holds no rollout engine handle;
- :mod:`.installer`: the inference side, which brings one engine to a
  published version from any package source without the trainer;
- :mod:`.consume`: offline consumer CLI.

The publisher and the installer only meet through packages, so the in-job
sync (``DeviceDirectBackend``) and an external orchestrator can drive them
alike. None of these modules imports Ray, Megatron or SGLang.
"""
