# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit ASGI boundary for a consumer-owned native SGLang HTTP engine."""

import json
from typing import Any


class ConsumerHTTP:
    """Expose native generate/encode routes and refuse legacy control routes.

    The tokenizer and every scheduler still enforce admission. This outer
    boundary additionally bars HTTP handlers that mutate global configuration
    directly without calling a tokenizer control method.
    """

    def __init__(self, application: Any, tokenizer: Any, shutdown: Any = None):
        self.application, self.tokenizer, self.shutdown = application, tokenizer, shutdown

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            # The engine is already initialized by its consumer. Do not run
            # the upstream full-server warmup/tool-server/sidecar lifecycle.
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    self.tokenizer.delta_ticket = None
                    self.tokenizer.delta_poisoned = True
                    if self.shutdown is not None:
                        self.shutdown()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] == "http":
            error = None
            if scope["path"] not in ("/generate", "/encode"):
                error = "This endpoint is disabled for a consumer-owned engine."
            elif self.tokenizer.delta_ticket is None or self.tokenizer.delta_poisoned:
                error = "Consumer is not active."
            if error is not None:
                body = json.dumps({"error": {"message": error}}).encode()
                await send(
                    {"type": "http.response.start", "status": 409, "headers": [(b"content-type", b"application/json")]}
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.application(scope, receive, send)
