"""
Async bridge connecting synchronous 3-tier pipeline to Chainlit's async WebSocket.

Runs the sync LegalGraphRAG.query_stream() in a background thread,
forwards (event_type, data) tuples via asyncio.Queue to the async consumer.
"""

import asyncio
import threading
from typing import Any, AsyncGenerator, Tuple


class AsyncRetrievalBridge:
    """Bridge synchronous 3-tier pipeline to async Chainlit WebSocket."""

    def __init__(self, graphrag, domain: str = "legal"):
        self.graphrag = graphrag
        self.domain = domain

    async def stream_query(
        self, query: str
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """Async generator that yields (event_type, data) from sync pipeline."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_sync():
            try:
                for event_type, data in self.graphrag.query_stream(
                    query, domain=self.domain
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", str(e))
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        thread = threading.Thread(target=_run_sync, daemon=True)
        thread.start()

        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                break
            yield event_type, data
