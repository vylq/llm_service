import asyncio

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


class Tracing:
    """One optional Langfuse client per process, one callback per request."""

    def __init__(self, settings):
        self.public_key = settings.langfuse_public_key
        self.client = None
        if settings.langfuse_enabled:
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
            )

    def callbacks(self):
        return [CallbackHandler(public_key=self.public_key)] if self.client else []

    async def close(self):
        if self.client:
            await asyncio.to_thread(self.client.shutdown)
