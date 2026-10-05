from contextlib import asynccontextmanager

from sqlalchemy import BigInteger, func, literal, select
from sqlalchemy.ext.asyncio import AsyncConnection

# Two-key corpus locks cannot collide with single-bigint chat locks.
CORPUS_LOCK = (78412, 1)
INGEST_LOCK = (78412, 2)


@asynccontextmanager
async def advisory_lock(
    conn: AsyncConnection, key: int | tuple[int, int], error: Exception, *, shared: bool = False
):
    """Hold a PostgreSQL session lock without an open transaction during caller work."""
    args = key if isinstance(key, tuple) else (literal(key, BigInteger()),)
    acquire = func.pg_try_advisory_lock_shared if shared else func.pg_try_advisory_lock
    release = func.pg_advisory_unlock_shared if shared else func.pg_advisory_unlock
    acquired = False
    try:
        async with conn.begin():
            acquired = await conn.scalar(select(acquire(*args)))
        if not acquired:
            raise error
        yield
    finally:
        if acquired:
            try:
                async with conn.begin():
                    await conn.execute(select(release(*args)))
            except BaseException:
                # Never keep a physical session with an uncertain lock state alive.
                await conn.invalidate()
                raise
