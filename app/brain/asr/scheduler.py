"""One non-preemptible inference at a time; bounded queues, aging, owner isolation."""
import asyncio
from dataclasses import dataclass, field
import time
import logging


class Busy(Exception):
    pass


@dataclass
class Job:
    owner: str
    source: str
    kind: str
    priority: int
    pcm: bytes
    future: asyncio.Future
    client_id: str = ''
    request_id: str = ''
    created: float = field(default_factory=time.monotonic)


class Scheduler:
    def __init__(self, engine, limit=32):
        self.engine, self.limit = engine, limit
        self.pending = []
        self.wake = asyncio.Event()
        self.active = None
        self.closed = False
        self.stats = dict(completed=0, errors=0, cancelled=0, rejected=0, superseded=0,
                          inference_seconds=0., queue_seconds=0.)
        self.by_source = {}

    def submit(self, owner, source, kind, priority, pcm, *, client_id='', request_id=''):
        # Partial hypotheses are replaceable; final supersedes queued partials.
        if kind in ('partial', 'final'):
            for job in list(self.pending):
                if job.owner == owner and job.kind == 'partial':
                    self.pending.remove(job)
                    job.future.cancel()
                    self.stats['superseded'] += 1
        own = sum(j.owner == owner for j in self.pending)
        if self.closed or len(self.pending) >= self.limit or own >= 2:
            self.stats['rejected'] += 1
            raise Busy('ASR queue full')
        future = asyncio.get_running_loop().create_future()
        self.pending.append(Job(owner, source, kind, priority, pcm, future,
                                client_id=client_id, request_id=request_id))
        self.wake.set()
        return future

    def cancel(self, owner):
        for job in list(self.pending):
            if job.owner == owner:
                self.pending.remove(job)
                job.future.cancel()
                self.stats['cancelled'] += 1
        if self.active and self.active.owner == owner:
            self.active.future.cancel()
        # The GPU operation itself continues, but its result is discarded.

    async def run(self):
        while not self.closed:
            if not self.pending:
                self.wake.clear()
                await self.wake.wait()
                continue
            now = time.monotonic()
            # Final/upload beats partial; foreground beats background. Aging
            # crosses priority classes so sustained foreground cannot starve work.
            job = min(self.pending, key=lambda j: (
                j.priority + (20 if j.kind == 'partial' else 0)
                - (now - j.created) * 5, j.created))
            self.pending.remove(job)
            if job.future.cancelled():
                self.stats['cancelled'] += 1
                job.pcm = b''
                job = None
                continue
            self.active = job
            started = time.monotonic()
            outcome = 'ok'
            self.stats['queue_seconds'] += started - job.created
            try:
                text = await self.engine.transcribe(job.pcm)
                if not job.future.done():
                    job.future.set_result(text)
                self.stats['completed'] += 1
                self.by_source[job.source] = self.by_source.get(job.source, 0) + 1
            except asyncio.CancelledError:
                outcome = 'cancelled'
                job.future.cancel()
                raise
            except Exception:
                outcome = 'error'
                self.stats['errors'] += 1
                if not job.future.done():
                    job.future.set_exception(RuntimeError('recognition_failed'))
            finally:
                if job.future.cancelled():
                    self.stats['cancelled'] += 1
                self.stats['inference_seconds'] += time.monotonic() - started
                logging.getLogger('asr.jobs').info(
                    'source=%s client=%s request=%s kind=%s queue_ms=%d inference_ms=%d outcome=%s',
                    job.source, job.client_id, job.request_id, job.kind, (started - job.created) * 1000,
                    (time.monotonic() - started) * 1000,
                    'discarded' if job.future.cancelled() else outcome)
                job.pcm = b''
                self.active = None
                # Do not retain the last recording/result while the worker idles.
                text = None
                job = None

    async def close(self):
        self.closed = True
        for job in self.pending:
            job.future.cancel()
        self.pending.clear()
        self.wake.set()
