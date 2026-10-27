"""Opt-in, bounded in-memory ASR inspection. Never writes recordings to disk."""
import asyncio
from contextlib import suppress
from collections import deque
import io
import ipaddress
from pathlib import Path
import secrets
import time
import uuid
import wave
from urllib.parse import urlsplit
from aiohttp import web


class DebugStore:
    TTL = 600
    MAX_BYTES = 16 * 1024 * 1024
    MAX_RECORDS = 20

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.enabled_until = 0
        self.token = secrets.token_urlsafe(32)
        self.records = {}
        self.events = deque(maxlen=300)

    @property
    def enabled(self):
        return self.clock() < self.enabled_until

    def purge(self):
        now = self.clock()
        for key, item in list(self.records.items()):
            if now - item['created_mono'] >= self.TTL:
                del self.records[key]
        while self.records and (len(self.records) > self.MAX_RECORDS or
                sum(len(r['pcm']) for r in self.records.values()) > self.MAX_BYTES):
            del self.records[next(iter(self.records))]

    def event(self, state, **fields):
        # Only caller-selected metadata: no authentication frames, exceptions or keys.
        self.events.append(dict(time=time.time(), state=state, **fields))

    def begin(self, client, session, request, transport='websocket', incomplete=False):
        self.purge()
        if not self.enabled:
            return None
        key = uuid.uuid4().hex
        self.records[key] = dict(id=key, client_id=client['id'], source=client['source'],
            session_id=session, request_id=request, transport=transport, time=time.time(),
            created_mono=self.clock(), pcm=bytearray(), results={}, status='recording',
            incomplete=incomplete)
        self.purge()
        return key

    def audio(self, key, pcm):
        self.purge()
        if self.enabled and key in self.records:
            item = self.records[key]
            if len(item['pcm']) + len(pcm) <= 60 * 32000:
                item['pcm'].extend(pcm)
            else:
                item['incomplete'] = True
            self.purge()
        elif key in self.records:
            self.records[key]['incomplete'] = True

    def result(self, key, kind, utterance, text, elapsed_ms):
        self.purge()
        if self.enabled and key in self.records:
            results = self.records[key]['results']
            if utterance not in results and len(results) >= 100:
                return
            results[utterance] = dict(kind=kind, utterance_id=utterance,
                                     text=text[:8192], elapsed_ms=elapsed_ms)

    def status(self, key, state):
        if key in self.records:
            self.records[key]['status'] = state

    def public(self, item, detail=False):
        result = {k: v for k, v in item.items() if k not in ('pcm', 'created_mono', 'results')}
        result.update(audio_bytes=len(item['pcm']), audio_seconds=round(len(item['pcm']) / 32000, 3),
                      expires_in_seconds=max(0, round(self.TTL - self.clock() + item['created_mono'])))
        if detail:
            result['results'] = list(item['results'].values())
        return result


def install_debug(app, service):
    store = service.debug
    folder = Path(__file__).with_name('test')

    @web.middleware
    async def guard(request, handler):
        if request.path == '/test' or request.path.startswith('/test/'):
            host = urlsplit('//' + request.host).hostname
            try:
                local_host = host == 'localhost' or ipaddress.ip_address(host).is_loopback
            except ValueError:
                local_host = False
            if not ipaddress.ip_address(request.remote).is_loopback or not local_host:
                raise web.HTTPForbidden(text='debug dashboard is localhost-only; use SSH forwarding')
            origin = request.headers.get('Origin')
            if origin and origin != f'{request.scheme}://{request.host}':
                raise web.HTTPForbidden(text='same origin required')
            if request.method != 'GET' and request.method != 'HEAD':
                if not secrets.compare_digest(request.headers.get('X-ASR-Debug-Token', ''), store.token):
                    raise web.HTTPForbidden(text='debug control token required')
            response = await handler(request)
            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            return response
        return await handler(request)

    app.middlewares.append(guard)

    async def retention(app):
        async def sweep():
            while True:
                await asyncio.sleep(5)
                store.purge()
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            store.enabled_until = 0
            store.records.clear()
            store.events.clear()

    app.cleanup_ctx.append(retention)

    async def page(request):
        return web.FileResponse(folder / {'index.html': 'index.html', 'app.js': 'app.js',
                                        'style.css': 'style.css'}[request.match_info.get('file', 'index.html')])

    async def state(request):
        store.purge()
        return web.json_response(dict(token=store.token, enabled=store.enabled,
            capture_remaining_seconds=max(0, round(store.enabled_until - store.clock())),
            retention_seconds=store.TTL, max_bytes=store.MAX_BYTES,
            memory_bytes=sum(len(r['pcm']) for r in store.records.values()),
            records=[store.public(r) for r in reversed(list(store.records.values()))],
            events=list(store.events)))

    async def control(request):
        data = await request.json()
        if not isinstance(data, dict) or data.get('action') not in ('start', 'pause', 'clear'):
            raise web.HTTPBadRequest(text='expected action: start, pause or clear')
        if data['action'] == 'start':
            store.enabled_until = store.clock() + store.TTL
        elif data['action'] == 'pause':
            store.enabled_until = 0
            for record in store.records.values():
                if record['status'] == 'recording':
                    record['incomplete'] = True
        else:
            store.enabled_until = 0
            store.records.clear()
            store.events.clear()
        return web.json_response(dict(ok=True))

    async def record(request):
        store.purge()
        item = store.records.get(request.match_info['id'])
        if item is None:
            raise web.HTTPNotFound(text='record expired or cleared')
        kind = request.match_info.get('kind')
        if kind is None:
            return web.json_response(store.public(item, detail=True))
        pcm = bytes(item['pcm'])
        if kind == 'wav':
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm)
            body, content_type = output.getvalue(), 'audio/wav'
        else:
            body, content_type = pcm, 'application/octet-stream'
        return web.Response(body=body, content_type=content_type,
            headers={'Content-Disposition': f'attachment; filename="asr-{item["id"]}.{kind}"'})

    app.router.add_get('/test', page)
    app.router.add_get('/test/', page)
    app.router.add_get('/test/{file:index.html|app.js|style.css}', page)
    app.router.add_get('/test/api/state', state)
    app.router.add_post('/test/api/control', control)
    app.router.add_get('/test/api/records/{id}', record)
    app.router.add_get('/test/api/records/{id}/audio.{kind:wav|pcm}', record)
