#!/usr/bin/env python3
"""Shared, authenticated HTTP + WebSocket ASR. No HA/model control side effects."""
import argparse
import asyncio
from contextlib import suppress
import io
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import ssl
import struct
import sys
import time
import uuid
import wave

# Project-local optional dependency installation, avoids modifying system Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data/asr/python-deps'))
from aiohttp import web, WSMsgType
import webrtcvad

try:
    from .debug import DebugStore, install_debug
    from .auth import Credentials, DEFAULT_KEYS, allowed_ip
    from .engine import WhisperEngine, MODEL, DEFAULT_INITIAL_PROMPT
    from .scheduler import Scheduler, Busy
    from .segmentation import Segmenter, FRAME
except ImportError:
    from debug import DebugStore, install_debug
    from auth import Credentials, DEFAULT_KEYS, allowed_ip
    from engine import WhisperEngine, MODEL, DEFAULT_INITIAL_PROMPT
    from scheduler import Scheduler, Busy
    from segmentation import Segmenter, FRAME

PRIORITY = {'watch': 0, 'local': 0, 'background': 30, 'legacy-http': 10}
EVENTS = logging.getLogger('asr.events')
DEFAULT_LOG = Path(__file__).resolve().parents[3] / 'data/asr/logs/asr.log'


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 64 and all(
        c.isascii() and (c.isalnum() or c in '_-') for c in value)


class Stream:
    def __init__(self, service, ws, client):
        self.service, self.ws, self.client = service, ws, client
        self.session = uuid.uuid4().hex
        self.request = None
        self.utterance = None
        self.segmenter = None
        self.recording = False
        self.sequence = 0
        self.owners = set()
        self.tasks = set()
        self.finished = set()
        self.revision = 0
        self.generation = 0
        self.audio_bytes = 0
        self.audio_packets = 0
        self.last_audio_at = None
        self.debug_id = None

    async def send(self, state, **data):
        phase = data.get('code') if state == 'error' else data.get('phase', state)
        if phase != 'pong':
            self.service.debug.event(state, client_id=self.client['id'], source=self.client['source'],
                session_id=self.session, request_id=self.request, detail=phase)
        if phase in ('stopped', 'cancelled') or state == 'error':
            self.service.debug.status(self.debug_id, phase)
        if not self.ws.closed:
            await self.ws.send_json(dict(state=state, client_id=self.client['id'],
                source=self.client['source'], session_id=self.session,
                request_id=self.request, **data))

    def cancel(self):
        self.generation += 1
        self.recording = False
        self.segmenter = None
        for owner in self.owners:
            self.service.scheduler.cancel(owner)
        self.owners.clear()
        self.finished.clear()
        for task in self.tasks:
            task.cancel()

    async def enqueue(self, kind, pcm):
        utterance, request, generation = self.utterance, self.request, self.generation
        debug_id = self.debug_id
        if kind == 'final':
            self.finished.add(utterance)
        owner = f'{self.session}:{utterance}'
        self.owners.add(owner)
        self.revision += 1
        revision = self.revision
        try:
            future = self.service.scheduler.submit(owner, self.client['source'], kind,
                PRIORITY[self.client['source']], pcm, client_id=self.client['id'], request_id=request)
        except Busy:
            if kind == 'final':
                self.service.scheduler.cancel(owner)
                self.owners.discard(owner)
                self.finished.discard(utterance)
                await self.send('error', code='busy', utterance_id=utterance,
                                message='识别队列已满，请稍后重试')
            return
        submitted = time.monotonic()

        async def deliver():
            try:
                text = await asyncio.wait_for(future, 120)
                if generation != self.generation or self.ws.closed:
                    return
                if kind == 'partial' and utterance in self.finished:
                    return
                await self.ws.send_json(dict(state=kind, client_id=self.client['id'],
                    source=self.client['source'], session_id=self.session, request_id=request,
                    utterance_id=utterance, revision=revision, text=text,
                    elapsed_ms=round((time.monotonic() - submitted) * 1000)))
                self.service.debug.result(debug_id, kind, utterance, text,
                    round((time.monotonic() - submitted) * 1000))
                self.service.debug.event(kind, client_id=self.client['id'], request_id=request,
                    session_id=self.session, detail='result_sent', text_chars=len(text))
                EVENTS.info('event=result_sent client=%s request=%s kind=%s text_chars=%d',
                            self.client['id'], request, kind, len(text))
                if kind == 'final':
                    self.service.final_seconds += time.monotonic() - submitted
                    self.service.finals += 1
            except asyncio.CancelledError:
                return
            except (Exception, asyncio.TimeoutError):
                if generation == self.generation:
                    await self.send('error', code='recognition_failed', utterance_id=utterance)
            finally:
                if kind == 'final':
                    self.service.scheduler.cancel(owner)
                    self.owners.discard(owner)
                    self.finished.discard(utterance)

        task = asyncio.create_task(deliver())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def run(self):
        await self.send('listening', phase='ready', protocol=1,
                        format='pcm_s16le', sample_rate=16000, channels=1)
        try:
            while not self.ws.closed:
                try:
                    msg = await asyncio.wait_for(self.ws.receive(), 10 if self.recording else 60)
                except asyncio.TimeoutError:
                    await self.send('error', code='timeout')
                    break
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError('invalid_message')
                    action = data.get('type')
                    if action == 'start':
                        if self.recording or self.owners:
                            await self.send('error', code='request_in_progress')
                            continue
                        if not identifier(data.get('request_id')):
                            raise ValueError('invalid_request_id')
                        if data.get('sample_rate', 16000) != 16000 or data.get('channels', 1) != 1 or data.get('format', 'pcm_s16le') != 'pcm_s16le':
                            raise ValueError('unsupported_audio_format')
                        self.request = data['request_id']
                        self.generation += 1
                        self.segmenter = Segmenter(self.service.vad_factory(),
                            self.service.args.silence_ms, self.service.args.partial_ms)
                        self.recording = True
                        self.sequence = 0
                        self.audio_bytes = 0
                        self.audio_packets = 0
                        self.last_audio_at = None
                        self.started = time.monotonic()
                        self.utterance = uuid.uuid4().hex
                        self.revision = 0
                        self.debug_id = self.service.debug.begin(self.client, self.session, self.request)
                        await self.send('listening', phase='recording', next_sequence=0)
                        EVENTS.info('event=recording_started client=%s request=%s',
                                    self.client['id'], self.request)
                    elif action == 'stop':
                        if not self.recording:
                            await self.send('error', code='not_recording')
                            continue
                        pcm = self.segmenter.finish()
                        self.recording = False
                        if pcm:
                            await self.enqueue('final', pcm)
                        await self.send('listening', phase='stopped',
                                        discarded_utterance_id=None if pcm else self.utterance)
                    elif action == 'cancel':
                        self.cancel()
                        await self.send('listening', phase='cancelled')
                    elif action == 'ping':
                        await self.send('listening', phase='pong')
                    else:
                        raise ValueError('unknown_command')
                elif msg.type == WSMsgType.BINARY:
                    if not self.recording:
                        raise ValueError('start_required')
                    payload = msg.data
                    if len(payload) < 4 + FRAME or len(payload) > 4 + FRAME * 10 or (len(payload) - 4) % FRAME:
                        raise ValueError('invalid_audio_packet')
                    sequence = struct.unpack_from('<I', payload)[0]
                    if sequence != self.sequence:
                        raise ValueError('sequence_mismatch_restart_recording')
                    self.sequence += 1
                    previous_bytes = self.audio_bytes
                    self.audio_bytes += len(payload) - 4
                    if time.monotonic() - self.started > 60 or self.audio_bytes > 60 * 32000:
                        raise ValueError('recording_limit_60_seconds')
                    # Limit accelerated uploads: at most 2 seconds ahead of real time.
                    if self.audio_bytes / 32000 > time.monotonic() - self.started + 2:
                        raise ValueError('audio_rate_limit')
                    if self.debug_id not in self.service.debug.records and self.service.debug.enabled:
                        self.debug_id = self.service.debug.begin(self.client, self.session, self.request,
                                                                 incomplete=previous_bytes > 0)
                    self.service.debug.audio(self.debug_id, payload[4:])
                    self.audio_packets += 1
                    self.last_audio_at = time.monotonic()
                    self.service.received_audio_bytes += len(payload) - 4
                    self.service.received_audio_packets += 1
                    if previous_bytes == 0 or self.audio_bytes // 32000 != previous_bytes // 32000:
                        EVENTS.info('event=audio_received client=%s request=%s packets=%d bytes=%d audio_ms=%d',
                            self.client['id'], self.request, self.audio_packets,
                            self.audio_bytes, self.audio_bytes / 32)
                    for offset in range(4, len(payload), FRAME):
                        for kind, pcm in self.segmenter.feed(payload[offset:offset + FRAME]):
                            if kind == 'speech':
                                EVENTS.info('event=speech_detected client=%s request=%s',
                                            self.client['id'], self.request)
                                await self.send('listening', phase='speech', utterance_id=self.utterance)
                            elif kind == 'discarded':
                                await self.send('listening', phase='no_speech', utterance_id=self.utterance)
                                self.utterance = uuid.uuid4().hex
                                self.revision = 0
                            else:
                                await self.enqueue(kind, pcm)
                                if kind == 'final':
                                    self.utterance = uuid.uuid4().hex
                                    self.revision = 0
                    if len(self.owners) > 2:
                        raise ValueError('too_many_pending_utterances')
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except (ValueError, json.JSONDecodeError) as error:
            EVENTS.info('event=recording_error client=%s request=%s code=%s',
                self.client['id'], self.request,
                'invalid_json' if isinstance(error, json.JSONDecodeError) else str(error))
            await self.send('error', code=str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else 'invalid_json')
        finally:
            self.cancel()
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            await self.ws.close()


class Service:
    def __init__(self, args, engine, vad_factory=None):
        self.args, self.engine = args, engine
        self.debug = DebugStore()
        self.credentials = Credentials(args.keys)
        self.scheduler = Scheduler(engine)
        self.vad_factory = vad_factory or (lambda: webrtcvad.Vad(2))
        self.sessions = {}
        self.http_active = {}
        self.connections = 0
        self.finals = 0
        self.final_seconds = 0.
        self.received_audio_bytes = 0
        self.received_audio_packets = 0

    def identify_http(self, request):
        auth = request.headers.get('Authorization', '')
        client = self.credentials.identify(auth[7:]) if auth.startswith('Bearer ') else None
        if client:
            return client
        # Explicit backward-compatible endpoint only, never trust caller-supplied source/priority.
        if not auth and not self.args.require_local_auth and ipaddress.ip_address(request.remote).is_loopback:
            return dict(id='legacy-http', source='legacy-http')
        raise web.HTTPUnauthorized(text='authentication required')

    async def ws(self, request):
        if self.connections >= 16:
            raise web.HTTPServiceUnavailable(text='connection limit')
        ws = web.WebSocketResponse(max_msg_size=8192, heartbeat=15, compress=False)
        self.connections += 1
        client = None
        try:
            await ws.prepare(request)
            self.debug.event('connected', peer=request.remote, detail='awaiting_authentication')
            EVENTS.info('event=ws_connected peer=%s', request.remote)
            # Authentication is the first text frame, never a URL query or log field.
            message = await asyncio.wait_for(ws.receive(), 5)
            if message.type != WSMsgType.TEXT:
                raise ValueError('authentication_required')
            data = json.loads(message.data)
            client = self.credentials.identify(data.get('token')) if isinstance(data, dict) and data.get('type') == 'auth' else None
            if not client:
                raise ValueError('authentication_failed')
            if client['id'] in self.sessions:
                raise ValueError('client_already_connected')
            stream = Stream(self, ws, client)
            self.sessions[client['id']] = stream
            EVENTS.info('event=authenticated client=%s source=%s session=%s',
                        client['id'], client['source'], stream.session)
            try:
                await stream.run()
            finally:
                self.sessions.pop(client['id'], None)
                self.debug.status(stream.debug_id, 'disconnected')
                self.debug.event('disconnected', client_id=client['id'], source=client['source'],
                    session_id=stream.session, request_id=stream.request)
                EVENTS.info('event=disconnected client=%s session=%s', client['id'], stream.session)
        except (ValueError, asyncio.TimeoutError):
            self.debug.event('error', peer=request.remote, detail='authentication_failed_or_duplicate')
            EVENTS.info('event=authentication_failed_or_duplicate peer=%s', request.remote)
            if not ws.closed:
                await ws.send_json(dict(state='error', code='authentication_failed_or_duplicate'))
        finally:
            self.connections -= 1
            await ws.close()
        return ws

    async def upload(self, request):
        client = self.identify_http(request)
        if request.content_type != 'multipart/form-data':
            raise web.HTTPUnsupportedMediaType(text='multipart file required')
        if self.http_active.get(client['id'], 0) >= 2:
            raise web.HTTPTooManyRequests(text='per-client request limit')
        request_id = request.headers.get('X-Request-ID', uuid.uuid4().hex)
        if not identifier(request_id):
            raise web.HTTPBadRequest(text='invalid request id')
        owner = uuid.uuid4().hex
        debug_id = None
        self.http_active[client['id']] = self.http_active.get(client['id'], 0) + 1
        try:
            reader = await request.multipart()
            audio = None
            total = 0
            async for part in reader:
                content = bytearray()
                while chunk := await asyncio.wait_for(part.read_chunk(8192), 10):
                    total += len(chunk)
                    if total > 60 * 32000 + 65536:
                        raise web.HTTPRequestEntityTooLarge(max_size=1985536, actual_size=total)
                    content.extend(chunk)
                if part.name == 'file':
                    audio = bytes(content)
            if audio is None:
                raise ValueError('file field required')
            with wave.open(io.BytesIO(audio)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE'):
                    raise ValueError('expected 16kHz mono PCM16 WAV')
                if wav.getnframes() > 60 * 16000 or not wav.getnframes():
                    raise ValueError('WAV duration must be 0–60 seconds')
                pcm = wav.readframes(wav.getnframes())
                if len(pcm) != wav.getnframes() * 2:
                    raise ValueError('truncated WAV')
            debug_id = self.debug.begin(client, owner, request_id, transport='http')
            self.debug.audio(debug_id, pcm)
            vad = self.vad_factory()
            voiced = sum(vad.is_speech(pcm[i:i + FRAME], 16000)
                         for i in range(0, len(pcm) - FRAME + 1, FRAME))
            if voiced < 13:
                self.debug.status(debug_id, 'no_speech')
                return web.json_response(dict(text='', request_id=request_id,
                    client_id=client['id'], source=client['source']))
            future = self.scheduler.submit(owner, client['source'], 'upload', PRIORITY[client['source']],
                                           pcm, client_id=client['id'], request_id=request_id)
            submitted = time.monotonic()
            text = await asyncio.wait_for(future, 120)
            self.debug.result(debug_id, 'final', owner, text, round((time.monotonic()-submitted)*1000))
            self.debug.status(debug_id, 'completed')
            return web.json_response(dict(text=text, request_id=request_id,
                client_id=client['id'], source=client['source']))
        except Busy:
            self.debug.status(debug_id, 'busy')
            self.debug.event('error', client_id=client['id'], request_id=request_id, detail='http_busy')
            raise web.HTTPTooManyRequests(text='ASR queue full')
        except (ValueError, wave.Error, EOFError):
            raise web.HTTPBadRequest(text='invalid multipart/WAV audio')
        except (asyncio.TimeoutError, RuntimeError):
            self.debug.status(debug_id, 'recognition_unavailable')
            self.debug.event('error', client_id=client['id'], request_id=request_id, detail='recognition_unavailable')
            raise web.HTTPServiceUnavailable(text='recognition unavailable')
        finally:
            self.scheduler.cancel(owner)
            self.http_active[client['id']] -= 1

    async def health(self, request):
        ready = getattr(self.engine, 'ready', True)
        return web.json_response(dict(status='ready' if ready else 'unavailable',
            backend=self.engine.backend, sample_rate=16000, protocol=1,
            language=getattr(self.engine, 'language', None),
            output_script='Hans' if isinstance(self.engine, WhisperEngine) else None,
            translate=False), status=200 if ready else 503)

    async def metrics(self, request):
        if not ipaddress.ip_address(request.remote).is_loopback:
            raise web.HTTPForbidden()
        stats = dict(self.scheduler.stats, queue_depth=len(self.scheduler.pending),
            connections=self.connections, completed_by_source=self.scheduler.by_source,
            final_count=self.finals, final_seconds=self.final_seconds)
        stats['authenticated_connections'] = len(self.sessions)
        stats['received_audio_bytes'] = self.received_audio_bytes
        stats['received_audio_packets'] = self.received_audio_packets
        stats['active_clients'] = [dict(client_id=s.client['id'], source=s.client['source'],
            session_id=s.session, request_id=s.request, recording=s.recording,
            audio_bytes=s.audio_bytes, audio_packets=s.audio_packets,
            last_audio_age_seconds=round(time.monotonic() - s.last_audio_at, 2)
                if s.last_audio_at is not None else None) for s in self.sessions.values()]
        attempts = stats['completed'] + stats['errors']
        stats['error_rate'] = stats['errors'] / attempts if attempts else 0
        # Jetson devfreq load is per-mille. Unsupported hardware explicitly reports null.
        stats['gpu_utilization_percent'] = None
        for pattern in ('/sys/class/devfreq/*gpu/load', '/sys/devices/platform/*gpu/load'):
            import glob
            for filename in glob.glob(pattern):
                with suppress(OSError, ValueError):
                    stats['gpu_utilization_percent'] = int(Path(filename).read_text()) / 10
        return web.json_response(stats)


def create_app(args, engine=None, vad_factory=None):
    engine = engine or WhisperEngine(args.backend, args.binary, args.model_path,
                                     args.worker_port, args.language, args.initial_prompt)
    service = Service(args, engine, vad_factory)

    @web.middleware
    async def boundary(request, handler):
        # Trust the TCP peer, never X-Forwarded-For. No permissive CORS.
        if not allowed_ip(request.remote, args.allow_subnet):
            raise web.HTTPForbidden(text='peer outside allowlist')
        if request.headers.get('Origin') and request.path == '/ws/asr':
            if request.headers['Origin'] not in args.allow_origin:
                raise web.HTTPForbidden(text='origin not allowed')
        try:
            if request.path == '/api/asr':
                async with asyncio.timeout(180):
                    response = await handler(request)
            else:
                response = await handler(request)
        except TimeoutError:
            raise web.HTTPRequestTimeout(text='request timeout')
        if not response.prepared:
            response.headers['Cache-Control'] = 'no-store'
        return response

    app = web.Application(middlewares=[boundary], client_max_size=2 * 1024 * 1024)
    app[web.AppKey('service', Service)] = service

    async def lifecycle(app):
        await engine.start()
        worker = asyncio.create_task(service.scheduler.run())
        try:
            yield
        finally:
            for stream in list(service.sessions.values()):
                stream.cancel()
                await stream.ws.close(code=1001, message=b'service shutdown')
            await service.scheduler.close()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            await engine.close()

    async def page(request):
        if not ipaddress.ip_address(request.remote).is_loopback:
            raise web.HTTPForbidden()
        return web.FileResponse(Path(__file__).with_name('test.html'))

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/', page)
    app.router.add_get('/asr-test', page)
    app.router.add_get('/health', service.health)
    app.router.add_get('/metrics', service.metrics)
    app.router.add_get('/ws/asr', service.ws)
    app.router.add_post('/api/asr', service.upload)
    install_debug(app, service)
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--socket-activation', action='store_true')
    parser.add_argument('--idle-seconds', type=int, default=0, help='legacy argument; persistent model no longer idles out')
    parser.add_argument('--backend', choices=['cuda', 'cpu'], default=os.getenv('ASR_BACKEND', 'cuda'))
    parser.add_argument('--binary', default=os.getenv('ASR_BINARY'))
    parser.add_argument('--model-path', default=os.getenv('ASR_MODEL_PATH', str(MODEL)))
    parser.add_argument('--worker-port', type=int, default=8002)
    parser.add_argument('--language', default=os.getenv('ASR_LANGUAGE', 'auto'))
    parser.add_argument('--initial-prompt', default=os.getenv('ASR_INITIAL_PROMPT', DEFAULT_INITIAL_PROMPT))
    parser.add_argument('--keys', default=str(DEFAULT_KEYS))
    parser.add_argument('--tls-cert')
    parser.add_argument('--tls-key')
    parser.add_argument('--lan-bind', help='Optional second WSS listener sharing the same engine')
    parser.add_argument('--lan-port', type=int, default=8444)
    parser.add_argument('--allow-subnet', action='append', default=['127.0.0.0/8', '::1/128'])
    parser.add_argument('--allow-origin', action='append', default=[])
    parser.add_argument('--require-local-auth', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--silence-ms', type=int, choices=range(800, 1201, 20), default=1000)
    parser.add_argument('--partial-ms', type=int, choices=range(500, 2001, 20), default=1000)
    args = parser.parse_args(argv)
    return args


async def serve(args):
    ssl_context = None
    if args.tls_cert and args.tls_key:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.load_cert_chain(args.tls_cert, args.tls_key)
    inherited = None
    if args.socket_activation:
        if int(os.getenv('LISTEN_PID', '0')) != os.getpid() or int(os.getenv('LISTEN_FDS', '0')) != 1:
            raise RuntimeError('invalid socket activation environment')
        inherited = socket.socket(fileno=3)
    address = inherited.getsockname()[0] if inherited else args.bind
    if not ipaddress.ip_address(address).is_loopback and ssl_context is None:
        raise RuntimeError('LAN listener requires --tls-cert and --tls-key')
    if args.lan_bind and ssl_context is None:
        raise RuntimeError('--lan-bind requires TLS certificate and key')
    engine = WhisperEngine(args.backend, args.binary, args.model_path, args.worker_port,
                           args.language, args.initial_prompt)
    app = create_app(args, engine)
    # No access/error body logger: keys, audio, transcripts do not enter logs.
    args.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.log_file, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    os.close(descriptor)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s',
        handlers=[logging.StreamHandler(), RotatingFileHandler(args.log_file,
            maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')], force=True)
    logging.getLogger('aiohttp.server').disabled = True
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=5)
    await runner.setup()
    try:
        # Separate local HTTP and LAN WSS listeners, one app/queue/model instance.
        primary_tls = None if ipaddress.ip_address(address).is_loopback else ssl_context
        primary = (web.SockSite(runner, inherited, ssl_context=primary_tls) if inherited else
                   web.TCPSite(runner, args.bind, args.port, ssl_context=primary_tls))
        await primary.start()
        if args.lan_bind:
            await web.TCPSite(runner, args.lan_bind, args.lan_port, ssl_context=ssl_context).start()
        EVENTS.info('event=service_ready backend=%s local=%s lan=%s log_file=%s',
                    engine.backend, address,
                    f'{args.lan_bind}:{args.lan_port}' if args.lan_bind else 'disabled', args.log_file)
        import signal
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        stopper = asyncio.create_task(stop.wait())
        exited = asyncio.create_task(engine.process.wait())
        try:
            done, _ = await asyncio.wait([stopper, exited], return_when=asyncio.FIRST_COMPLETED)
            if exited in done and not stop.is_set():
                raise RuntimeError('Whisper backend exited; restart service to reload model')
        finally:
            stopper.cancel()
            exited.cancel()
            await asyncio.gather(stopper, exited, return_exceptions=True)
    finally:
        await runner.cleanup()


def main():
    asyncio.run(serve(parse_args()))


if __name__ == '__main__':
    main()
