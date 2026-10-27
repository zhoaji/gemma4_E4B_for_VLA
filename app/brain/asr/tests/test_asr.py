"""No GPU required: deterministic lifecycle, routing, scheduling and wire tests."""
import asyncio
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'data/asr/python-deps'))
sys.path.insert(0, str(ROOT))
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from app.brain.asr.engine import wav_bytes
from app.brain.asr.auth import Credentials
from app.brain.asr.engine import WhisperEngine
from app.brain.asr.scheduler import Scheduler, Busy
from app.brain.asr.segmentation import Segmenter
from app.brain.asr.server import create_app, parse_args, serve

VOICE = b'\x01\x00' * 320
QUIET = b'\x00' * 640


class PairingTests(unittest.TestCase):
    def test_pair_rotate_revoke_and_no_plaintext_token_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'keys.json'
            command = [sys.executable, str(ROOT / 'app/brain/asr/auth.py'),
                       '--file', str(file), '--id', 'test-local', '--source', 'local']
            def pair():
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                return next(line[6:] for line in result.stdout.splitlines() if line.startswith('token='))
            first = pair()
            credentials = Credentials(file)
            self.assertEqual(credentials.identify(first)['source'], 'local')
            self.assertNotIn(first, file.read_text())
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            second = pair()
            self.assertIsNone(credentials.identify(first))
            self.assertEqual(credentials.identify(second)['id'], 'test-local')
            subprocess.run([*command, '--revoke'], capture_output=True, check=True)
            self.assertIsNone(credentials.identify(second))


class TextOutputTests(unittest.TestCase):
    def test_traditional_is_simplified_without_translating_english(self):
        engine = WhisperEngine()
        self.assertEqual(engine.normalize_text(' 請打開臥室燈，亮度調到30%。 '),
                         '请打开卧室灯，亮度调到30%。')
        self.assertEqual(engine.normalize_text('請打開 Wi-Fi 和 Bluetooth，使用 Home Assistant。'),
                         '请打开 Wi-Fi 和 Bluetooth，使用 Home Assistant。')
        self.assertEqual(engine.normalize_text('Good morning! Turn on the light.'),
                         'Good morning! Turn on the light.')


class InferenceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_inference_explicitly_transcribes_and_normalizes(self):
        import aiohttp
        from aiohttp import web
        requests = []

        async def inference(request):
            data = await request.post()
            requests.append({key: value for key, value in data.items() if key != 'file'})
            return web.json_response({'text': '請打開 Wi-Fi。' if len(requests) == 1 else 'Good morning!'})

        app = web.Application()
        app.router.add_post('/inference', inference)
        server = TestServer(app)
        await server.start_server()
        engine = WhisperEngine(port=server.port)
        engine.client = aiohttp.ClientSession()
        try:
            self.assertEqual(await engine.transcribe(QUIET), '请打开 Wi-Fi。')
            self.assertEqual(await engine.transcribe(QUIET), 'Good morning!')
            for fields in requests:
                self.assertEqual(fields['language'], 'auto')
                self.assertEqual(fields['translate'], 'false')
                self.assertEqual(fields['detect_language'], 'false')
                self.assertEqual(fields['temperature'], '0')
                self.assertEqual(fields['prompt'], engine.initial_prompt)
        finally:
            await engine.close()
            await server.close()


class VAD:
    def is_speech(self, pcm, rate):
        return pcm != QUIET


class Engine:
    backend = 'test'
    def __init__(self):
        self.calls = []
        self.running = 0
        self.maximum = 0
        self.gate = None
    async def start(self):
        pass
    async def close(self):
        pass
    async def transcribe(self, pcm):
        self.running += 1
        self.maximum = max(self.maximum, self.running)
        self.calls.append(pcm)
        try:
            if self.gate:
                await self.gate.wait()
            await asyncio.sleep(.005)
            return f'result-{pcm[0]}'
        finally:
            self.running -= 1


class SegmentationTests(unittest.TestCase):
    def test_silence_and_short_noise_not_transcribed(self):
        s = Segmenter(VAD())
        for _ in range(100):
            self.assertEqual(s.feed(QUIET), [])
        for _ in range(5):
            s.feed(VOICE)
        self.assertIsNone(s.finish())

    def test_partial_and_final_after_one_second_silence(self):
        s = Segmenter(VAD())
        events = []
        for frame in [VOICE] * 60 + [QUIET] * 50:
            events += s.feed(frame)
        self.assertIn('partial', [x[0] for x in events])
        self.assertEqual(sum(kind == 'final' for kind, _ in events), 1)
        self.assertTrue(next(pcm for kind, pcm in events if kind == 'final').startswith(VOICE))


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timed_out_future_is_not_inferred(self):
        engine = Engine()
        scheduler = Scheduler(engine)
        timed_out = scheduler.submit('a', 'watch', 'final', 0, b'a')
        timed_out.cancel()
        live = scheduler.submit('b', 'local', 'final', 0, b'b')
        worker = asyncio.create_task(scheduler.run())
        await live
        self.assertEqual(engine.calls, [b'b'])
        await scheduler.close()
        await worker

    async def test_gpu_missing_never_falls_back_and_lan_requires_tls(self):
        with self.assertRaises(RuntimeError):
            await WhisperEngine(binary='/missing/whisper', model='/missing/model').start()
        with self.assertRaises(RuntimeError):
            await serve(parse_args(['--bind', '0.0.0.0']))
        with self.assertRaises(RuntimeError):
            await serve(parse_args(['--lan-bind', '192.168.1.204']))

    async def test_priorities_coalescing_cancel_and_serial_inference(self):
        engine = Engine()
        scheduler = Scheduler(engine)
        partial = scheduler.submit('a', 'watch', 'partial', 0, b'a')
        replaced = scheduler.submit('a', 'watch', 'partial', 0, b'b')
        self.assertTrue(partial.cancelled())
        background = scheduler.submit('b', 'background', 'upload', 30, b'c')
        final = scheduler.submit('a', 'watch', 'final', 0, b'd')
        self.assertTrue(replaced.cancelled())
        cancelled = scheduler.submit('c', 'local', 'final', 0, b'e')
        scheduler.cancel('c')
        self.assertTrue(cancelled.cancelled())
        worker = asyncio.create_task(scheduler.run())
        self.assertEqual(await final, 'result-100')
        await background
        self.assertEqual(engine.calls, [b'd', b'c'])
        self.assertEqual(engine.maximum, 1)
        await scheduler.close()
        await worker

    async def test_aging_and_queue_bound(self):
        engine = Engine()
        scheduler = Scheduler(engine, limit=2)
        old = scheduler.submit('old', 'background', 'upload', 30, b'o')
        scheduler.pending[0].created -= 20
        new = scheduler.submit('new', 'watch', 'final', 0, b'n')
        with self.assertRaises(Busy):
            scheduler.submit('third', 'watch', 'final', 0, b't')
        worker = asyncio.create_task(scheduler.run())
        await asyncio.gather(old, new)
        self.assertEqual(engine.calls[0], b'o')
        await scheduler.close()
        await worker


class WireTests(unittest.IsolatedAsyncioTestCase):
    async def test_dead_backend_is_not_reported_ready(self):
        self.engine.ready = False
        response = await self.client.get('/health')
        self.assertEqual(response.status, 503)
        self.assertEqual((await response.json())['status'], 'unavailable')

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        keys = Path(self.directory.name) / 'clients.json'
        keys.write_text(json.dumps({'clients': [dict(id=x, source=source,
            sha256=hashlib.sha256(x.encode()).hexdigest()) for x, source in
            [('watch1', 'watch'), ('local1', 'local'), ('batch1', 'background')]]}))
        self.engine = Engine()
        args = parse_args(['--keys', str(keys), '--backend', 'cpu'])
        self.app = create_app(args, self.engine, VAD)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.directory.cleanup()

    async def connect(self, token='watch1'):
        ws = await self.client.ws_connect('/ws/asr')
        await ws.send_json(dict(type='auth', token=token, source='background', priority=-999))
        message = await ws.receive_json()
        self.assertEqual(message['state'], 'listening')
        return ws, message

    async def start(self, ws, request='request1'):
        await ws.send_json(dict(type='start', request_id=request))
        self.assertEqual((await ws.receive_json())['phase'], 'recording')

    async def send_voice(self, ws, marker=1, count=20):
        pcm = bytes([marker, 0]) * 320
        for seq in range(count // 5):
            await ws.send_bytes(struct.pack('<I', seq) + pcm * 5)

    async def until(self, ws, state):
        for _ in range(20):
            value = await asyncio.wait_for(ws.receive_json(), 3)
            if value['state'] == state:
                return value
        self.fail(f'no {state}')

    async def test_two_clients_results_are_routed_and_source_is_not_spoofable(self):
        a, ma = await self.connect()
        b, mb = await self.connect('local1')
        self.assertNotEqual(ma['session_id'], mb['session_id'])
        self.assertEqual(ma['source'], 'watch')
        await self.start(a, 'A')
        await self.start(b, 'B')
        await self.send_voice(a, 1)
        await self.send_voice(b, 2)
        await a.send_json(dict(type='stop'))
        await b.send_json(dict(type='stop'))
        fa, fb = await asyncio.gather(self.until(a, 'final'), self.until(b, 'final'))
        self.assertEqual((fa['text'], fa['request_id'], fa['client_id']), ('result-1', 'A', 'watch1'))
        self.assertEqual((fb['text'], fb['request_id'], fb['client_id']), ('result-2', 'B', 'local1'))
        self.assertEqual(self.engine.maximum, 1)
        await a.close()
        await b.close()

    async def test_cancel_in_flight_suppresses_old_results_and_restart_works(self):
        self.engine.gate = asyncio.Event()
        ws, _ = await self.connect()
        await self.start(ws, 'old')
        await self.send_voice(ws)
        await ws.send_json(dict(type='stop'))
        await self.until(ws, 'listening')  # speech or stopped
        await asyncio.sleep(.03)
        await ws.send_json(dict(type='cancel'))
        while (await ws.receive_json()).get('phase') != 'cancelled':
            pass
        self.engine.gate.set()
        await self.start(ws, 'new')
        await self.send_voice(ws, 3)
        await ws.send_json(dict(type='stop'))
        final = await self.until(ws, 'final')
        self.assertEqual(final['request_id'], 'new')
        self.assertEqual(final['text'], 'result-3')
        await ws.close()

    async def test_auth_bad_sequence_duplicate_and_reconnect(self):
        bad = await self.client.ws_connect('/ws/asr')
        await bad.send_json(dict(type='auth', token='wrong'))
        self.assertEqual((await bad.receive_json())['state'], 'error')
        await bad.close()
        ws, initial = await self.connect()
        duplicate = await self.client.ws_connect('/ws/asr')
        await duplicate.send_json(dict(type='auth', token='watch1'))
        self.assertEqual((await duplicate.receive_json())['state'], 'error')
        await duplicate.close()
        await self.start(ws)
        await ws.send_bytes(struct.pack('<I', 4) + VOICE)
        error = await self.until(ws, 'error')
        self.assertEqual(error['code'], 'sequence_mismatch_restart_recording')
        await ws.close()
        await asyncio.sleep(.01)
        ws, renewed = await self.connect()
        self.assertNotEqual(initial['session_id'], renewed['session_id'])
        await ws.close()

    async def test_short_speech_and_silence_end_without_inference(self):
        ws, _ = await self.connect()
        await self.start(ws)
        for seq in range(2):
            await ws.send_bytes(struct.pack('<I', seq) + VOICE * 5)
        await ws.send_json(dict(type='stop'))
        stopped = None
        for _ in range(3):
            msg = await ws.receive_json()
            if msg.get('phase') == 'stopped':
                stopped = msg
                break
        self.assertIsNotNone(stopped['discarded_utterance_id'])
        self.assertEqual(self.engine.calls, [])
        await ws.close()

    async def test_http_compatibility_and_same_worker_as_websocket(self):
        ws, _ = await self.connect()
        await self.start(ws)
        await self.send_voice(ws)
        await ws.send_json(dict(type='stop'))
        form = FormData()
        form.add_field('file', wav_bytes(bytes([4, 0]) * 6400), filename='audio.wav', content_type='audio/wav')
        response = await self.client.post('/api/asr', data=form,
            headers={'Authorization': 'Bearer local1', 'X-Request-ID': 'http-1'})
        self.assertEqual(response.status, 200)
        result = await response.json()
        self.assertEqual((result['text'], result['source'], result['request_id']), ('result-4', 'local', 'http-1'))
        await self.until(ws, 'final')
        self.assertEqual(self.engine.maximum, 1)
        await ws.close()

    async def test_http_auth_format_and_ip_origin_boundaries(self):
        response = await self.client.post('/api/asr', headers={'Authorization': 'Bearer wrong'})
        self.assertEqual(response.status, 401)
        form = FormData()
        form.add_field('file', b'not-a-wav', filename='x.wav')
        response = await self.client.post('/api/asr', data=form)
        self.assertEqual(response.status, 400)
        response = await self.client.get('/ws/asr', headers={'Origin': 'https://untrusted.invalid'})
        self.assertEqual(response.status, 403)
        metrics = await (await self.client.get('/metrics')).json()
        self.assertNotIn('text', metrics)

    async def test_http_silence_does_not_reach_whisper(self):
        form = FormData()
        form.add_field('file', wav_bytes(QUIET * 50), filename='silence.wav')
        response = await self.client.post('/api/asr', data=form)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['text'], '')
        self.assertEqual(self.engine.calls, [])

    async def test_connection_and_silent_audio_are_observable_without_logging_content(self):
        with self.assertLogs('asr.events', level='INFO') as logs:
            ws, _ = await self.connect()
            await self.start(ws)
            await ws.send_bytes(struct.pack('<I', 0) + QUIET * 5)
            await ws.send_json(dict(type='ping'))
            self.assertEqual((await ws.receive_json())['phase'], 'pong')
            response = await self.client.get('/metrics')
            metrics = await response.json()
            self.assertEqual(metrics['authenticated_connections'], 1)
            self.assertEqual(metrics['received_audio_bytes'], 3200)
            self.assertEqual(metrics['received_audio_packets'], 1)
            client = metrics['active_clients'][0]
            self.assertEqual(client['client_id'], 'watch1')
            self.assertTrue(client['recording'])
            self.assertEqual(client['audio_bytes'], 3200)
            self.assertIsNotNone(client['last_audio_age_seconds'])
            self.assertEqual(self.engine.calls, [])
            await ws.close()
        output = '\n'.join(logs.output)
        self.assertIn('event=authenticated', output)
        self.assertIn('event=audio_received', output)
        self.assertNotIn('token=', output)
        self.assertNotIn('text=', output)

    async def test_documented_ready_control_fields_and_empty_recording_completion(self):
        ws, ready = await self.connect()
        self.assertEqual(set(ready), {'state', 'client_id', 'source', 'session_id',
            'request_id', 'phase', 'protocol', 'format', 'sample_rate', 'channels'})
        self.assertIsNone(ready['request_id'])
        self.assertEqual((ready['protocol'], ready['format'], ready['sample_rate'], ready['channels']),
                         (1, 'pcm_s16le', 16000, 1))
        self.assertEqual(len(ready['session_id']), 32)
        await ws.send_json(dict(type='start', request_id='spec-request',
                                sample_rate=16000, channels=1, format='pcm_s16le'))
        recording = await ws.receive_json()
        self.assertEqual(recording['next_sequence'], 0)
        self.assertEqual(recording['phase'], 'recording')
        await ws.send_json(dict(type='stop'))
        stopped = await ws.receive_json()
        self.assertEqual(stopped['phase'], 'stopped')
        self.assertEqual(len(stopped['discarded_utterance_id']), 32)
        await ws.send_json(dict(type='ping'))
        pong = await ws.receive_json()
        self.assertEqual((pong['phase'], pong['request_id']), ('pong', 'spec-request'))
        self.assertEqual(self.engine.calls, [])
        await ws.close()

    async def test_documented_fatal_validation_codes(self):
        cases = [('{', False, 'invalid_json'),
                 ('[]', False, 'invalid_message'),
                 ('{"type":"start","request_id":"bad id"}', False, 'invalid_request_id'),
                 ('{"type":"start","request_id":"ok","channels":2}', False, 'unsupported_audio_format'),
                 ('{"type":"auth"}', False, 'unknown_command'),
                 (struct.pack('<I', 0) + QUIET, False, 'start_required'),
                 (struct.pack('<I', 0) + QUIET[:-1], True, 'invalid_audio_packet'),
                 (struct.pack('<I', 0) + QUIET * 11, True, 'invalid_audio_packet'),
                 (struct.pack('<I', 1) + QUIET, True, 'sequence_mismatch_restart_recording')]
        for payload, started, expected in cases:
            with self.subTest(code=expected):
                ws, _ = await self.connect()
                if started:
                    await self.start(ws)
                if isinstance(payload, bytes):
                    await ws.send_bytes(payload)
                else:
                    await ws.send_str(payload)
                error = await self.until(ws, 'error')
                self.assertEqual(error['code'], expected)
                self.assertEqual(error['client_id'], 'watch1')
                from aiohttp import WSMsgType
                self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)
                await ws.close()
                await asyncio.sleep(.01)

    async def test_stop_keeps_pending_final_and_rejects_new_start_until_done(self):
        self.engine.gate = asyncio.Event()
        ws, _ = await self.connect()
        await self.start(ws, 'old')
        await self.send_voice(ws)
        await ws.send_json(dict(type='stop'))
        while (await ws.receive_json()).get('phase') != 'stopped':
            pass
        await ws.send_json(dict(type='start', request_id='premature'))
        error = await self.until(ws, 'error')
        self.assertEqual((error['code'], error['request_id']), ('request_in_progress', 'old'))
        self.engine.gate.set()
        final = await self.until(ws, 'final')
        self.assertEqual(final['request_id'], 'old')
        self.assertGreater(final['revision'], 0)
        await ws.send_json(dict(type='stop'))
        self.assertEqual((await self.until(ws, 'error'))['code'], 'not_recording')
        await self.start(ws, 'new')
        await ws.close()

    async def test_documented_min_and_max_packet_sizes(self):
        ws, _ = await self.connect()
        await self.start(ws)
        await ws.send_bytes(struct.pack('<I', 0) + QUIET)
        await ws.send_bytes(struct.pack('<I', 1) + QUIET * 10)
        await ws.send_json(dict(type='ping'))
        self.assertEqual((await ws.receive_json())['phase'], 'pong')
        metrics = await (await self.client.get('/metrics')).json()
        self.assertEqual(metrics['received_audio_bytes'], 640 * 11)
        self.assertEqual(metrics['received_audio_packets'], 2)
        await ws.close()

    async def debug_control(self, action):
        state = await (await self.client.get('/test/api/state')).json()
        response = await self.client.post('/test/api/control', json={'action': action},
            headers={'X-ASR-Debug-Token': state['token']})
        self.assertEqual(response.status, 200)

    async def test_debug_opt_in_and_exact_audio_and_separate_devices(self):
        import io
        import wave
        a, _ = await self.connect()
        await self.start(a)
        await self.send_voice(a)
        await a.send_json({'type': 'stop'})
        await self.until(a, 'final')
        state = await (await self.client.get('/test/api/state')).json()
        self.assertFalse(state['enabled'])
        self.assertEqual(state['records'], [])
        await self.debug_control('start')
        b, _ = await self.connect('local1')
        await self.start(a, 'same-request')
        await self.start(b, 'same-request')
        await self.send_voice(a, 1)
        await self.send_voice(b, 2)
        await a.send_json({'type': 'stop'})
        await b.send_json({'type': 'stop'})
        await asyncio.gather(self.until(a, 'final'), self.until(b, 'final'))
        state = await (await self.client.get('/test/api/state')).json()
        self.assertEqual(len(state['records']), 2)
        for record in state['records']:
            marker = 1 if record['client_id'] == 'watch1' else 2
            prefix = '/test/api/records/' + record['id']
            raw = await (await self.client.get(prefix + '/audio.pcm')).read()
            self.assertEqual(raw, bytes([marker, 0]) * 320 * 20)
            wav = await (await self.client.get(prefix + '/audio.wav')).read()
            with wave.open(io.BytesIO(wav)) as audio:
                self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()), (16000, 1, 2))
                self.assertEqual(audio.readframes(audio.getnframes()), raw)
            detail = await (await self.client.get(prefix)).json()
            self.assertEqual(detail['results'][0]['text'], 'result-' + str(marker))
        await self.debug_control('clear')
        self.assertEqual((await self.client.get(prefix)).status, 404)
        await a.close()
        await b.close()

    async def test_debug_security_and_fixed_assets(self):
        self.assertEqual((await self.client.get('/test/')).status, 200)
        self.assertEqual((await self.client.get('/test/app.js')).status, 200)
        self.assertEqual((await self.client.get('/test/README.md')).status, 404)
        self.assertEqual((await self.client.get('/test/api/state', headers={'Host': 'outside.example'})).status, 403)
        self.assertEqual((await self.client.get('/test/api/state', headers={'Origin': 'http://evil.example'})).status, 403)
        self.assertEqual((await self.client.post('/test/api/control', json={'action': 'start'})).status, 403)
        self.assertFalse((await (await self.client.get('/test/api/state')).json())['enabled'])

    async def test_debug_clear_during_inference_does_not_restore_text(self):
        self.engine.gate = asyncio.Event()
        await self.debug_control('start')
        ws, _ = await self.connect()
        await self.start(ws)
        await self.send_voice(ws)
        await ws.send_json({'type': 'stop'})
        await self.debug_control('clear')
        self.engine.gate.set()
        await self.until(ws, 'final')
        state = await (await self.client.get('/test/api/state')).json()
        self.assertEqual(state['records'], [])
        self.assertNotIn('result-1', json.dumps(state))
        await ws.close()

    async def test_debug_http_upload_and_pause(self):
        await self.debug_control('start')
        data = FormData()
        data.add_field('file', wav_bytes(VOICE * 20), filename='test.wav', content_type='audio/wav')
        response = await self.client.post('/api/asr', data=data, headers={'X-Request-ID': 'http-debug'})
        self.assertEqual(response.status, 200)
        state = await (await self.client.get('/test/api/state')).json()
        self.assertEqual(state['records'][0]['transport'], 'http')
        self.assertEqual(state['records'][0]['request_id'], 'http-debug')
        await self.debug_control('pause')
        self.assertFalse((await (await self.client.get('/test/api/state')).json())['enabled'])


class DebugStoreTests(unittest.TestCase):
    def test_memory_count_ttl_and_capture_expiration(self):
        from app.brain.asr.debug import DebugStore
        clock = [100.]
        store = DebugStore(clock=lambda: clock[0])
        client = {'id': 'watch', 'source': 'watch'}
        self.assertIsNone(store.begin(client, 's', 'r'))
        store.enabled_until = 700
        store.MAX_BYTES = 1000
        for _ in range(30):
            key = store.begin(client, 's', 'r')
            store.audio(key, bytes(100))
        self.assertLessEqual(sum(len(r['pcm']) for r in store.records.values()), 1000)
        self.assertLessEqual(len(store.records), store.MAX_RECORDS)
        for _ in range(400):
            store.event('connected')
        self.assertEqual(len(store.events), 300)
        clock[0] = 701
        store.purge()
        self.assertFalse(store.enabled)
        self.assertEqual(store.records, {})


if __name__ == '__main__':
    unittest.main()
