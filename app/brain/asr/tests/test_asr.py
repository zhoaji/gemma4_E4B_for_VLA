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


if __name__ == '__main__':
    unittest.main()
