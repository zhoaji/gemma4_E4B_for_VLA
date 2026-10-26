"""Persistent whisper.cpp backend. Audio and transcripts never go to disk/logs."""
import asyncio
import io
import json
import shutil
from pathlib import Path
import subprocess
import wave

import aiohttp

ROOT = Path(__file__).resolve().parents[3]
CUDA = ROOT / 'third_party/whisper.cpp/build/bin/whisper-server'
CPU = ROOT / 'third_party/whisper.cpp/build-cpu/bin/whisper-server'
MODEL = ROOT / 'data/asr/whispercpp/ggml-base.bin'


def wav_bytes(pcm):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return output.getvalue()


class WhisperEngine:
    @property
    def ready(self):
        return self.process is not None and self.process.returncode is None

    def __init__(self, backend='cuda', binary=None, model=MODEL, port=8002, language='zh'):
        self.backend = backend
        self.binary = Path(binary or (CUDA if backend == 'cuda' else CPU))
        self.model, self.port, self.language = Path(model), port, language
        self.process = None
        self.client = None

    async def start(self):
        if not self.binary.is_file() or not self.model.is_file():
            raise RuntimeError('Whisper executable/model missing; CUDA requires a completed CUDA build')
        # Never attach to an unrelated process that happens to own this port.
        import socket
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.port))
        command = [str(self.binary), '--host', '127.0.0.1', '--port', str(self.port),
                   '-m', str(self.model), '-t', '4', '-sns', '-nt', '--max-context', '0',
                   '--no-flash-attn']
        if self.backend == 'cpu':
            command.append('--no-gpu')
        command += ['-l', self.language]
        # whisper.cpp prints recognized speech even without our request logger.
        # Capture startup lines only to verify CUDA initialization; drain and discard afterwards.
        self.process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.gpu_seen = False
        self.log_task = asyncio.create_task(self._drain())
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90))
        try:
            for _ in range(600):
                if self.process.returncode is not None:
                    raise RuntimeError('Whisper startup failed (check binary/model/backend)')
                try:
                    async with self.client.get(f'http://127.0.0.1:{self.port}/health') as r:
                        ready = r.status == 200
                    if ready:
                        if self.backend == 'cuda' and not self.gpu_seen:
                            raise RuntimeError('CUDA GPU initialization was not confirmed; no CPU fallback')
                        return
                except aiohttp.ClientConnectionError:
                    pass
                await asyncio.sleep(.2)
            raise RuntimeError('Whisper startup timeout')
        except BaseException:
            await self.close()
            raise

    async def _drain(self):
        while line := await self.process.stderr.readline():
            # Require Whisper's selected CUDA backend, not just enumerated devices.
            if b'whisper_backend_init_gpu: using CUDA' in line:
                self.gpu_seen = True
            if b'failed to initialize CUDA' in line or b'no GPU found' in line:
                self.gpu_seen = False

    async def transcribe(self, pcm):
        form = aiohttp.FormData()
        form.add_field('file', wav_bytes(pcm), filename='audio.wav', content_type='audio/wav')
        form.add_field('response_format', 'json')
        form.add_field('temperature', '0')
        async with self.client.post(f'http://127.0.0.1:{self.port}/inference', data=form) as r:
            if r.status != 200:
                raise RuntimeError('Whisper inference failed')
            result = json.loads(await r.read())
            text = str(result.get('text', '')).strip()
            if text and shutil.which('opencc'):
                def simplify():
                    try:
                        return subprocess.run(['opencc', '-c', 'tw2sp.json'], input=text,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=3, check=True).stdout.strip() or text
                    except (OSError, subprocess.SubprocessError):
                        return text
                return await asyncio.to_thread(simplify)
            return text

    async def close(self):
        if self.client:
            await self.client.close()
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if hasattr(self, 'log_task'):
            await self.log_task
