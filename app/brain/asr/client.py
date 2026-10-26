#!/usr/bin/env python3
"""WAV watch simulator / local streaming client. Prints results only on the client."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import ssl
import struct
import sys
import uuid
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data/asr/python-deps'))
import aiohttp


async def stream(args):
    token = os.environ.get('ASR_TOKEN')
    if not token:
        raise RuntimeError('Set ASR_TOKEN; never put credentials in URLs or command arguments')
    with wave.open(str(args.wav)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError('expected 16 kHz mono PCM16 WAV')
        if not 0 < wav.getnframes() <= 60 * 16000:
            raise ValueError('recording must be 0–60 seconds')
        pcm = wav.readframes(wav.getnframes())
    context = ssl.create_default_context(cafile=args.ca) if args.ca else None
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, ssl=context, heartbeat=15) as ws:
            await ws.send_json(dict(type='auth', token=token))
            if (await ws.receive_json()).get('phase') != 'ready':
                raise RuntimeError('authentication failed')
            await ws.send_json(dict(type='start', request_id=uuid.uuid4().hex,
                                   format='pcm_s16le', channels=1, sample_rate=16000))
            if (await ws.receive_json()).get('phase') != 'recording':
                raise RuntimeError('recording rejected')
            stopped = asyncio.Event()
            pending = set()
            stop_ack = False

            async def receive():
                nonlocal stop_ack
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = json.loads(message.data)
                    print(json.dumps(data, ensure_ascii=False), flush=True)
                    if data.get('phase') == 'speech':
                        pending.add(data['utterance_id'])
                    if data.get('state') == 'final':
                        pending.discard(data['utterance_id'])
                    if data.get('phase') == 'no_speech':
                        pending.discard(data['utterance_id'])
                    if data.get('state') == 'error':
                        raise RuntimeError(data.get('code', 'ASR error'))
                    if data.get('phase') == 'stopped':
                        stop_ack = True
                        pending.discard(data.get('discarded_utterance_id'))
                    if stop_ack and not pending:
                        stopped.set()

            receiver = asyncio.create_task(receive())
            waiter = asyncio.create_task(stopped.wait())
            try:
                for seq, offset in enumerate(range(0, len(pcm), 3200)):
                    chunk = pcm[offset:offset + 3200]
                    chunk += b'\0' * ((-len(chunk)) % 640)
                    await ws.send_bytes(struct.pack('<I', seq) + chunk)
                    await asyncio.sleep(len(chunk) / 32000)
                await ws.send_json(dict(type='stop'))
                done, _ = await asyncio.wait([receiver, waiter], timeout=125,
                                             return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    await receiver
                elif not done:
                    raise TimeoutError('no final result')
            finally:
                waiter.cancel()
                receiver.cancel()
                await asyncio.gather(waiter, receiver, return_exceptions=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', type=Path)
    parser.add_argument('--url', default='ws://127.0.0.1:8000/ws/asr')
    parser.add_argument('--ca', help='Trusted CA PEM for WSS; never disable validation')
    asyncio.run(stream(parser.parse_args()))
