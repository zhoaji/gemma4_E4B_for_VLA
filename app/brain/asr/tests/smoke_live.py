#!/usr/bin/env python3
"""Opt-in live test: two real Whisper clients, one WSS and one WS.
Uses loopback ports 18001/18002/18444, ephemeral certificates and credentials.
TEST_ASR_BACKEND=cpu|cuda python3 app/brain/asr/tests/smoke_live.py
"""
import asyncio, hashlib, json, os, secrets, signal, sys, tempfile, time
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.brain.asr.tls import generate
async def main():
    with tempfile.TemporaryDirectory(prefix='asr-smoke-') as directory:
        directory=Path(directory)
        tokens={name:secrets.token_urlsafe(32) for name in ('watch','local')}
        keys=directory/'clients.json'
        keys.write_text(json.dumps({'clients':[dict(id=k+'-test',source=k,sha256=hashlib.sha256(v.encode()).hexdigest()) for k,v in tokens.items()]}))
        cert, key, ca = generate('127.0.0.1', directory)
        log=open(directory/'server.log','wb')
        backend=os.getenv('TEST_ASR_BACKEND','cpu')
        proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'app/brain/asr/server.py'),'--backend',backend,'--language','auto','--port','18001','--worker-port','18002','--keys',str(keys),'--partial-ms','2000','--lan-bind','127.0.0.1','--lan-port','18444','--tls-cert',str(cert),'--tls-key',str(key),stdout=log,stderr=log)
        start=time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(600):
                    if proc.returncode is not None: raise RuntimeError('server exited: '+(directory/'server.log').read_text())
                    try:
                        async with session.get('http://127.0.0.1:18001/health') as r:
                            health=await r.json()
                            if health['status']=='ready':break
                    except aiohttp.ClientError:pass
                    await asyncio.sleep(.2)
                else:raise RuntimeError('model load timeout')
                async def run_client(source):
                    env=dict(os.environ,ASR_TOKEN=tokens[source])
                    url='wss://127.0.0.1:18444/ws/asr' if source=='watch' else 'ws://127.0.0.1:18001/ws/asr'
                    client=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'app/brain/asr/client.py'),str(ROOT/'third_party/whisper.cpp/samples/jfk.wav'),'--url',url,'--ca',str(ca),env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                    output,error=await asyncio.wait_for(client.communicate(),150)
                    if client.returncode:raise RuntimeError(source+' client failed: '+error.decode())
                    records=[json.loads(line) for line in output.splitlines()]
                    finals=[r for r in records if r['state']=='final']
                    assert finals and all(r['text'] for r in finals),source+' no final'
                    assert all(r['source']==source and r['client_id']==source+'-test' for r in records)
                    return dict(source=source,finals=len(finals),partials=sum(r['state']=='partial' for r in records),session_id=finals[0]['session_id'],final_elapsed_ms=[r['elapsed_ms'] for r in finals])
                clients=await asyncio.gather(run_client('watch'),run_client('local'))
                assert clients[0]['session_id']!=clients[1]['session_id']
                async with session.get('http://127.0.0.1:18001/metrics') as r:metrics=await r.json()
                report=dict(backend=backend,health=health,clients=clients,metrics=metrics,total_seconds=round(time.monotonic()-start,2))
                (ROOT/f'data/asr/streaming-verification-{backend}.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(report,indent=2))
        finally:
            if proc.returncode is None:
                proc.send_signal(signal.SIGTERM)
                try:await asyncio.wait_for(proc.wait(),15)
                except asyncio.TimeoutError:proc.kill();await proc.wait()
            log.close()
            logs=(directory/'server.log').read_text()
            assert not any(t in logs for t in tokens.values()),'credential leaked'
            assert 'fellow Americans' not in logs,'transcript leaked'
asyncio.run(main())
