#!/usr/bin/env python3
"""Create a private local CA + server certificate. Never overwrite trust material."""
import argparse
import ipaddress
import os
from pathlib import Path
import re
import shutil
import subprocess


def generate(host, directory):
    try:
        ipaddress.ip_address(host)
        san = 'IP:' + host
    except ValueError:
        if len(host) > 253 or not all(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
                                      for label in host.split('.')):
            raise ValueError('host must be an IP address or DNS name')
        san = 'DNS:' + host
    if not shutil.which('openssl'):
        raise RuntimeError('openssl is required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    names = ['ca.key', 'ca.crt', 'ca.srl', 'server.key', 'server.crt', 'server.csr', 'server.ext']
    if any((directory / name).exists() for name in names):
        raise FileExistsError('certificate files already exist; use another directory to avoid changing device trust')

    def run(*args):
        subprocess.run(['openssl', *args], cwd=directory, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)

    previous = os.umask(0o077)
    try:
        run('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '3650',
            '-keyout', 'ca.key', '-out', 'ca.crt', '-subj', '/CN=ASR Local CA',
            '-addext', 'basicConstraints=critical,CA:TRUE,pathlen:0',
            '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
        run('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'server.key',
            '-out', 'server.csr', '-subj', '/CN=' + host)
        (directory / 'server.ext').write_text('basicConstraints=critical,CA:FALSE\n'
            'keyUsage=critical,digitalSignature,keyEncipherment\n'
            'extendedKeyUsage=serverAuth\nsubjectAltName=' + san + '\n')
        run('x509', '-req', '-in', 'server.csr', '-CA', 'ca.crt', '-CAkey', 'ca.key',
            '-CAcreateserial', '-out', 'server.crt', '-days', '365', '-sha256',
            '-extfile', 'server.ext')
        (directory / 'server.csr').unlink()
        (directory / 'server.ext').unlink()
    finally:
        os.umask(previous)
    return directory / 'server.crt', directory / 'server.key', directory / 'ca.crt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, help='Jetson LAN IP or DNS name in certificate SAN')
    parser.add_argument('--directory', type=Path,
        default=Path(__file__).resolve().parents[3] / 'data/asr/tls')
    args = parser.parse_args()
    cert, key, ca = generate(args.host, args.directory)
    print(f'Server certificate: {cert}\nServer private key: {key}\nDevice trust CA: {ca}')
