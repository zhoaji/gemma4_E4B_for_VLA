"""Offline per-client pairing. Store only token digests on the server."""
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import secrets

DEFAULT_KEYS = Path(__file__).resolve().parents[3] / 'data/asr/clients.json'


class Credentials:
    def __init__(self, path=DEFAULT_KEYS):
        self.path = Path(path)

    def identify(self, token):
        if not isinstance(token, str) or len(token) > 256:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        try:
            clients = json.loads(self.path.read_text())['clients']
        except FileNotFoundError:
            return None
        for client in clients:
            if not client.get('disabled', False) and hmac.compare_digest(client['sha256'], digest):
                return client
        return None


def allowed_ip(address, networks):
    try:
        ip = ipaddress.ip_address(address)
        return any(ip in ipaddress.ip_network(n) for n in networks)
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Local pairing administration; never exposed over HTTP')
    parser.add_argument('--file', type=Path, default=DEFAULT_KEYS)
    parser.add_argument('--id', required=True)
    parser.add_argument('--source', choices=['watch', 'local', 'background'], default='watch')
    parser.add_argument('--revoke', action='store_true')
    args = parser.parse_args()
    if not args.id or len(args.id) > 64 or not all(c.isalnum() or c in '-_' for c in args.id):
        parser.error('id must contain 1–64 letters/digits/_/-')
    args.file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement, exclusive lock prevents concurrent pairing losing records.
    import fcntl
    with open(str(args.file) + '.lock', 'a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(args.file.read_text()) if args.file.exists() else {'clients': []}
        data['clients'] = [c for c in data['clients'] if c['id'] != args.id]
        token = secrets.token_urlsafe(32)
        if not args.revoke:
            data['clients'].append(dict(id=args.id, source=args.source,
                sha256=hashlib.sha256(token.encode()).hexdigest()))
        temporary = args.file.with_suffix('.tmp')
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(data, output, indent=2)
        os.replace(temporary, args.file)
    print('revoked' if args.revoke else f'client_id={args.id}\ntoken={token}\nSave this token securely; it is shown only now.')


if __name__ == '__main__':
    main()
