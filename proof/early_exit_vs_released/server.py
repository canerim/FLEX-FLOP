"""Local live benchmark dashboard. Fixed workloads, one run at a time."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = threading.Lock()
STATE = {'state': 'idle', 'events': [], 'last_result': None}


def run_benchmark(gpu, sequence, qp, blocks):
    output = HERE / 'results' / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    command = [sys.executable, str(HERE/'benchmark.py'), '--gpu', str(gpu), '--sequence', sequence,
               '--qp', str(qp), '--blocks', str(blocks), '--out', str(output)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with LOCK:
        STATE['pid'] = process.pid
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {'type': 'log', 'text': line.strip()[:500]}
        with LOCK:
            STATE['events'].append(event)
            STATE['events'] = STATE['events'][-120:]
    rc = process.wait()
    with LOCK:
        STATE['state'] = 'complete' if rc == 0 else 'blocked' if rc == 3 else 'failed'
        STATE['returncode'] = rc
        if rc == 0:
            STATE['last_result'] = str(output)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, code=200):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status':
            with LOCK:
                payload = dict(STATE)
            self.send_json(payload)
        elif self.path in ('/', '/index.html'):
            data = (HERE/'ui/index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != '/api/run':
            return self.send_error(404)
        if self.headers.get('X-Proof-Token') != self.server.proof_token:
            return self.send_json({'error': 'Invalid run token'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length < 256:
                raise ValueError('Invalid request length')
            body = json.loads(self.rfile.read(length))
            gpu, sequence, qp, blocks = body['gpu'], body['sequence'], body['qp'], body['blocks']
            if type(gpu) is not int or gpu not in range(8):
                raise ValueError('GPU must be 0..7')
            if type(qp) is not int or qp not in (0, 16, 32, 48, 63):
                raise ValueError('Unsupported QP')
            if sequence not in ('videoSRC05', 'FourPeople', 'BQMall'):
                raise ValueError('Unsupported sequence')
            if type(blocks) is not int or not 5 <= blocks <= 100:
                raise ValueError('Blocks must be 5..100')
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.send_json({'error': str(exc)}, 400)
        with LOCK:
            if STATE['state'] == 'running':
                return self.send_json({'error': 'Another run is active'}, 409)
            STATE.update(state='running', events=[], gpu=gpu, sequence=sequence, qp=qp, blocks=blocks, returncode=None)
            thread = threading.Thread(target=run_benchmark, args=(gpu, sequence, qp, blocks), daemon=True)
            thread.start()
        self.send_json({'started': True}, 202)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()
    token = os.environ.get('PROOF_RUN_TOKEN') or secrets.token_urlsafe(24)
    if a.host not in ('127.0.0.1', 'localhost', '::1') and not os.environ.get('PROOF_RUN_TOKEN'):
        p.error('Set PROOF_RUN_TOKEN before binding a non-loopback address')
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.proof_token = token
    print(f'URL: http://{a.host}:{a.port}/\nRun token: {token}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
