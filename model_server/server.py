"""Loopback-only model API for local integration. One GPU generation at a time."""
import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer




def make_server(runtime, port=8765, host='127.0.0.1', token=None):
    if host not in ('127.0.0.1', 'localhost', '::1') and not token:
        raise ValueError('Non-loopback binding requires MODEL_API_TOKEN')
    if token and len(token) < 32:
        raise ValueError('MODEL_API_TOKEN must contain at least 32 characters')
    lock = threading.Lock()
    accepts_reference_time = getattr(runtime, 'prompt_mode', None) == 'slim-v9'

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            if not self.authorized(): return
            if self.path != '/health': return self.reply(404, {'error': 'not_found'})
            self.reply(200, {'status': 'ready', 'busy': lock.locked(), 'token_required': bool(token), 'model': runtime.identity})

        def authorized(self):
            values = self.headers.get_all('X-Model-Token') or []
            if token and (len(values) != 1 or not hmac.compare_digest(values[0].encode('utf-8'), token.encode('utf-8'))):
                self.reply(401, {'error': 'unauthorized'})
                return False
            return True

        def do_POST(self):
            if not self.authorized(): return
            if self.path != '/extract': return self.reply(404, {'error': 'not_found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 400000: return self.reply(413, {'error': 'request_size'})
                data = json.loads(self.rfile.read(size))
                allowed = {'title', 'body', 'user_content'} | ({'reference_time'} if accepts_reference_time else set())
                if not isinstance(data, dict) or set(data) - allowed:
                    raise ValueError('expected title/body')
                if any(not isinstance(data.get(key), str) for key in ('title', 'body')):
                    raise ValueError('title/body must be strings')
                if not data['title'].strip() and not data['body'].strip(): raise ValueError('empty_input')
                if len(data['title']) > 4000 or len(data['body']) > 80000: return self.reply(413, {'error': 'input_size'})
            except (ValueError, UnicodeError):
                return self.reply(422, {'error': 'invalid_request'})
            if not lock.acquire(blocking=False): return self.reply(503, {'error': 'model_busy'})
            try:
                if accepts_reference_time:
                    result = runtime.predict(data['title'], data['body'], reference_time=data.get('reference_time'))
                else:
                    result = runtime.predict(data['title'], data['body'])
                self.reply(200, result)
            except ValueError as exc:
                self.reply(422, {'error': str(exc)})
            except Exception as exc:
                print(f'Inference failed: {type(exc).__name__}', flush=True)
                self.reply(500, {'error': 'inference_failed'})
            finally: lock.release()

    return ThreadingHTTPServer((host, port), Handler)
