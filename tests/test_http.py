import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from model_server.slim_v1 import messages as build_messages, envelope
from model_server.server import make_server


class FakeRuntime:
    identity = {'checkpoint': 'fake'}
    def predict(self, title, body):
        return envelope('{"events":[]}', 'stop', {'title': title, 'body': body})


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = make_server(FakeRuntime(), 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, data):
        req = Request(self.url + '/extract', data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=3) as response: return json.load(response)

    def test_health_and_request(self):
        with urlopen(self.url + '/health') as response:
            self.assertEqual(json.load(response)['status'], 'ready')
        value = self.post({'title': '제목', 'body': '본문', 'user_content': 'ignored old template'})
        self.assertEqual(value['metadata']['body'], '본문')
        self.assertTrue(value['review_required'])

    def test_invalid_input(self):
        for value in ({}, {'title': '', 'body': ''}, {'title': [], 'body': 'x'}):
            with self.assertRaises(HTTPError) as caught: self.post(value)
            self.assertEqual(caught.exception.code, 422)

    def test_incomplete_generation_not_accepted(self):
        self.assertEqual(envelope('{"events":[]}', 'length', {})['status'], 'invalid')
        self.assertEqual(envelope('{"events":[]}', 'timeout', {})['status'], 'invalid')
        self.assertEqual(envelope('```json\n{}\n```', 'stop', {})['status'], 'invalid')

    def test_current_input_not_augmented_with_unseen_sources(self):
        message = build_messages('제목', '실제 본문')[-1]['content']
        self.assertIn('실제 본문', message)
        self.assertNotIn('| attachment', message)
        self.assertIn('"reference_time": null', message)

    def test_busy_rejected_without_second_generation(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking(FakeRuntime):
            def predict(self, title, body):
                entered.set()
                release.wait(3)
                return super().predict(title, body)
        # Replace the server used by this test with a blocking model.
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.server = make_server(Blocking(), 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        first = threading.Thread(target=lambda: self.post({'title': 'first', 'body': 'body'}))
        first.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(HTTPError) as caught: self.post({'title': 'second', 'body': 'body'})
            self.assertEqual(caught.exception.code, 503)
        finally:
            release.set(); first.join()

    def test_network_bind_requires_secret(self):
        with self.assertRaises(ValueError): make_server(FakeRuntime(), 0, host='0.0.0.0')

    def test_shared_token_protects_health_and_inference(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        token = 'test-only-token-' + 'x' * 32
        self.server = make_server(FakeRuntime(), 0, token=token)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        for path, data in [('/health', None), ('/extract', b'{"title":"x","body":"y"}')]:
            for header in ({}, {'X-Model-Token':'wrong'}):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(Request(self.url + path, data=data, headers=header), timeout=3)
                self.assertEqual(caught.exception.code, 401)
            with urlopen(Request(self.url + path, data=data, headers={'X-Model-Token':token}), timeout=3) as response:
                self.assertEqual(response.status, 200)


if __name__ == '__main__': unittest.main()
