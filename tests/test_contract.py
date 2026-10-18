"""Slim-v9 CPU contract tests; no model initialization or evaluation datasets."""
import hashlib
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from model_server.runtime import ROOT, Runtime
from model_server.server import make_server
from model_server.slim_server import build_parser
from model_server.slim_v1 import SlimRuntime, messages, system_prompt, normalize_reference_time, envelope, apply_interview_type
from tests.test_postprocess import event

V9_HASH = '6c5941436aa1b069907fa696b988eb2e57fe03e5ed854b61f99cba545ea7f096'
V9_PROMPT_HASH = 'e0bf6f031bd78ac0505b29ce552ed8b42bb7b200add8bd04d051cf6493341e81'

def fake_runtime(mode='slim-v9', interview_type='passthrough'):
    runtime = SlimRuntime.__new__(SlimRuntime)
    runtime.prompt_mode = mode
    runtime.interview_type = interview_type
    runtime.identity = {'prompt_mode': mode}
    def predict_messages(chat):
        runtime.last_chat = chat
        raw = json.dumps({'events': [event('면접', start='2026-10-07', event_type='interview')]})
        return apply_interview_type(envelope(raw, 'stop', {}), interview_type)
    runtime.predict_messages = predict_messages
    return runtime

class ContractTests(unittest.TestCase):
    def test_prompt_bytes_match_evaluation_freeze(self):
        path = ROOT / 'model_server/prompts/system_prompt.txt'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), V9_HASH)
        # Evaluation read_text normalizes CRLF to LF; pin both artifact and effective prompt.
        self.assertEqual(hashlib.sha256(system_prompt('slim-v9').encode()).hexdigest(), V9_PROMPT_HASH)
        self.assertEqual(messages('t', 'b', 'slim-v9')[0]['content'], system_prompt('slim-v9'))

    def test_iso_publication_values(self):
        for value in ('2026-10-07', '2028-02-29', '2026-10-07T19:30',
                      '2026-10-07T19:30:12.123+09:00', '2026-10-07T10:30:00Z'):
            self.assertEqual(normalize_reference_time(value), (value, None))
        self.assertEqual(normalize_reference_time(None), (None, None))

    def test_invalid_reference_is_null_with_warning(self):
        for value in ('', 'yesterday', '2026-02-29', '2026-10-07T24:00', '20261007',
                      20261007, [], {}, True, '2026-10-07T10:00+99:00', '2026-10-07T10:00+09:99'):
            self.assertEqual(normalize_reference_time(value), (None, 'invalid_reference_time'))
            runtime = fake_runtime()
            result = runtime.predict('t', 'b', reference_time=value)
            self.assertIsNone(result['metadata']['reference_time'])
            self.assertEqual(result['metadata']['reference_time_warning'], 'invalid_reference_time')
            self.assertIn('"reference_time": null', runtime.last_chat[1]['content'])

    def test_reference_metadata_position_and_default(self):
        for value in (None, '2026-10-07', '2026-10-07T10:00:00Z'):
            runtime = fake_runtime()
            result = runtime.predict('t', 'b', reference_time=value)
            content = runtime.last_chat[1]['content']
            metadata = json.loads(content.split('[입력 메타데이터]\n')[1].split('\n\n')[0])
            self.assertEqual(metadata, {'reference_time': value, 'default_timezone': 'Asia/Seoul', 'input_issues': []})
            self.assertEqual(result['metadata']['reference_time'], value)
            self.assertNotIn('reference_time_warning', result['metadata'])

    def test_formal_mode_identity_and_cli(self):
        args = build_parser().parse_args(['--prompt-mode', 'slim-v9', '--model-dir', 'x', '--interview-type', 'passthrough'])
        self.assertEqual(args.interview_type, 'passthrough')
        def init(runtime, *args, **kwargs): runtime.identity = {}
        with patch.object(Runtime, '__init__', init):
            runtime = SlimRuntime('x', prompt_mode='slim-v9', interview_type='passthrough')
        self.assertEqual(runtime.identity['prompt_sha256'], V9_PROMPT_HASH)
        self.assertEqual(runtime.identity['prompt_version'], 'slim-v9-policies11-15-20261007')

    def test_passthrough_and_mapping(self):
        for policy, expected in [('passthrough', 'interview'), ('map-to-event', 'event')]:
            response = fake_runtime(interview_type=policy).predict('t', 'b')
            self.assertEqual(response['candidate']['events'][0]['event_type'], expected)
            self.assertEqual(json.loads(response['raw_output'])['events'][0]['event_type'], 'interview')

    def test_v9_repair_and_undated_rules(self):
        raw = json.dumps({'events': [event('undated'),
            event('physical online', start='2026-10-07', location='서울시청 3층', attendance_mode='online'),
            event('invalid date', start='2026-02-29', end='2026-10-07', start_time='10:00'),
            event('reverse', start='2026-10-08', end='2026-10-07')]})
        result = envelope(raw, 'stop', {'prompt_mode': 'slim-v9'})
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['raw_output'], raw)
        items = result['candidate']['events']
        self.assertEqual(len(items), 3)
        self.assertEqual((items[0]['attendance_mode'], items[0]['location']), ('unknown', '서울시청 3층'))
        self.assertIsNone(items[1]['start_date']); self.assertIsNone(items[1]['start_time'])
        self.assertIsNone(items[2]['end_date'])
        self.assertEqual(result['metadata']['excluded_undated_events_count'], 1)
        self.assertEqual(result['metadata']['postprocess_review_count'], 3)


    def test_http_reference_and_legacy_contract(self):
        # The shared handler branches solely on v9; native/compact remain two-argument APIs.
        for mode in ('slim-v9',):
            runtime = fake_runtime(mode)
            server = make_server(runtime, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for reference in ('2026-10-07', 'invalid', [], None):
                    data = {'title': 't', 'body': 'b', 'user_content': 'ignored', 'reference_time': reference}
                    request = Request(f'http://127.0.0.1:{server.server_port}/extract',
                        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
                    if mode == 'slim-v9':
                        with urlopen(request, timeout=3) as stream:
                            self.assertEqual(stream.status, 200)
                            result = json.load(stream)
                        self.assertEqual(result['candidate']['events'][0]['event_type'], 'interview')
                        self.assertNotIn('ignored', runtime.last_chat[1]['content'])
                        if reference not in ('2026-10-07', None):
                            self.assertEqual(result['metadata']['reference_time_warning'], 'invalid_reference_time')
                    else:
                        with self.assertRaises(HTTPError) as caught: urlopen(request, timeout=3)
                        self.assertEqual(caught.exception.code, 422)
            finally:
                server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
