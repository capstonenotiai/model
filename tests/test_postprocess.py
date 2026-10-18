"""CPU-only slim transport postprocessing tests; no model initialization."""
import json
import unittest

from model_server.slim_v1 import envelope


def event(title, start=None, end=None, **extra):
    return {'event_type': 'event', 'title': title, 'start_date': start, 'end_date': end,
        'start_time': None, 'end_time': None, 'location': None, 'attendance_mode': 'unknown', **extra}


class SlimUndatedTests(unittest.TestCase):
    def test_mixed_events_keep_one_sided_dates_and_order(self):
        items = [event('undated first'), event('start only', start='2026-10-06'),
            event('end only', end='2026-10-07'), event('undated last'),
            event('both dates', start='2026-10-06', end='2026-10-07')]
        raw = json.dumps({'events': items}, indent=2)
        metadata = {'prompt_mode': 'slim-v1', 'output_tokens': 123}
        result = envelope(raw, 'stop', metadata)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual([e['title'] for e in result['candidate']['events']],
                         ['start only', 'end only', 'both dates'])
        self.assertEqual(result['metadata']['excluded_undated_events_count'], 2)
        self.assertEqual(result['metadata']['output_tokens'], 123)
        self.assertEqual(result['raw_output'], raw)
        self.assertEqual(metadata, {'prompt_mode': 'slim-v1', 'output_tokens': 123})

    def test_all_undated_including_times_becomes_empty(self):
        raw = json.dumps({'events': [event('no dates', start_time='10:00'), event('also no dates')]})
        result = envelope(raw, 'stop', {})
        self.assertEqual(result['candidate'], {'events': []})
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['metadata']['excluded_undated_events_count'], 2)
        self.assertEqual(result['raw_output'], raw)
        self.assertIn('transport_omitted_orphan_times', result['metadata'])

    def test_empty_events_records_zero(self):
        result = envelope('{"events":[]}', 'stop', {})
        self.assertEqual(result['candidate'], {'events': []})
        self.assertEqual(result['metadata']['excluded_undated_events_count'], 0)

    def test_invalid_json_stays_invalid_and_raw_is_preserved(self):
        raw = '{"events":['
        result = envelope(raw, 'stop', {})
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['raw_output'], raw)
        self.assertEqual(result['metadata']['excluded_undated_events_count'], 0)

    def test_filter_does_not_accept_incomplete_generation(self):
        for finish in ('timeout', 'length'):
            with self.subTest(finish=finish):
                result = envelope(json.dumps({'events': [event('undated')]}), finish, {})
                self.assertEqual(result['candidate'], {'events': []})
                self.assertEqual(result['metadata']['excluded_undated_events_count'], 1)
                self.assertEqual(result['status'], 'invalid')
                self.assertIn('generation_' + finish, result['validation_errors'])

    def test_only_null_dates_are_filtered_and_validation_is_preserved(self):
        for item in (event('empty string', start=''), event('')):
            with self.subTest(item=item):
                result = envelope(json.dumps({'events': [item]}), 'stop', {})
                self.assertEqual(result['metadata']['raw_status'], 'invalid')
                self.assertTrue(result['metadata']['raw_validation_errors'])
                if item['start_date'] == '':
                    self.assertEqual(result['status'], 'candidate')
                    self.assertEqual(result['candidate'], {'events': []})
                    self.assertEqual(result['metadata']['excluded_undated_events_count'], 1)
                else:
                    self.assertEqual(result['candidate'], {'events': []})
                    self.assertEqual(result['metadata']['excluded_undated_events_count'], 1)


class SlimN8Tests(unittest.TestCase):
    def wrap(self, items, finish='stop', metadata=None):
        raw = json.dumps({'events': items}, ensure_ascii=False, indent=2)
        result = envelope(raw, finish, metadata or {})
        self.assertEqual(result['raw_output'], raw)
        return result

    def test_definite_online_media_and_placeholders_are_cleared(self):
        for location in ('온라인 Zoom', 'Zoom', '유튜브 라이브', '구글 미트',
                         'Microsoft Teams', 'https://zoom.us/j/123', '' + '@'.join(['a', 'example.com']) + '',
                         '온라인', '비대면', '추후 안내', '홈페이지'):
            with self.subTest(location=location):
                result = self.wrap([event('x', start='2026-10-06', location=location, attendance_mode='online')])
                self.assertEqual(result['status'], 'candidate')
                self.assertIsNone(result['candidate']['events'][0]['location'])
                self.assertEqual(result['metadata']['online_location_null_count'], 1)
                self.assertEqual(result['metadata']['online_location_null_changes'][0]['before'], location)

    def test_physical_and_ambiguous_locations_are_unchanged(self):
        for location in ('서울시청 본관 3층', 'Zoom 빌딩 3층', '서울시청 / Zoom',
                         '온라인교육센터', '미디어센터', '플랫폼A', 'YouTube 서울 스튜디오'):
            with self.subTest(location=location):
                result = self.wrap([event('x', start='2026-10-06', location=location, attendance_mode='online')])
                self.assertEqual(result['candidate']['events'][0]['location'], location)
                self.assertEqual(result['metadata']['online_location_null_count'], 0)
                self.assertEqual(result['status'], 'candidate')
                self.assertEqual(result['candidate']['events'][0]['attendance_mode'], 'unknown')
                self.assertEqual(result['metadata']['online_physical_mode_unknown_count'], 1)
                self.assertIn('0: online physical location', result['metadata']['raw_validation_errors'])

    def test_other_attendance_modes_and_null_location_are_unchanged(self):
        items = [event('x', start='2026-10-06', location='Zoom', attendance_mode=m)
                 for m in ('offline','hybrid','unknown')]
        items.append(event('x', start='2026-10-06', attendance_mode='online'))
        result = self.wrap(items)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['metadata']['online_location_null_count'], 0)
        self.assertEqual([e['location'] for e in result['candidate']['events']], ['Zoom','Zoom','Zoom',None])

    def test_reversed_dates_are_preserved_and_review_reason_is_event_specific(self):
        items = [event('undated'), event('reverse', start='2026-10-07', end='2026-10-06'),
                 event('valid', start='2026-10-06', end='2026-10-07')]
        result = self.wrap(items)
        self.assertEqual(result['candidate']['events'][0]['start_date'], '2026-10-07')
        self.assertIsNone(result['candidate']['events'][0]['end_date'])
        self.assertTrue(result['review_required'])
        self.assertEqual(result['metadata']['reversed_end_date_null_count'], 1)
        self.assertEqual(result['metadata']['review_required_events'], [{
            'raw_event_index': 1, 'candidate_event_index': 0, 'review_required': True,
            'reason': 'end_date_before_start_date', 'start_date':'2026-10-07',
            'field':'end_date','before':'2026-10-06','after':None}])
        self.assertIn('1: reversed_dates', result['metadata']['raw_validation_errors'])
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(set(result['candidate']['events'][0]), set(items[1])|{'schedule_status'})

    def test_equal_and_one_sided_dates_do_not_trigger_review_reason(self):
        result = self.wrap([event('equal', start='2026-10-06', end='2026-10-06'),
            event('start', start='2026-10-06'), event('end', end='2026-10-06')])
        self.assertEqual(result['metadata']['reversed_date_review_count'], 0)
        self.assertEqual(result['metadata']['review_required_events'], [])

    def test_postprocessing_preserves_metadata_and_incomplete_status(self):
        metadata = {'checkpoint':'test', 'online_location_null_count':999, 'reversed_date_review_count':999}
        result = self.wrap([event('x', start='2026-10-06', attendance_mode='online', location='Zoom')], 'length', metadata)
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('generation_length', result['validation_errors'])
        self.assertEqual(result['metadata']['checkpoint'], 'test')
        self.assertEqual(result['metadata']['online_location_null_count'], 1)
        self.assertEqual(result['metadata']['reversed_date_review_count'], 0)
        self.assertEqual(metadata['online_location_null_count'], 999)


if __name__ == '__main__':
    unittest.main()
