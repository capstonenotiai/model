"""Slim-v9 runtime and preserved service postprocessing."""
import hashlib
import json
import re
import time

from model_server.runtime import ROOT, Runtime, TRANSPORT
FIELDS = ('event_type', 'title', 'start_date', 'end_date', 'start_time', 'end_time', 'location', 'attendance_mode')


def system_prompt(prompt_mode='slim-v9'):
    if prompt_mode != 'slim-v9':
        raise ValueError('unsupported_slim_prompt_mode')
    return (ROOT / 'model_server' / 'prompts' / 'system_prompt.txt').read_text(encoding='utf-8')


def normalize_reference_time(value):
    """Accept explicit ISO publication dates only; invalid values become null."""
    from datetime import date, datetime
    if value is None:
        return None, None
    if isinstance(value, str):
        try:
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                date.fromisoformat(value)
                return value, None
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?', value):
                offset = re.search(r'[+-](\d{2}):(\d{2})$', value)
                if offset and (int(offset[1]) > 23 or int(offset[2]) > 59):
                    return None, 'invalid_reference_time'
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return value, None
        except ValueError:
            pass
    return None, 'invalid_reference_time'


def messages(title, body, prompt_mode='slim-v9', reference_time=None):
    from model_server.prompt_input import user_content
    reference = normalize_reference_time(reference_time)[0] if prompt_mode == 'slim-v9' else None
    record = {'notice_title': title, 'reference_time': reference, 'default_timezone': 'Asia/Seoul',
        'input_issues': [], 'sources': [{'source_id': 'title', 'kind': 'title', 'text': title},
                                     {'source_id': 'body', 'kind': 'body', 'text': body}]}
    return [{'role': 'system', 'content': system_prompt(prompt_mode)}, {'role': 'user', 'content': user_content(record)}]


def nonphysical_online_location(value):
    """Only clear definite media/placeholder strings, never guess at a venue."""
    if not isinstance(value, str) or not value.strip():
        return False
    value = value.strip().lower()
    if re.fullmatch(r'(?:https?://|www\.)\S+|[^\s@]+@[^\s@]+\.[^\s@]+', value):
        return True
    if re.fullmatch(r'(?:없음|해당\s*없음|미정|추후\s*안내|별도\s*안내|개별\s*안내|장소\s*미정|null|n/?a|-)', value):
        return True
    # Whole-string matching keeps "Zoom 빌딩 3층", "서울시청 / Zoom", etc.
    # unchanged: an online marker alone is not proof that the venue is fictitious.
    return bool(re.fullmatch(
        r'[\s()\[\]:/,-]*(?:온라인|online|비대면|원격)?[\s()\[\]:/,-]*'
        r'(?:zoom|줌|youtube|유튜브|google\s*meet|구글\s*미트|'
        r'microsoft\s*teams|ms\s*teams|teams|팀즈|webex|웹엑스|'
        r'네이버\s*폼|구글\s*폼|google\s*forms|홈페이지|웹사이트|'
        r'온라인|online|비대면|인터넷|원격)'
        r'(?:[\s()\[\]:/,-]*(?:화상|회의|면접|플랫폼|라이브|live|'
        r'생중계|채널|앱|웹|홈페이지|진행|참여|접속|온라인|online))*'
        r'[\s()\[\]:/,-]*', value))


def raw_envelope(raw, finish_reason, metadata):
    from model_server.validation import validate
    online_locations, reversed_dates = [], []
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'events'} or not isinstance(value['events'], list):
            raise ValueError('slim_root_requires_events')
        if len(value['events']) > 50:
            raise ValueError('too_many_events')
        if any(not isinstance(e, dict) or set(e) != set(FIELDS) or e['event_type'] == 'result' for e in value['events']):
            raise ValueError('slim_eight_fields_required_no_result')
        # Status is absent from the trained output; do not claim confirmed dates.
        candidate = {'events': [{**e, 'schedule_status': 'unknown'} for e in value['events']]}
        orphan_times = []
        for index, event in enumerate(candidate['events']):
            if event['attendance_mode'] == 'online' and nonphysical_online_location(event['location']):
                online_locations.append({'raw_event_index': index, 'before': event['location'], 'after': None})
                event['location'] = None
            for prefix in ('start', 'end'):
                if event[prefix + '_time'] is not None and event[prefix + '_date'] is None:
                    event[prefix + '_time'] = None
                    orphan_times.append({'event_index': index, 'field': prefix + '_time'})
        if orphan_times:
            metadata = {**metadata, 'transport_omitted_orphan_times': orphan_times}
        # Extend only this slim validator. The shared compact/native contract
        # remains unchanged; interview has the same attendance rules as event.
        validation_candidate = {'events': [{**e, 'event_type': 'event'}
            if e['event_type'] == 'interview' else e for e in candidate['events']]}
        errors = validate(validation_candidate)
        for index, event in enumerate(candidate['events']):
            if len(event['title']) > 1000 or (event['location'] and len(event['location']) > 255):
                errors.append(f'{index}: backend_string_length')
            if event['start_date'] and event['end_date'] and event['start_date'] > event['end_date']:
                reversed_dates.append({'raw_event_index': index, 'review_required': True,
                    'reason': 'end_date_before_start_date', 'start_date': event['start_date'],
                    'end_date': event['end_date']})
                # The backend EventEdit validator rejects reversed dates. Keep
                # this transport invalid, preserve both dates and expose why.
                errors.append(f'{index}: reversed_dates')
            if (event['start_date'] == event['end_date'] and event['start_time'] and event['end_time']
                    and event['start_time'] > event['end_time']):
                errors.append(f'{index}: reversed_times')
    except (ValueError, TypeError, KeyError) as exc:
        candidate, errors = None, [str(exc)]
    excluded_count = 0
    if candidate is not None:
        events = candidate['events']
        retained = [i for i,e in enumerate(events) if not (e['start_date'] is None and e['end_date'] is None)]
        for record in online_locations + reversed_dates:
            record['candidate_event_index'] = retained.index(record['raw_event_index']) if record['raw_event_index'] in retained else None
        candidate['events'] = [event for event in events
            if not (event['start_date'] is None and event['end_date'] is None)]
        excluded_count = len(events) - len(candidate['events'])
    metadata = {**metadata, 'excluded_undated_events_count': excluded_count,
        'slim_postprocess_version': 'n8-20261006',
        'online_location_null_count': len(online_locations),
        'online_location_null_changes': online_locations,
        'reversed_date_review_count': len(reversed_dates),
        'review_required_events': reversed_dates}
    if finish_reason != 'stop':
        errors.append('generation_' + finish_reason)
    return {'schema_version': TRANSPORT, 'status': 'invalid' if errors else 'candidate',
        'candidate': candidate, 'raw_output': raw, 'validation_errors': errors,
        'review_required': True, 'metadata': metadata}


def envelope(raw, finish_reason, metadata):
    """Service postprocessing repairs transport fields while preserving original validity/output.

    Uncertain online physical venues retain location and become unknown mode.
    Invalid dates are nulled individually; for a reversed pair only end_date is
    nulled. No replacement date, year, venue or attendance mode is inferred.
    """
    from datetime import date
    original = raw_envelope(raw, finish_reason, metadata)
    repairs = []
    try:
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {'events'}
                or not isinstance(value['events'], list) or len(value['events']) > 50
                or any(not isinstance(e, dict) or set(e) != set(FIELDS)
                       or e['event_type'] == 'result' for e in value['events'])):
            raise ValueError('unrepairable_root_or_schema')
        for index, event in enumerate(value['events']):
            location = event['location']
            if (event['attendance_mode'] == 'online' and isinstance(location, str)
                    and location.strip() and not nonphysical_online_location(location)):
                repairs.append({'raw_event_index': index, 'field': 'attendance_mode',
                    'before': 'online', 'after': 'unknown', 'review_required': True,
                    'reason': 'online_physical_location_uncertain', 'location': location})
                event['attendance_mode'] = 'unknown'
            for field in ('start_date', 'end_date'):
                before = event[field]
                if before is None:
                    continue
                try:
                    if not isinstance(before, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', before):
                        raise ValueError('invalid_iso_date')
                    date.fromisoformat(before)
                except (ValueError, TypeError):
                    repairs.append({'raw_event_index': index, 'field': field,
                        'before': before, 'after': None, 'review_required': True,
                        'reason': 'invalid_calendar_date'})
                    event[field] = None
            if event['start_date'] and event['end_date'] and event['end_date'] < event['start_date']:
                repairs.append({'raw_event_index': index, 'field': 'end_date',
                    'before': event['end_date'], 'after': None, 'review_required': True,
                    'reason': 'end_date_before_start_date', 'start_date': event['start_date']})
                event['end_date'] = None
        retained = [i for i, e in enumerate(value['events'])
                    if e['start_date'] is not None or e['end_date'] is not None]
        for repair in repairs:
            i = repair['raw_event_index']
            repair['candidate_event_index'] = retained.index(i) if i in retained else None
        result = raw_envelope(json.dumps(value, ensure_ascii=False), finish_reason, metadata) if repairs else original
    except (ValueError, TypeError, KeyError):
        result, repairs = original, []
    result['raw_output'] = raw
    result['metadata'] = {**result['metadata'], 'slim_postprocess_version': 'task05c-20261007',
        'raw_status': original['status'], 'raw_validation_errors': original['validation_errors'],
        'postprocess_repairs': repairs, 'postprocess_review_count': len(repairs),
        'online_physical_mode_unknown_count': sum(r['reason'] == 'online_physical_location_uncertain' for r in repairs),
        'invalid_date_null_count': sum(r['reason'] == 'invalid_calendar_date' for r in repairs),
        'reversed_end_date_null_count': sum(r['reason'] == 'end_date_before_start_date' for r in repairs),
        'review_required_events': repairs + result['metadata']['review_required_events']}
    return result


def apply_interview_type(response, interview_type='map-to-event'):
    """Backend compatibility projection, preserving the native model output."""
    if interview_type not in ('map-to-event', 'passthrough'):
        raise ValueError('unsupported_interview_type')
    events = (response.get('candidate') or {}).get('events', [])
    indexes = [i for i, e in enumerate(events) if e.get('event_type') == 'interview']
    if not indexes:
        return response
    import copy
    result = copy.deepcopy(response)
    result['metadata'] = {**result.get('metadata', {}),
        'interview_type_policy': interview_type,
        'original_event_types': [{'candidate_event_index': i, 'event_type': 'interview'} for i in indexes]}
    if interview_type == 'map-to-event':
        for i in indexes:
            result['candidate']['events'][i]['event_type'] = 'event'
    return result


class SlimRuntime(Runtime):
    def __init__(self, model_dir, max_new_tokens=1024, context_limit=12288, max_seconds=60,
                 prompt_mode='slim-v9', interview_type='passthrough'):
        if prompt_mode != 'slim-v9':
            raise ValueError('unsupported_slim_prompt_mode')
        if interview_type not in ('map-to-event', 'passthrough'):
            raise ValueError('unsupported_interview_type')
        super().__init__(model_dir, max_new_tokens=max_new_tokens, context_limit=context_limit,
                         max_seconds=max_seconds, prompt_mode='compact')
        self.prompt_mode = prompt_mode
        self.interview_type = interview_type
        self.identity.update(prompt_mode=prompt_mode, prompt_version=prompt_mode+'-eight-fields-20261006',
            prompt_sha256=hashlib.sha256(system_prompt(prompt_mode).encode()).hexdigest(), examples_sha256=None)
        if prompt_mode == 'slim-v9':
            self.identity['prompt_version'] = 'slim-v9-policies11-15-20261007'

    def predict(self, title, body, reference_time=None):
        if self.prompt_mode != 'slim-v9':
            return self.predict_messages(messages(title, body, self.prompt_mode))
        reference, warning = normalize_reference_time(reference_time)
        response = self.predict_messages(messages(title, body, self.prompt_mode, reference))
        response['metadata'] = {**response.get('metadata', {}), 'reference_time': reference}
        if warning:
            response['metadata']['reference_time_warning'] = warning
        return response

    def predict_messages(self, chat):
        torch = self.torch
        rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False, truncation=False)
        count = inputs['input_ids'].shape[1]
        if count + self.max_new_tokens > self.context_limit:
            raise ValueError('input_context_exceeded')
        inputs = inputs.to('cuda:0')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                max_time=self.max_seconds, do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = output[0, count:]
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = ('stop' if len(tokens) and tokens[-1].item() in eos else
                  'length' if len(tokens) >= self.max_new_tokens else 'timeout')
        if elapsed > self.max_seconds:
            finish = 'timeout'
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        metadata = {**self.identity, 'input_tokens': count, 'output_tokens': len(tokens),
            'elapsed_seconds': round(elapsed, 3), 'finish_reason': finish,
            'input_sha256': hashlib.sha256(chat[-1]['content'].encode()).hexdigest()}
        return apply_interview_type(envelope(raw, finish, metadata), self.interview_type)
