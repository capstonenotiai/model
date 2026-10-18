"""Service candidate validation, preserved from the shared validator."""
FIELDS = ('event_type', 'title', 'start_date', 'end_date', 'start_time',
          'end_time', 'location', 'attendance_mode', 'schedule_status')

def validate(value):
    """Check intermediate structure only; never claim semantic grounding."""
    from datetime import date, time
    if not isinstance(value, dict) or set(value) != {'events'} or not isinstance(value['events'], list):
        return ['root must contain only events array']
    errors = []
    for i, e in enumerate(value['events']):
        if not isinstance(e, dict) or set(e) != set(FIELDS):
            errors.append(f'{i}: event keys'); continue
        for k, allowed in [('event_type', {'application','submission','event','result','service_change'}),
                           ('attendance_mode', {'offline','online','hybrid','unknown','not_applicable'}),
                           ('schedule_status', {'confirmed','tentative','cancelled','unknown'})]:
            if not isinstance(e[k], str) or e[k] not in allowed:
                errors.append(f'{i}: {k}')
        if not isinstance(e['title'], str) or not e['title'].strip():
            errors.append(f'{i}: title')
        if e['location'] is not None and (not isinstance(e['location'], str) or not e['location'].strip()):
            errors.append(f'{i}: location')
        for k in ('start_date','end_date','start_time','end_time'):
            v = e[k]
            if v is None: continue
            try:
                if not isinstance(v, str): raise ValueError()
                if k.endswith('date'):
                    if len(v) != 10: raise ValueError()
                    date.fromisoformat(v)
                else:
                    if len(v) != 5: raise ValueError()
                    time.fromisoformat(v)
            except ValueError:
                errors.append(f'{i}: {k}')
        if e['event_type'] != 'event' and e['attendance_mode'] != 'not_applicable':
            errors.append(f'{i}: non-event attendance')
        if e['attendance_mode'] == 'online' and e['location'] is not None:
            errors.append(f'{i}: online physical location')
    return errors
