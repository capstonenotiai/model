"""Render the trained user input format."""
def user_content(record):
    import json
    metadata = {k: record[k] for k in ('reference_time', 'default_timezone', 'input_issues')}
    parts = ['[공지 제목]\n' + record['notice_title'],
             '[입력 메타데이터]\n' + json.dumps(metadata, ensure_ascii=False)]
    for source in record['sources']:
        parts.append(f"[출처 {source['source_id']} | {source['kind']}]\n{source['text']}")
    return '\n\n'.join(parts)
