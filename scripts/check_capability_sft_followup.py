"""Read only safe repeat rows and prerequisite paths; never load question bodies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(paths):
    reports = {k: [json.loads(p.read_text()) for p in ps] for k, ps in paths.items()}
    maps = {k: [{r['item_hash']: r for r in report['rows']} for report in rs] for k, rs in reports.items()}
    ids = set(maps['cpt'][0])
    hashes = []
    for label, rs in reports.items():
        for report, rows in zip(rs, maps[label]):
            assert set(rows) == ids and len(rows) == len(report['rows']) == 1672
            assert sum(r['correct'] for r in rows.values()) == report['correct']
            hashes.append(report['input_sha256']['cases_jsonl'])
    assert len(set(hashes)) == 1
    stable = {k: {i for i in ids if len({r[i]['correct'] for r in rs}) == 1} for k, rs in maps.items()}
    shared = stable['cpt'] & stable['sft']
    changes = Counter()
    for i in shared:
        a, b = maps['cpt'][0][i]['correct'], maps['sft'][0][i]['correct']
        changes['both_correct' if a and b else 'regressed' if a else 'improved' if b else 'both_wrong'] += 1
    return {'items': len(ids), 'case_sha256': hashes[0],
            'repeat_correct': {k: [r['correct'] for r in rs] for k, rs in reports.items()},
            'repeat_parse_failures': {k: [r['parse_failures'] for r in rs] for k, rs in reports.items()},
            'correctness_unstable': {k: len(ids - s) for k, s in stable.items()},
            'stable_in_both': len(shared), 'stable_transitions': dict(changes),
            'runtime': {k: [{key: r[key] for key in r if key in ('runtime', 'runtime_config', 'request', 'prompt_version', 'prediction_stability')} for r in rs] for k, rs in reports.items()},
            'report_keys': {k: list(rs[0]) for k, rs in reports.items()},
            'source_sha256': {k: [hashlib.sha256(p.read_bytes()).hexdigest() for p in ps] for k, ps in paths.items()},
            'private_content_included': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    baseline = a.root / 'runs/logistics-cpt-exposure-curve-diagnostics-20260904/safe/public_eval'
    suffixes = ('.safe.json', '.safe.repeat2.json', '.safe.repeat3.json')
    result = analyze({'cpt': [baseline / ('cpt_4x' + s) for s in suffixes],
                      'sft': [a.run / ('evaluation' + s) for s in suffixes]})
    step = a.root / 'runs/llin-step120-opensource-20260825-02'
    result['step120_checkpoint_entries'] = [str(p.relative_to(step)) for p in (step / 'checkpoints').rglob('*') if p.is_dir() or p.name == '.metadata'][:80]
    result['step120_native_metadata'] = [str(p) for p in (step / 'checkpoints').rglob('.metadata')]
    result['step120_hf_exists'] = (step / 'hf_export_step120_opensource/config.json').is_file()
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
