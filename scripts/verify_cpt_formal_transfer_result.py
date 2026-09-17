"""Verify archived formal-transfer results locally with a second answer parser."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cpt_formal_transfer import verify
from verify_cpt_pilot_result import independently_parse
from audit_cpt_transfer import read_rows


def verify_local(out, history):
    cases_path = history/'frozen_cases.private.jsonl'
    result = verify(out, cases_path, history/'step120')
    remote = json.loads((out/'comparison.safe.json').read_text())
    if result != remote:
        raise ValueError('local and remote aggregates differ')
    cases = {r['item_hash']: r for r in read_rows(cases_path)}
    independent = {}
    for label in ('step120_current', 'p1', 'p3'):
        rows = read_rows(out/label/'predictions.private.jsonl')
        for row in rows:
            case = cases[row['item_hash']]
            answer, ok = independently_parse(row['prediction'], len(case['options']))
            valid = ok and not row['truncated']
            if answer != row['parsed'] or valid != row['valid']:
                raise ValueError('independent parser mismatch')
            if row['correct'] != (valid and answer == sorted(case['expected'])):
                raise ValueError('independent correctness mismatch')
        summary = json.loads((out/label/'summary.safe.json').read_text())
        if (summary['requests'] != 5016 or len(summary['table']) != 2 or
                {x['dataset'] for x in summary['table']} != {'SC-bench-knowledge', 'LogistikaBench'}):
            raise ValueError('summary coverage mismatch')
        for saved in summary['table']:
            item = next(x for x in result['table'] if x['model']==label and x['dataset']==saved['dataset'])
            if saved['n'] != item['versus_historical']['n'] or saved['correct'] != item['versus_historical']['after']:
                raise ValueError('summary score mismatch')
        if summary['truncated'] != result['audits'][label]['truncated']:
            raise ValueError('truncation summary mismatch')
        independent[label] = dict(raw_answers=len(rows), independent_parser_passed=True, summary_matched=True)
    registration = json.loads((out/'registration.safe.json').read_text())
    if (set(registration['token_checks']) != {'step120_current', 'p1', 'p3'} or
            not all(v['prompts']==1672 and v['historical_match'] for v in registration['token_checks'].values())):
        raise ValueError('prompt preflight failed')
    if json.loads((out/'status.safe.json').read_text())['status'] != 'completed_verified_remote':
        raise ValueError('remote completion missing')
    result.update(local_verification=dict(remote_aggregate_identical=True, independent_parser=independent,
        new_raw_answers=15048, historical_raw_answers=5016, token_prompt_preflight=registration['token_checks'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        status='completed_verified_both_sides', launch_registration='cpt_formal_transfer_launch_20260917.safe.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = verify_local(a.run, a.history)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], new_requests=result['new_requests'],
        table=[dict(model=x['model'], dataset=x['dataset'], correct=x['versus_current']['after'],
                    gains=x['versus_current']['gains'], losses=x['versus_current']['losses']) for x in result['table']])))
