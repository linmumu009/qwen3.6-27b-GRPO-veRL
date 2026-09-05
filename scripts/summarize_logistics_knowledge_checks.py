"""Emit only fixed-key aggregate counts from private knowledge conversion checks."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    failure_keys = ('entailed_by_gold','polarity_preserved','scope_preserved',
                    'all_gold_covered','standalone','no_unsupported_facts')
    failures = Counter()
    for row in rows:
        audit = row.get('audit')
        if isinstance(audit,dict):
            for key in failure_keys:
                if audit.get(key) is not True:
                    failures[key] += 1
            if audit.get('issues') != []:
                failures['issues_present'] += 1
        else:
            failures['not_audited'] += 1
            candidate = row.get('candidate')
            if not isinstance(candidate, dict):
                failures['invalid_generation_json'] += 1
            elif candidate.get('reason') == 'full source options exceed model context':
                failures['source_context_exceeded'] += 1
            elif candidate.get('status') != 'candidate':
                failures['model_withheld'] += 1
            else:
                failures['candidate_without_valid_audit'] += 1
    print(json.dumps({'private_content_included':False,'processed':len(rows),
        'automatic_passed':sum(r['automatic_entailment_passed'] for r in rows),
        'withheld':sum(not r['automatic_entailment_passed'] for r in rows),
        'check_failure_counts_nonexclusive':dict(failures)},indent=2))


if __name__ == '__main__':
    main()
