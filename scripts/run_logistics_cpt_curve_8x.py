"""Bounded continuous CPT curve with complete checkpoint and paired evaluation gates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/workspace/llin-verl-grpo')
RUN = ROOT / 'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x'
BASE = ROOT / 'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
CASES = ROOT / 'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
CASE_SHA = 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def majority(rows):
    if len(rows) != 3 or {r['repeat'] for r in rows} != {0, 1, 2}:
        raise ValueError('Expected exactly three unique repeats')
    answer, count = Counter(tuple(r['parsed']) if r['valid'] else None for r in rows).most_common(1)[0]
    return answer is not None and count >= 2 and sum(r['correct'] for r in rows) >= 2


def paired(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError('Unmatched case identities')
    return dict(n=len(baseline), before=sum(baseline.values()), after=sum(candidate.values()),
                gains=sum(not baseline[k] and candidate[k] for k in baseline),
                losses=sum(baseline[k] and not candidate[k] for k in baseline))


def evaluation_env(environ):
    """Prefer the installed vLLM source root while preserving Ascend imports."""
    return dict(environ, ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                PYTHONPATH='/vllm:' + environ.get('PYTHONPATH', ''),
                VLLM_WORKER_MULTIPROC_METHOD='spawn', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def checkpoint_gate(path):
    manifest = json.loads((path / 'ckpt_contents.json').read_text())
    if not {'model', 'optimizer', 'extra'} <= set(manifest['save_contents']):
        raise ValueError('Incomplete training state')
    for key in ('model', 'optimizer', 'lr_scheduler', 'rng_state'):
        relative = Path(manifest['contents'][key]['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint manifest')
        directory = path / relative
        if not (directory / '.metadata').exists() or not any(directory.glob('*.distcp')):
            raise ValueError(f'Missing {key} checkpoint data')
    if not list(path.glob('data_*.pt')):
        raise ValueError('Missing dataloader state')
    return manifest


def evaluate(model, output):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_logistics_strategy_diagnostic import messages_for, parse_answers
    if hashlib.sha256(CASES.read_bytes()).hexdigest() != CASE_SHA:
        raise ValueError('Frozen evaluation source changed')
    cases = [json.loads(s) for s in CASES.read_text().splitlines() if s.strip()]
    assert len(cases) == len({r['item_hash'] for r in cases}) == 1672
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = []
    for row in cases:
        messages, _ = messages_for(row, 'original')
        ids = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        assert len(ids) + 96 <= 8192
        prompts.append(dict(prompt_token_ids=ids))
    save(output / 'protocol.safe.json', dict(model=str(model), cases_sha256=CASE_SHA, cases=1672,
         repeats=3, temperature=0, seed=1024, max_tokens=96, max_model_len=8192, tp=8,
         max_num_seqs=32, prompt='original', prompt_hashes=[hashlib.sha256(
             json.dumps(p['prompt_token_ids']).encode()).hexdigest() for p in prompts]))
    llm = LLM(model=str(model), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192, max_num_seqs=32, gpu_memory_utilization=.8, seed=1024, enforce_eager=True)
    groups = defaultdict(list)
    with (output / 'predictions.private.jsonl').open('x') as handle:
        for repeat in range(3):
            for start in range(0, len(cases), 128):
                batch = cases[start:start + 128]
                supplied = prompts[start:start + 128]
                results = llm.generate(supplied, SamplingParams(temperature=0, max_tokens=96, seed=1024))
                results = sorted(results, key=lambda x: int(x.request_id))
                assert len(results) == len(batch)
                for row, prompt, result in zip(batch, supplied, results):
                    assert list(result.prompt_token_ids) == prompt['prompt_token_ids'], 'Prompt/output mismatch'
                    answer = result.outputs[0]
                    parsed, ok = parse_answers(answer.text, len(row['options']))
                    valid = ok and answer.finish_reason == 'stop'
                    record = dict(item_hash=row['item_hash'], dataset=row['dataset'], repeat=repeat,
                        parsed=list(parsed), valid=valid, correct=valid and list(parsed) == sorted(row['expected']),
                        truncated=answer.finish_reason == 'length', output_tokens=len(answer.token_ids),
                        prediction=answer.text)
                    groups[row['item_hash']].append(record)
                    handle.write(json.dumps(record) + '\n')
                handle.flush()
                save(output / 'progress.safe.json', dict(repeat=repeat, completed=sum(map(len, groups.values()))))
    scores = {key: majority(rows) for key, rows in groups.items()}
    save(output / 'scores.safe.json', {row['item_hash']: dict(dataset=row['dataset'], correct=scores[row['item_hash']]) for row in cases})
    save(output / 'summary.safe.json', dict(requests=5016, table=[dict(dataset=dataset,
        n=sum(r['dataset'] == dataset for r in cases), correct=sum(scores[r['item_hash']] for r in cases if r['dataset'] == dataset))
        for dataset in sorted({r['dataset'] for r in cases})], truncated=sum(r['truncated'] for v in groups.values() for r in v)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--after-training', action='store_true',
                        help='Attach to existing training; never restart or overwrite it')
    parser.add_argument('--log-suffix', default='')
    args = parser.parse_args()
    os.umask(0o077)
    if args.evaluate:
        return evaluate(args.evaluate, args.output)
    import fcntl
    with (ROOT / 'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if args.after_training else fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RUN.exists() and not args.after_training:
            raise FileExistsError('Immutable run already exists; inspect before explicit recovery')
        # Eight model + fp32 Adam/master states, with conservative export/headroom.
        if not args.after_training and shutil.disk_usage(RUN.parent).free < 3_650_000_000_000:
            raise RuntimeError('Less than 3.65 TB free on dedicated experiment disk')
        code = Path(__file__).resolve().parent
        status = RUN.parent / 'status.safe.json'
        def run(command, label, env=None):
            save(status, dict(status=label))
            if '/' in args.log_suffix or '\\' in args.log_suffix:
                raise ValueError('Invalid log suffix')
            with (RUN.parent / (label + args.log_suffix + '.log')).open('x') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            env = dict(os.environ, OUTPUT_DIR=str(RUN), RUN_NAME='logistics-cpt-book-exposure-curve-8x-20260910-01')
            if args.after_training:
                save(status, dict(status='waiting_for_existing_training'))
                with (ROOT / 'runs/.logistics-cpt-book-exposure-curve.lock').open('a') as training_lock:
                    fcntl.flock(training_lock, fcntl.LOCK_EX)
                # An exited training process is not necessarily a successful one.
                if not (RUN / 'checkpoints/global_step_232/ckpt_contents.json').exists():
                    raise RuntimeError('Existing training exited without final checkpoint')
            else:
                run(['bash', str(code / 'run_logistics_cpt_curve_8x.sh')], 'training_8x', env)
            for epoch in range(1, 9):
                checkpoint_gate(RUN / f'checkpoints/global_step_{epoch * 29}')
            eval_env = evaluation_env(os.environ)
            for epoch in range(9):
                model = BASE if epoch == 0 else RUN / f'hf_export_step_{epoch * 29}'
                if epoch:
                    run([sys.executable, str(code / 'export_megatron_dist_to_hf.py'), '--actor-checkpoint',
                         str(RUN / f'checkpoints/global_step_{epoch * 29}'), '--base-model', str(BASE),
                         '--output-dir', str(model)], f'export_epoch_{epoch}')
                run([sys.executable, str(Path(__file__)), '--evaluate', str(model), '--output',
                     str(RUN / f'eval_epoch_{epoch}')], f'evaluate_epoch_{epoch}', eval_env)
            baseline = json.loads((RUN / 'eval_epoch_0/scores.safe.json').read_text())
            curve = []
            for epoch in range(9):
                current = json.loads((RUN / f'eval_epoch_{epoch}/scores.safe.json').read_text())
                for dataset in ('SC-bench-knowledge', 'LogistikaBench', 'all'):
                    a = {k: v['correct'] for k, v in baseline.items() if dataset == 'all' or v['dataset'] == dataset}
                    b = {k: v['correct'] for k, v in current.items() if dataset == 'all' or v['dataset'] == dataset}
                    curve.append(dict(epoch=epoch, dataset=dataset, **paired(a, b)))
            save(RUN / 'curve.safe.json', dict(curve=curve, promotion=False,
                 note='Selection benchmarks, not independent tests; second-stage and retention still pending'))
            save(status, dict(status='curve_complete_stage2_pending'))
        except BaseException as exc:
            save(status, dict(status='failed', error=type(exc).__name__, detail=str(exc)))
            raise


if __name__ == '__main__':
    main()
