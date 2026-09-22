"""Prepare a bounded, source-backed context diagnostic without generating labels."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    root = Path('CPT_resources')
    out = root/'llin-context-bridge-20260922-01'
    out.mkdir(exist_ok=True)
    if not args.verify_existing:
        assert not (out/'cases.private.json').exists(), 'Inputs already frozen'
    nll = root/'llin-source-nll-20260922-01'
    recall = root/'llin-source-recall-20260922-01'
    cases = read(nll/'cases.private.json')
    spans = read(root/'llin-source-gap-20260922-01/fact_spans.private.json')
    previous = read(Path('docs/cpt_source_gap_results_20260922.safe.json'))['units']
    predictions = [json.loads(s) for s in (recall/'results/K_direct/predictions.private.jsonl').read_text().splitlines()]
    questions = read(out/'followup_questions.private.json')
    expected = {'031182c8e97a128c8c31': 4, '05db8a7a420b9cc25b7f': 4,
                '284682e6d05c9f063d99': 2, '9adae425cbbabbd2bf0c': 2}
    tokenizer_dir = root/'llin-transfer-p4-tokenizer-20260918-01'
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    result, selection = [], []
    for c, s, u in zip(cases, spans, previous):
        uid = c['unit_id']
        assert uid == s['unit_id'] == u['unit_id']
        heading, body = c['body'].split('\n\n', 1)
        body_start = len(heading)+2
        eligible = []
        for f, history in zip(s['facts'], u['facts']):
            if history['models']['K']['supported_repeats'] != 0 or len(f['ranges']) != 1:
                continue
            a, b = f['ranges'][0]
            text = c['body'][a:b]
            if text.endswith('.') and a >= body_start and '.' in c['body'][body_start:a] and c['body'][a-1].isspace():
                eligible.append(f)
        if not eligible:
            selection.append(dict(unit_id=uid, selected=False, reason='No always-missing full-sentence fact with preceding complete source sentence'))
            continue
        f = eligible[0]
        assert f['fact_index'] == expected[uid.removeprefix('llin-core-')]
        a, b = f['ranges'][0]
        target = c['body'][a:b]
        own = next(p for p in predictions if p['unit_id'] == uid and p['repeat'] == 0)
        own_body = own['prediction'].split('\n\n', 1)[1]
        contexts = dict(source=c['body'][:a].rstrip(), self=heading+'\n\n'+own_body)
        selection.append(dict(unit_id=uid, selected=True, fact_index=f['fact_index'],
                              target_sha256=hashlib.sha256(target.encode()).hexdigest()))
        for condition, context in contexts.items():
            assert target not in context
            messages = c['messages'] + [dict(role='assistant', content=context), dict(role='user',
                content=questions[uid.removeprefix('llin-core-')]+' Answer in one sentence, according to the same glossary.')]
            prompt = tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False), add_special_tokens=False)
            result.append(dict(id=uid+'-'+condition, unit_id=uid, condition=condition,
                fact_index=f['fact_index'], messages=messages, target_sentence=target,
                required=f['required'], prompt_ids=prompt,
                answer_ids=tok.encode(target, add_special_tokens=False)))
    assert len(result) == 8
    p = out/'cases.private.json'
    if args.verify_existing:
        expected_text = json.dumps(result, indent=2)+'\n'
        assert p.read_text(encoding='utf-8') == expected_text
        # The frozen Windows artifact uses CRLF; preserve its original byte fingerprint.
        newline = '\r\n' if b'\r\n' in p.read_bytes() else '\n'
        assert p.read_bytes() == expected_text.replace('\n', newline).encode('utf-8')
        print('Existing frozen cases reproduced exactly; no files changed')
        return
    p.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    reg = dict(id=out.name, date='2026-09-22', models=['K','R'], units=4,
        contexts=['source','self'], self_context='K original recall repeat 0; same text supplied to both models; heading normalized',
        selection=selection, selection_rule='In stable original fact order, first always-missing full source sentence after another complete source sentence; no eligible fact means exclude',
        max_response_calls=16, max_nll_requests=16, max_discarded_decode_tokens=16,
        max_response_tokens=6144, repeats=1, training_steps=0, formal_reruns=0,
        cases_sha256=sha(p), runner_sha256=sha(Path('scripts/run_cpt_context_bridge.py')),
        preparation_sha256=sha(Path(__file__)),
        chat_template_sha256=sha(tokenizer_dir/'chat_template.jinja'),
        source_nll_cases_sha256=sha(nll/'cases.private.json'),
        K_prior_predictions_sha256=sha(recall/'results/K_direct/predictions.private.jsonl'),
        fact_gap_results_sha256=sha(Path('docs/cpt_source_gap_results_20260922.safe.json')),
        protocol=dict(tp=8, dtype='bfloat16', temperature=0, seed=1024, max_model_len=8192,
            max_num_seqs=32, max_response_tokens=384, prompt_logprobs=1, nll_decode_tokens=1,
            enable_thinking=False, enable_prefix_caching=False),
        primary='Within each model compare same target sentence NLL excluding EOS under K-self vs source context; positive self-minus-source means source context helps reference prediction. Also report first token and EOS.',
        secondary='Blind model/condition review of generated response against frozen target fact; support regardless natural stop, contradiction prevents pass. No credit from context, no new facts or post-hoc rubric changes.',
        blinding='Reviewer sees follow-up question, target reference, required fact and response, not preceding context or model. Freeze ratings before identity reveal.',
        limitations='Targeted question and supplied context; not free recall or generalization. Prefix content, length and style differ. R sees K-self text, not R own text. Single pass, no independent seeds.',
        stop='No extra models, prompts, seeds, retries with unresolved reservation, training or promotion. Preserve completed batches.', promotion=False)
    Path('docs/cpt_context_bridge_registration_20260922.safe.json').write_text(json.dumps(reg, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(selection=selection, response_calls=16, nll_requests=16), indent=2))


if __name__ == '__main__':
    main()
