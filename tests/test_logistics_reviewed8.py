from scripts.prepare_logistics_reviewed8 import REVIEWED, clean_label
from scripts.summarize_logistics_cpt_run import parse_metrics
import pytest
import json
from scripts.summarize_logistics_cpt_exposure_curve import summarize_curve


def test_review_scope_is_exactly_eight_unique_items():
    assert len(REVIEWED) == len(set(REVIEWED)) == 8


def test_answer_list_markers_removed_but_negative_numbers_preserved():
    assert clean_label('(4) reverse flow') == 'reverse flow'
    assert clean_label('-Inventory note') == 'Inventory note'
    assert clean_label('-10 degrees') == '-10 degrees'
    assert clean_label('a-b') == 'a-b'


def test_validation_merges_without_hiding_conflicting_training_metrics():
    metrics=parse_metrics('step:16 - train/loss:1.0\nstep:16 - val/loss:0.8\n')
    assert metrics[16] == {'train/loss':1.0,'val/loss':0.8}
    with pytest.raises(ValueError,match='conflicting'):
        parse_metrics('step:16 - train/loss:1.0\nstep:16 - train/loss:2.0\n')


def test_mask_diagnostic_requires_optimizer_but_book_contract_forbids_it(tmp_path):
    log=tmp_path/'torchrun_logs/run/attempt_0/0/stdout.log'
    log.parent.mkdir(parents=True)
    log.write_text('step:1 - perf/max_memory_allocated_gb:1 - perf/max_memory_reserved_gb:1 '
        '- perf/cpu_memory_used_gb:1 - train/loss:1 - train/grad_norm:1 - train/lr:0.000002 '
        '- train/global_tokens:3 - train/total_tokens(B):0.000000003\n')
    ckpt=tmp_path/'checkpoints/global_step_1'
    contents={}
    for name in ('model','optimizer'):
        path=ckpt/name/'dist_ckpt'
        path.mkdir(parents=True)
        (path/'.metadata').write_bytes(b'metadata')
        (path/'__0_0.distcp').write_bytes(b'state')
        contents[name]={'format':'megatron_dist_checkpoint','path':f'{name}/dist_ckpt'}
    manifest={'global_step':1,'save_contents':['model','optimizer','extra'],'contents':contents}
    (ckpt/'ckpt_contents.json').write_text(json.dumps(manifest))
    kwargs=dict(steps_per_exposure=1,total_exposures=1,sequence_tokens_per_exposure=3,checkpoint_exposures=(1,))
    result=summarize_curve(tmp_path,experiment='logistics_reviewed8_mask_all',**kwargs)
    assert result['checkpoints'][0]['optimizer_declared'] is True
    with pytest.raises(ValueError,match='unexpected manifest'):
        summarize_curve(tmp_path,**kwargs)
    manifest['save_contents']=['model','extra']
    (ckpt/'ckpt_contents.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='unexpected manifest'):
        summarize_curve(tmp_path,experiment='logistics_reviewed8_mask_all',**kwargs)


def test_final_comparison_checks_scope_and_preserves_validation(tmp_path, monkeypatch, capsys):
    from scripts.summarize_logistics_mask8 import main
    root=tmp_path/'data'
    root.mkdir()
    row={'arm':'reviewed8','prefix_tokens':1,'prefix_nll_sum':2,'answer_top1':1,
         'answer_nll_sum':1,'greedy_gold_token_prefix_exact':False,
         'generation_censored':False,'boundary_crossing':False}
    rows=[dict(row,item_hash=key,answer_tokens=4 if i==0 else 5) for i,key in enumerate(REVIEWED)]
    for name in ('step120','all','answer'):
        (root/f'{name}.json').write_text(json.dumps({'source_hashes':{'items':'fixed'},'rows':rows}))
    for name in ('manifest','mask-gate'):
        (root/f'{name}.safe.json').write_text('{}')
    for arm in ('all','answer'):
        run=tmp_path/f'logistics-mask8-{arm}-20260905'
        log=run/'torchrun_logs/run/attempt_0/0/stdout.log'
        log.parent.mkdir(parents=True)
        log.write_text('\n'.join(f'step:{step} - val/loss:0.1' for step in (16,32,48,64)))
        (run/'training_summary.safe.json').write_text(json.dumps({'total_steps':64,
            'total_sequence_tokens_with_eos':15712,'checkpoints':[{'valid':True,'optimizer_declared':True,'global_step':64}],
            'exposures':[{'loss':{'mean':1}}]*32,'peak_memory_gb':{}}))
        export=run/'hf_export_step_64/llin_export_manifest.json'
        export.parent.mkdir()
        export.write_text(json.dumps({'verification':{'valid':True}}))
    monkeypatch.setattr('sys.argv',['summary','--root',str(root),'--runs',str(tmp_path)])
    main()
    result=json.loads((root/'comparison.safe.json').read_text())
    assert result['eval']['all']['reviewed8']['answer_tokens']==39
    assert result['training']['answer']['training_set_validation_loss_by_step']['64']==0.1
    assert result['answer_vs_all']['reviewed8']['relative_answer_nll_reduction_percent']==0
    assert result['private_content_included'] is False
    capsys.readouterr()
    with pytest.raises(ValueError,match='refusing overwrite'):
        main()
