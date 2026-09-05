"""Verify real-tokenizer mask ablation on all reviewed records, without text output."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from scripts.qwen36_exam_mask_dataset import Qwen36ExamMaskDataset
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    datasets=[]
    for answer_only in (False,True):
        config=OmegaConf.create({'text_key':'text','pad_mode':'no_padding','max_length':1024,
                                'truncation':'error','answer_only':answer_only})
        datasets.append(Qwen36ExamMaskDataset(args.data,tokenizer,config))
    if len(datasets[0]) != 8 or len(datasets[1]) != 8:
        raise ValueError('expected eight reviewed records')
    all_targets=answer_targets=0
    for index in range(8):
        a,b=datasets[0][index],datasets[1][index]
        for key in ('input_ids','position_ids'):
            assert a[key].equal(b[key])
        assert a['input_ids'][-1].item()==tokenizer.eos_token_id
        assert b['loss_mask'][0].item()==0 and b['loss_mask'][-1].item()==1
        assert b['loss_mask'].roll(-1)[-1].item()==0
        assert 0 < b['loss_mask'].sum().item() < a['loss_mask'].sum().item()
        assert (b['loss_mask'] <= a['loss_mask']).all().item()
        all_targets+=int(a['loss_mask'].sum().item())
        answer_targets+=int(b['loss_mask'].sum().item())
    result={'passed':True,'items':8,'private_content_included':False,
            'identical_input_and_position_ids':True,'eos_supervised_both':True,
            'all_supervised_tokens_per_epoch':all_targets,'answer_supervised_tokens_per_epoch':answer_targets}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
