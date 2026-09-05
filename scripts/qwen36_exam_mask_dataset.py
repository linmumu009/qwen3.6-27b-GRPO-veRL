"""Same-text all-token versus answer-only mask ablation; EOS supervised in both."""
from scripts.qwen36_causal_lm_dataset import Qwen36CausalLMDataset
from scripts.probe_logistics_exam_answers import split_answer


def answer_start(tokenizer, text):
    prefix, _ = split_answer(text)
    encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    start = next(i for i,(_,end) in enumerate(encoded['offset_mapping']) if end > len(prefix))
    if start < 1:
        raise ValueError('invalid answer boundary')
    return start


class Qwen36ExamMaskDataset(Qwen36CausalLMDataset):
    def __init__(self,parquet_files,tokenizer,config,processor=None,max_samples=-1):
        self.answer_only = bool(config.get('answer_only',False))
        super().__init__(parquet_files,tokenizer,config,processor,max_samples)

    def __getitem__(self,item):
        result = super().__getitem__(item)
        if self.answer_only:
            text = self.dataframe.iloc[item][self.text_key]
            start = answer_start(self.tokenizer,text)
            result['loss_mask'][:start] = 0
        return result
