"""Load the full epoch-one checkpoint and replay exactly one update without saving it."""
import json
import os
from pathlib import Path

from cpt_checkpoint_resume import install_epoch_boundary_restore


def main():
    import torch
    import verl.utils.tensordict_utils as tu
    from verl.trainer import sft_trainer
    from verl.utils.distributed import initialize_global_process_group
    install_epoch_boundary_restore()
    class ProbeFinished(Exception):
        pass
    def run_sft(config):
        initialize_global_process_group()
        trainer = sft_trainer.SFTTrainer(config)
        assert trainer.resume_global_step == 29
        report_dir = Path(os.environ['CPT_PROBE_REPORT_DIR'])
        original_train = trainer.training_client.train_batch
        called = 0
        def train_once(data):
            nonlocal called
            called += 1
            assert called == 1
            lengths = trainer._get_batch_seqlens(data)
            output = original_train(data=data)
            if trainer.engine.is_mp_src_rank_with_outputs():
                values = tu.get(output, 'metrics')
                metrics = {key: float(values[key]) for key in ('loss', 'lr', 'grad_norm')}
                record = dict(resumed_step=29, replayed_step=30, input_tokens=sum(lengths),
                    metrics=metrics, model_optimizer_rng_loaded=True,
                    dataloader_boundary_normalized=True, checkpoint_written=False)
                (report_dir/f'rank_{torch.distributed.get_rank()}.safe.json').write_text(json.dumps(record,indent=2))
            raise ProbeFinished()
        trainer.training_client.train_batch = train_once
        try:
            trainer.fit()
        except ProbeFinished:
            pass
        assert called == 1, 'No resumed optimizer update was executed'
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    sft_trainer.run_sft = run_sft
    sft_trainer.main()


if __name__ == '__main__':
    main()
