"""Pause every 50 steps for generation evaluation, preserving optimizer/RNG state."""
import os
from pathlib import Path
import subprocess


def evaluation_environment(environment):
    return {key: value for key, value in environment.items()
            if key not in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                           'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
            and not key.startswith(('TORCHELASTIC_', 'HCCL_', 'NCCL_'))}


def install_evaluation_hook(trainer_class):
    original = trainer_class._build_ckpt_handler

    def build(self):
        import torch.distributed as dist
        from verl.utils.memory_utils import aggressive_empty_cache
        original(self)
        save = self.ckpt_handler.save_checkpoint

        def save_and_evaluate(step):
            save(step=step)
            dist.barrier()
            self.training_client.to('cpu', model=True, optimizer=True, grad=True)
            aggressive_empty_cache(force_sync=True)
            dist.barrier()
            error = [None]
            if self.rank == 0:
                try:
                    run = Path(os.environ['WRONG_TEXT_RUN'])
                    (run / 'status.txt').write_text(f'evaluating_step_{step}\n')
                    command = Path(__file__).with_name('evaluate_logistics_wrong_text_checkpoint.sh')
                    with (run / f'eval_step_{step}.log').open('x') as log:
                        subprocess.run(['bash', str(command), str(step)], check=True,
                            env=evaluation_environment(os.environ), stdout=log, stderr=subprocess.STDOUT)
                except Exception as exc:
                    error[0] = str(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0]:
                raise RuntimeError('generation evaluation failed: ' + error[0])
            if step < self.total_training_steps:
                self.training_client.to('device', model=True, optimizer=True, grad=True)
                aggressive_empty_cache(force_sync=True)
                if self.rank == 0:
                    (Path(os.environ['WRONG_TEXT_RUN']) / 'status.txt').write_text(f'training_after_step_{step}\n')
            dist.barrier()

        self.ckpt_handler.save_checkpoint = save_and_evaluate

    trainer_class._build_ckpt_handler = build


if __name__ == '__main__':
    from verl.trainer import sft_trainer
    install_evaluation_hook(sft_trainer.SFTTrainer)
    sft_trainer.main()
