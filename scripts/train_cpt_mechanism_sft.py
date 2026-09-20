"""Process-local SFT entry point that pins engine AND sampler seeds to one."""


def install_seed_contract(trainer_class):
    original=trainer_class._build_dataloader
    def build(self):
        assert self.config.trainer.seed==1 and self.config.engine.seed==1
        original(self)
        self.train_sampler.seed=1
        self.train_sampler.set_epoch(0)
    trainer_class._build_dataloader=build


if __name__=='__main__':
    from verl.trainer import sft_trainer
    install_seed_contract(sft_trainer.SFTTrainer)
    sft_trainer.main()
