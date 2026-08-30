from pathlib import Path
import py_compile
import shutil

from scripts.patch_verl_fully_async_validation_step import (
    patch_base_trainer,
    patch_rollouter,
    patch_trainer,
)


def test_validation_patch_passes_policy_version_and_is_idempotent(tmp_path: Path):
    trainer = tmp_path / "fully_async_trainer.py"
    trainer.write_text(
        """\
async def first(self):
    val_metrics = await self.rollouter.do_validate.remote()

async def second(self):
    val_metrics = await self.rollouter.do_validate.remote()
""",
        encoding="utf-8",
    )

    assert patch_trainer(trainer) == "patched"
    assert patch_trainer(trainer) == "already-patched"
    text = trainer.read_text(encoding="utf-8")
    assert text.count("do_validate.remote(self.current_param_version)") == 2
    assert "LLIN_FULLY_ASYNC_VALIDATION_STEP" in text


def test_rollouter_patch_restores_its_data_counter(tmp_path: Path):
    rollouter = tmp_path / "fully_async_rollouter.py"
    rollouter.write_text(
        '''\
    def do_validate(self):
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics
''',
        encoding="utf-8",
    )

    assert patch_rollouter(rollouter) == "patched"
    assert patch_rollouter(rollouter) == "already-patched"
    text = rollouter.read_text(encoding="utf-8")
    assert "validation_step: int | None = None" in text
    assert "self.global_steps = int(validation_step)" in text
    assert "self.global_steps = original_global_steps" in text


def test_both_ray_roles_apply_validation_step_patch():
    root = Path(__file__).resolve().parents[1]
    for name in ("start_ray_m05.sh", "start_ray_m06.sh"):
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        if "start_ray_rollout_node.sh" in text:
            text += (root / "scripts" / "start_ray_rollout_node.sh").read_text(encoding="utf-8")
        assert "patch_verl_fully_async_validation_step.py" in text


def test_patch_compiles_against_vendored_verl_sources(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = root / "reference" / "verl" / "verl" / "experimental" / "fully_async_policy"
    base_source = root / "reference" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    trainer = tmp_path / "fully_async_trainer.py"
    rollouter = tmp_path / "fully_async_rollouter.py"
    base_trainer = tmp_path / "ray_trainer.py"
    shutil.copy2(source / trainer.name, trainer)
    shutil.copy2(source / rollouter.name, rollouter)
    shutil.copy2(base_source, base_trainer)

    assert patch_trainer(trainer) == "patched"
    assert patch_rollouter(rollouter) == "patched"
    assert patch_base_trainer(base_trainer) == "patched"
    assert patch_base_trainer(base_trainer) == "already-patched"
    py_compile.compile(str(trainer), doraise=True)
    py_compile.compile(str(rollouter), doraise=True)
    py_compile.compile(str(base_trainer), doraise=True)

    patched = base_trainer.read_text(encoding="utf-8")
    assert "LLIN_VALIDATION_IDENTITY_JOIN_V1" in patched
    assert "align_returned_validation(" in patched
    assert "select_idxs(expected_indices)" in patched
    assert "select_idxs(returned_indices)" in patched
    assert "unpad_dataproto(test_output_gen_batch_padded" not in patched
    assert "identity_status.jsonl" in patched
