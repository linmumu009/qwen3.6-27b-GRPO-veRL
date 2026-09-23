"""Release only the fixed 80-request screen after frozen content/design reviews."""
import argparse
from pathlib import Path
from cpt_two_family import read, sha, freeze, screen_packet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    quality = read(root / "quality.safe.json")
    assert quality["passed"] is True
    for name, fingerprint in quality["inputs"].items():
        assert sha(root / name) == fingerprint, "reviewed input changed"
    tasks = read(root / "development_author.private.json")["items"]
    # Generic packet/scoring logic uses form only as descriptive metadata.
    tasks = [dict(t, form=t["measurement_tier"]) for t in tasks]
    families = {f["family_id"]: f for f in read(root / "source_only.private.json")["families"]}
    packet = screen_packet(tasks, families)
    freeze(root / "packet.private.json", packet)
    old = read(Path("CPT_resources/llin-presentation-20260923-01/execution.safe.json"))
    reg = dict(id="llin-matrix-screen-20260923-01", max_calls=80, training_allowed=False,
               quality_released=True, quality_sha256=sha(root / "quality.safe.json"),
               packet_sha256=sha(root / "packet.private.json"),
               runner_sha256=sha(Path("scripts/run_cpt_matrix_screen.py")),
               protocol_code_sha256=old["protocol_code_sha256"], model_path=old["model_path"],
               temperature=0, seed=1024, max_tokens=96, max_model_len=8192, tp=8, max_num_seqs=32,
               registered_plan_sha256=sha(Path("docs/cpt_matrix_trial_registration_20260923.md")),
               interpretation="No additional reference versus full reference; MU1-3 already state core prerequisites. Not a pure source-recall test.")
    freeze(root / "execution.safe.json", reg)
    print("Released exactly 80 P1 screening requests; training remains disallowed.")


if __name__ == "__main__":
    main()
