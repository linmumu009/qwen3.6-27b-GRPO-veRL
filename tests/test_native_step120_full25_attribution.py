from scripts.analyze_native_step120_full25_attribution import exact_sign_pvalue


def test_exact_sign_pvalue_handles_balanced_and_directional_pairs() -> None:
    assert exact_sign_pvalue(0, 0) == 1.0
    assert exact_sign_pvalue(5, 5) == 1.0
    assert exact_sign_pvalue(0, 6) == 0.03125
    assert 0.16 < exact_sign_pvalue(13, 6) < 0.17
