"""T43: evaluation protocols are config, fixed before results, and enforced (D-028)."""

import pytest

from indichallu.data.schema import Split
from indichallu.eval.protocol import Protocol, load_protocol


def test_m1_protocol_is_the_approved_one():
    p = load_protocol("m1")
    assert p.decision == "D-028" and p.dev_split is Split.VAL
    assert p.threshold_objective == "macro_response_f1"
    assert p.allow_test is False and p.refusals == "separate"


def test_m0_protocol_reproduces_the_pilot():
    p = load_protocol("m0")
    assert p.threshold_objective == "span_f1" and p.allow_test and p.refusals == "claims"


def test_m1_refuses_test_but_allows_val():
    p = load_protocol("m1")
    p.check_split(Split.VAL)
    p.check_split("val")
    with pytest.raises(PermissionError, match="D-028"):
        p.check_split(Split.TEST)
    load_protocol("m0").check_split(Split.TEST)  # the pilot was allowed to


def test_record_is_what_a_run_stores():
    assert load_protocol("m1").record() == {
        "name": "m1",
        "decision": "D-028",
        "dev_split": "val",
        "allow_test": False,
        "threshold_objective": "macro_response_f1",
        "refusals": "separate",
    }


@pytest.mark.parametrize(
    "change, match",
    [
        ({"threshold_objective": "accuracy"}, "objective"),
        ({"refusals": "ignore"}, "refusals"),
        ({"dev_split": Split.TEST}, "test"),
        ({"allow_test": "false"}, "boolean"),  # a quoted string would be truthy
        ({"allow_test": 0}, "boolean"),
    ],
)
def test_invalid_protocols_are_rejected(change, match):
    base = dict(
        name="x",
        decision="D-0",
        dev_split=Split.VAL,
        allow_test=False,
        threshold_objective="span_f1",
        refusals="claims",
    )
    with pytest.raises(ValueError, match=match):
        Protocol(**{**base, **change})


def test_a_quoted_allow_test_in_yaml_cannot_unlock_test(tmp_path):
    """Codex on #32: `allow_test: "false"` must be rejected, not read as true."""
    (tmp_path / "q.yaml").write_text(
        'name: q\ndecision: D-0\ndev_split: val\nallow_test: "false"\n'
        "threshold_objective: span_f1\nrefusals: claims\n",
        "utf-8",
    )
    with pytest.raises(ValueError, match="boolean"):
        load_protocol("q", root=tmp_path)


def test_config_name_must_match_its_file(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "name: b\ndecision: D-0\ndev_split: val\nallow_test: false\n"
        "threshold_objective: span_f1\nrefusals: claims\n",
        "utf-8",
    )
    with pytest.raises(ValueError, match="name: a"):
        load_protocol("a", root=tmp_path)
