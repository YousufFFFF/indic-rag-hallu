"""T12: deterministic, leakage-free train/val/test splits."""

import random
from collections import Counter

import pytest

from indichallu.data.download import DEFAULT_DATA_ROOT, load_pinned_dataset
from indichallu.data.loaders.indicqa import IndicQA, Passage, QAPair, load_indicqa
from indichallu.data.schema import Split
from indichallu.data.split import (
    LeakageError,
    SplitAssignment,
    SplitConfig,
    assign_groups,
    check_no_leakage,
    load_split_config,
    split_counts,
    split_dataset,
    split_stats,
)

R = {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}


def make_ds(n_titles=20, paras_per_title=1, shared_question_titles=()) -> IndicQA:
    passages, questions = [], []
    for t in range(n_titles):
        for k in range(paras_per_title):
            pid = f"hi-p{t:04d}{k:02d}"
            passages.append(Passage(pid, "hi", f"लेख {t}", f"अनुच्छेद {t}-{k}"))
            text = "साझा प्रश्न?" if t in shared_question_titles else f"प्रश्न {t}-{k}?"
            questions.append(
                QAPair(
                    qa_id=f"hi-q{len(questions):06d}",
                    lang="hi",
                    question=text,
                    gold_passage_id=pid,
                    category="SHORT",
                    answerable=True,
                    answer_status="exact",
                    answer_text="x",
                    answer_start=0,
                    answer_end=1,
                    given_answer_start=0,
                )
            )
    return IndicQA("hi", passages, questions)


# --- counts ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(10, (6, 2, 2)), (261, (157, 52, 52)), (258, (155, 52, 51)), (3, (1, 1, 1)), (0, (0, 0, 0))],
)
def test_split_counts_are_exact(n, expected):
    c = split_counts(n, R)
    assert (c[Split.TRAIN], c[Split.VAL], c[Split.TEST]) == expected
    assert sum(c.values()) == n


# --- determinism -----------------------------------------------------------------------


def test_same_seed_same_split_regardless_of_input_order():
    keys = {f"title:{i}" for i in range(100)}
    cfg = SplitConfig(13, R)
    a = assign_groups(keys, cfg)
    shuffled = list(keys)
    random.Random(0).shuffle(shuffled)
    assert assign_groups(set(shuffled), cfg) == a
    assert assign_groups(keys, SplitConfig(14, R)) != a
    assert Counter(a.values()) == {Split.TRAIN: 60, Split.VAL: 20, Split.TEST: 20}


def test_assignment_is_pinned_across_python_versions():
    """Hash-based ranking: this exact assignment must never change for seed 13. If it
    does, every reported result's split has silently changed."""
    a = assign_groups({f"k{i}" for i in range(5)}, SplitConfig(13, R))
    assert {k: v.value for k, v in a.items()} == {
        "k0": "train",
        "k1": "val",
        "k2": "train",
        "k3": "train",
        "k4": "test",
    }


# --- grouping and leakage ----------------------------------------------------------------


def test_paragraphs_of_one_article_stay_together():
    ds = make_ds(n_titles=30, paras_per_title=3)
    a = split_dataset(ds, SplitConfig(13, R, "title"))
    check_no_leakage(ds, a)
    for t in range(30):
        assert len({a.of_passage(f"hi-p{t:04d}{k:02d}") for k in range(3)}) == 1


def test_identical_questions_pull_their_articles_into_one_split():
    ds = make_ds(n_titles=30, shared_question_titles={1, 7, 22})
    for seed in range(20):  # whatever the seed, the three articles stay together
        a = split_dataset(ds, SplitConfig(seed, R))
        check_no_leakage(ds, a)
        assert len({a.of_passage(f"hi-p{t:04d}00") for t in (1, 7, 22)}) == 1


def test_leakage_checker_catches_planted_leaks():
    ds = make_ds(n_titles=10, paras_per_title=2, shared_question_titles={3, 4})
    good = split_dataset(ds, SplitConfig(13, R))
    check_no_leakage(ds, good)

    def planted(changes):
        return SplitAssignment(good.config, {**good.passage_split, **changes})

    other = {Split.TRAIN: Split.TEST, Split.VAL: Split.TEST, Split.TEST: Split.TRAIN}
    p = "hi-p000000"
    with pytest.raises(LeakageError, match="article"):  # one paragraph of an article moved
        check_no_leakage(ds, planted({p: other[good.of_passage(p)]}))
    q3, q4 = "hi-p000300", "hi-p000400"
    with pytest.raises(LeakageError, match="question"):  # shared question split apart
        check_no_leakage(
            ds,
            planted(
                {
                    q3: Split.TRAIN,
                    "hi-p000301": Split.TRAIN,
                    q4: Split.TEST,
                    "hi-p000401": Split.TEST,
                }
            ),
        )
    with pytest.raises(LeakageError, match="no split"):
        check_no_leakage(ds, SplitAssignment(good.config, {}))


def test_group_by_passage_and_untitled_passages():
    ds = make_ds(n_titles=10, paras_per_title=2)
    a = split_dataset(ds, SplitConfig(13, R, "passage"))
    check_no_leakage(ds, a)
    assert sum(split_stats(ds, a)[s]["passages"] for s in a.config.ratios) == 20


# --- config ----------------------------------------------------------------------------


def test_repo_config():
    cfg = load_split_config()
    assert cfg.seed == 13 and cfg.group_by == "title"
    assert cfg.ratios == R


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratios": {Split.TRAIN: 0.7, Split.VAL: 0.2, Split.TEST: 0.2}},
        {"ratios": {Split.TRAIN: 1.2, Split.VAL: -0.1, Split.TEST: -0.1}},
        {"ratios": {Split.TRAIN: 0.8, Split.TEST: 0.2}},
        {"ratios": R, "group_by": "sentence"},
    ],
)
def test_bad_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(seed=1, **kwargs)


# --- the real file (runs only where it has been downloaded; CI skips) -----------------


def test_real_hindi_split_has_no_leakage():
    path = load_pinned_dataset().local_path("hi", DEFAULT_DATA_ROOT)
    if not path.exists():
        pytest.skip("Hindi IndicQA not downloaded (python scripts/download_indicqa.py)")
    ds = load_indicqa(path, "hi")
    a = split_dataset(ds)
    check_no_leakage(ds, a)
    stats = split_stats(ds, a)
    assert [stats[s]["passages"] for s in (Split.TRAIN, Split.VAL, Split.TEST)] == [156, 52, 53]
    assert sum(stats[s]["questions"] for s in stats) == 1547
    assert split_dataset(ds) == a  # deterministic
