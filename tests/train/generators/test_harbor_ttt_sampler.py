import json
from pathlib import Path

from examples.train_integrations.harbor.ttt.sampler import GreedySampler, PUCTSampler
from examples.train_integrations.harbor.ttt.state import (
    State,
    read_construction_artifact,
)


def _task(tmp_path: Path, name: str = "task") -> State:
    task = tmp_path / name
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text(
        "Improve this solution:\n<<<LAST_CODE>>>\n"
        "Parent: <<<INITIAL_CONSTRUCTION>>>\n"
    )
    (task / "solution" / "solution.py").write_text("initial")
    (task / "initial_value.txt").write_text("-0.5\n")
    return State.initial(str(task))


def test_child_task_contains_verified_construction_without_mutation_leak(tmp_path):
    initial = _task(tmp_path)
    values = [0.2, 0.8]
    child = initial.make_child(
        "better",
        value=-0.4,
        timestep=7,
        tasks_dir=str(tmp_path / "states"),
        construction=values,
    )
    values[0] = 0.9

    rendered = Path(child.task_path, "instruction.md").read_text()
    artifact = json.loads(
        Path(child.task_path, "tests", "ttt_state.json").read_text()
    )
    assert "Improve this solution:\nbetter" in rendered
    assert "available (length=2)" in rendered
    assert artifact["construction"] == [0.2, 0.8]
    assert child.construction == [0.2, 0.8]
    assert initial.construction is None


def test_greedy_deduplicates_construction_before_source(tmp_path):
    initial = _task(tmp_path)
    first = initial.make_child(
        "same source", -0.4, 1, str(tmp_path / "states"), construction=[0.2, 0.8]
    )
    second = initial.make_child(
        "same source", -0.39, 1, str(tmp_path / "states"), construction=[0.3, 0.7]
    )
    duplicate = initial.make_child(
        "different source", -0.38, 1, str(tmp_path / "states"), construction=[0.2, 0.8]
    )
    sampler = GreedySampler(str(tmp_path / "sampler"), topk_children=0)
    sampler.update_states(
        [first, second, duplicate], [initial, initial, initial], save=False
    )

    assert {tuple(state.construction or []) for state in sampler._states} == {
        (0.2, 0.8),
        (0.3, 0.7),
    }


def test_puct_checkpoint_and_parent_aggregated_backup(tmp_path):
    initial = _task(tmp_path, "first")
    children = [
        initial.make_child(
            f"candidate-{index}",
            value,
            3,
            str(tmp_path / "states"),
            construction=[0.1 + index * 0.1, 0.9 - index * 0.1],
        )
        for index, value in enumerate((-0.40, -0.39, -0.41))
    ]
    sampler_dir = str(tmp_path / "sampler")
    sampler = PUCTSampler(sampler_dir, group_size=4, topk_children=2)
    sampler.sample_states(1, fallback_states=[initial])
    sampler.update_states(children, [initial] * 3, step=3)

    assert sampler._T == 1
    assert sampler._n[initial.id] == 1
    assert sampler._m[initial.id] == -0.39
    assert len([state for state in sampler._states if state.id != initial.id]) == 2
    assert sampler.sample_states(1, fallback_states=[initial])[0].value == -0.39

    resumed = PUCTSampler(sampler_dir, group_size=4, resume_step=3)
    assert resumed._T == 1
    assert resumed._m[initial.id] == -0.39
    assert len(resumed._initial_states) == 1


def test_construction_artifact_round_trip(tmp_path):
    path = tmp_path / "construction.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "construction": [0.2, 0.8],
                "selection_value": -0.4,
            }
        )
    )

    construction, selection_value = read_construction_artifact(path)
    assert construction == [0.2, 0.8]
    assert selection_value == -0.4
