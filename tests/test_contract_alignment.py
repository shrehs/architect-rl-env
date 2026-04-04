import yaml

from env.tasks import TASKS, TASK_CLASS_NAMES


def test_openenv_yaml_task_ids_align_with_code_registry() -> None:
    with open("openenv.yaml", "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    yaml_task_ids = [item["id"] for item in spec.get("tasks", [])]
    assert yaml_task_ids == ["easy", "medium", "hard"]
    assert sorted(yaml_task_ids) == sorted(TASKS.keys())


def test_task_name_conventions() -> None:
    assert TASK_CLASS_NAMES == {
        "easy": "EasyTask",
        "medium": "MediumTask",
        "hard": "HardTask",
    }
    assert set(TASK_CLASS_NAMES.keys()) == set(TASKS.keys())
