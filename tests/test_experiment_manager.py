from app.experiment.manager import ExperimentManager


def test_manager_creation():

    manager = ExperimentManager()

    assert manager.active() is False