from pathlib import Path


def test_main_file_exists():
    assert Path("app/main.py").exists()


def test_requirements_exists():
    assert Path("requirements.txt").exists()


def test_app_folder_exists():
    assert Path("app").exists()