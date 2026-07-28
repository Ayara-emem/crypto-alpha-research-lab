from crypto_alpha_lab.data.paths import PROJECT_ROOT


def test_project_root_exists():
    assert PROJECT_ROOT.exists()

from crypto_alpha_lab.data.paths import DATA_DIR


def test_data_directory_exists():
    assert DATA_DIR.exists()

from crypto_alpha_lab.data.paths import RAW_DATA_DIR


def test_raw_directory_exists():
    assert RAW_DATA_DIR.exists()

from crypto_alpha_lab.data.paths import CACHE_DIR


def test_cache_directory_exists():
    assert CACHE_DIR.exists()

from crypto_alpha_lab.data.paths import PROCESSED_DATA_DIR


def test_processed_directory_exists():
    assert PROCESSED_DATA_DIR.exists()