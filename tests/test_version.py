from validate_version_code import validate_version_code
from bioinformatics_project.__version__ import __version__


def test_version():
    assert validate_version_code(__version__)
