import metacells
import metacells.version


def test_version():
    assert metacells.__version__ == metacells.version.__version__
