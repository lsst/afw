__all__ = ["imageCompressionDisabled"]

from contextlib import contextmanager
from .fits import getAllowImageCompression, setAllowImageCompression


@contextmanager
def imageCompressionDisabled():
    """Create a context where FITS image compression is disabled.

    The previous compression setting is restored on successful exit.
    """
    old = getAllowImageCompression()
    setAllowImageCompression(False)
    yield
    setAllowImageCompression(old)
