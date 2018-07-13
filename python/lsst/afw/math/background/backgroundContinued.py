from lsst.utils import continueClass
from .background import Background

__all__ = []  # import this module only for its side effects


@continueClass  # noqa F811
class Background:
    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.getImageBBox(), self.getStatsImage())
