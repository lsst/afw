from .record import addRecordMethods
from .table import addTableMethods
from .sortedCatalog import addSortedCatalogMethods
from ._source import SourceRecord, SourceTable, SourceCatalog, SourceFitsFlags

__all__ = ["SOURCE_IO_NO_FOOTPRINTS", "SOURCE_IO_NO_HEAVY_FOOTPRINTS"]


def addSourceCatalogMethods(cls):
    """Add pure Python methods to the SourceCatalog class
    """
    def getChildren(self, parent, *args):
        """Return the subset of self for which the parent field equals the given value.

        In order for this method to return the correct result, it must be sorted by parent
        (i.e. self.isSorted(SourceTable.getParentKey()) must be True).  This is naturally the
        case with SourceCatalogs produced by the detection and deblending tasks, but it may
        not be true when concatenating multiple such catalogs.

        Additional Catalogs or sequences whose elements correspond in order to the records
        of self (i.e. zip(self, *args) is valid) will be subset using the same slice object
        used on self, and these subsets will be returned along with the subset of self.
        """
        if not self.isSorted(SourceTable.getParentKey()):
            raise AssertionError("The table is not sorted by parent, so cannot getChildren")
        s = self.equal_range(parent, SourceTable.getParentKey())
        if args:
            return (self[s],) + tuple(arg[s] for arg in args)
        else:
            return self[s]

    cls.getChildren = getChildren
    addSortedCatalogMethods(cls)


addRecordMethods(SourceRecord)
addTableMethods(SourceTable)
addSourceCatalogMethods(SourceCatalog)

# for backwards compatibility
SOURCE_IO_NO_FOOTPRINTS = SourceFitsFlags.SOURCE_IO_NO_FOOTPRINTS
SOURCE_IO_NO_HEAVY_FOOTPRINTS = SourceFitsFlags.SOURCE_IO_NO_HEAVY_FOOTPRINTS
