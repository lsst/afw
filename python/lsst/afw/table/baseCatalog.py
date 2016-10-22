from __future__ import absolute_import, division, print_function
import numpy as np

from ._baseCatalog import BaseCatalog
from .catalog import addCatalogMethods

__all__ = []  # import this module only for its side effects


def _asAstropy(self, cls=None, copy=False, unviewable="copy"):
    """!
    Return an astropy.table.Table (or subclass thereof) view into this catalog.

    @param[in]   cls        Table subclass to use; None implies astropy.table.Table itself.
                            Use astropy.table.QTable to get Quantity columns.

    @param[in]  copy        Whether to copy data from the LSST catalog to the astropy table.
                            Not copying is usually faster, but can keep memory from being
                            freed if columns are later removed from the Astropy view.

    @param[in]  unviewable  One of the following options, indicating how to handle field types
                            (string and Flag) for which views cannot be constructed:
                              - 'copy' (default): copy only the unviewable fields.
                              - 'raise': raise ValueError if unviewable fields are present.
                              - 'skip': do not include unviewable fields in the Astropy Table.
                            This option is ignored if copy=True.
    """
    import astropy.table
    if cls is None:
        cls = astropy.table.Table
    if unviewable not in ("copy", "raise", "skip"):
        raise ValueError("'unviewable' must be one of 'copy', 'raise', or 'skip'")
    ps = self.getMetadata()
    meta = ps.toOrderedDict() if ps is not None else None
    columns = []
    items = self.schema.extract("*", ordered=True)
    for name, item in items.items():
        key = item.key
        unit = item.field.getUnits() or None  # use None instead of "" when empty
        if key.getTypeString() == "String":
            if not copy:
                if unviewable == "raise":
                    raise ValueError(
                        "Cannot extract string unless copy=True or unviewable='copy' or 'skip'.")
                elif unviewable == "skip":
                    continue
            data = np.zeros(len(self), dtype=np.dtype((str, key.getSize())))
            for i, record in enumerate(self):
                data[i] = record.get(key)
        elif key.getTypeString() == "Flag":
            if not copy:
                if unviewable == "raise":
                    raise ValueError(
                        "Cannot extract packed bit columns unless copy=True or unviewable='copy' or 'skip'."
                    )
                elif unviewable == "skip":
                    continue
            data = self.columns.get_bool_array(key)
        elif key.getTypeString() == "Angle":
            data = self.columns.get(key)
            unit = "radian"
            if copy:
                data = data.copy()
        else:
            data = self.columns.get(key)
            if copy:
                data = data.copy()
        columns.append(
            astropy.table.Column(
                data,
                name=item.field.getName(),
                unit=unit,
                description=item.field.getDoc()
            )
        )
    return cls(columns, meta=meta, copy=False)

addCatalogMethods(BaseCatalog)
BaseCatalog.asAstropy = _asAstropy
