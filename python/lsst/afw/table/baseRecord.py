from __future__ import absolute_import, division, print_function

from past.builtins import basestring

from ._baseRecord import BaseRecord

def _baseRecordSet(self, key, value):
    """
    Given a string or `lsst.afw.table.Key`, find the `BaseRecord` for the given key and set its value.
    """
    if isinstance(key, basestring):
        try:
            funcKey = self.getSchema().find(key).key
        except:
            raise KeyError("Field '%s' not found in Schema." % key)
    else:
        funcKey = key
    
    try:
        prefix, suffix = type(funcKey).__name__.split("_")
    except ValueError:
        # Try using a functor
        return key.set(self, value)
    except Exception:
        raise TypeError("Argument to BaseRecord.set must be a string or Key.")

    if prefix != "Key":
        raise TypeError("Argument to BaseRecord.set must be a string or Key.")
    method = getattr(self, 'set'+suffix)
    return method(funcKey, value)

def _baseRecordSetItem(self, key, value):
    """
    Given a string or `lsst.afw.table.Key`, find the `BaseRecord` for the given key and set its value.
    """
    self.set(key, value)

def _baseRecordGet(self, key):
    """
    Given a string or `lsst.afw.table.Key`, get the `BaseRecord` for the given key and return it.
    """
    if isinstance(key, basestring):
        try:
            funcKey = self.getSchema().find(key).key
        except:
            raise KeyError("Field '%s' not found in Schema." % key)
    else:
        funcKey = key
    
    try:
        prefix, suffix = type(funcKey).__name__.split("_")
    except ValueError:
        # Try using a functor
        return key.get(self)
        
    except Exception:        
        raise TypeError("Argument to BaseRecord.set must be a string or Key.")
    if prefix != "Key":
        raise TypeError("Argument to BaseRecord.set must be a string or Key.")
    method = getattr(self, '_get_'+suffix)
    return method(funcKey)

def _baseRecordGetItem(self, key):
    """
    Given a string or `lsst.afw.table.Key`, get the `BaseRecord` for the given key and return it.
    """
    if isinstance(key, basestring):
        try:
            return self[self.getSchema().find(key).key]
        except:
            raise KeyError("Field '%s' not found in Schema." % key)
    # Try to get the item
    try: 
        return self._getitem_(key)
    except TypeError:
        # Try to get a functor
        return key.get(self)

def _extract(self, *patterns, **kwds):
    """
    Extract a dictionary of {<name>: <field-value>} in which the field names
    match the given shell-style glob pattern(s).
    Any number of glob patterns may be passed; the result will be the union of all
    the result of each glob considered separately.
    Additional optional arguments may be passed as keywords:
      items ------ The result of a call to self.schema.extract(); this will be used instead
                   of doing any new matching, and allows the pattern matching to be reused
                   to extract values from multiple records.  This keyword is incompatible
                   with any position arguments and the regex, sub, and ordered keyword
                   arguments.
      split ------ If True, fields with named subfields (e.g. points) will be split into
                   separate items in the dict; instead of {"point": lsst.afw.geom.Point2I(2,3)},
                   for instance, you'd get {"point.x": 2, "point.y": 3}.
                   Default is False.
      regex ------ A regular expression to be used in addition to any glob patterns passed
                   as positional arguments.  Note that this will be compared with re.match,
                   not re.search.
      sub -------- A replacement string (see re.MatchObject.expand) used to set the
                   dictionary keys of any fields matched by regex.
      ordered----- If True, a collections.OrderedDict will be returned instead of a standard
                   dict, with the order corresponding to the definition order of the Schema.
                   Default is False.
    """
    d = kwds.pop("items", None)
    split = kwds.pop("split", False)
    if d is None:
        d = self.schema.extract(*patterns, **kwds).copy()
    elif kwds:
        raise ValueError("Unrecognized keyword arguments for extract: %s" % ", ".join(kwds.keys()))
    for name, schemaItem in list(d.items()):  # must use list because we might be adding/deleting elements
        key = schemaItem.key
        if split and key.HAS_NAMED_SUBFIELDS:
            for subname, subkey in zip(key.subfields, key.subkeys):
                d["%s.%s" % (name, subname)] = self.get(subkey)
            del d[name]
        else:
            d[name] = self.get(schemaItem.key)
    return d

BaseRecord.set = _baseRecordSet
BaseRecord.get = _baseRecordGet
BaseRecord.__setitem__ = _baseRecordSetItem
BaseRecord.__getitem__ = _baseRecordGetItem
BaseRecord.extract = _extract
BaseRecord.schema = property(BaseRecord.getSchema)