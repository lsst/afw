import collections.abc

import numpy

import lsst.geom
from .schemaMapper import SchemaMapper
from .aggregates import CoordKey
from .source import SourceRecord


class MultiMatch:

    def __init__(self, schema, dataIdFormat, coordField="coord", idField="id", radius=None,
                 RecordClass=SourceRecord):
        """Initialize a multi-catalog match.

        Arguments:
          schema -------- schema shared by all catalogs to be included in the match.
          dataIdFormat -- dict of name: type for all data ID keys (e.g. {"visit":int, "ccd":int}).
          coordField ---- prefix for _ra and _dec fields that contain the coordinates to use for the match.
          idField ------- name of the field in schema that contains unique object IDs.
          radius -------- lsst.geom.Angle; maximum separation for a match.  Defaults to 0.5 arcseconds.
          RecordClass --- type of record (a subclass of lsst.afw.table.BaseRecord) to expect in catalogs
                          to be matched.
        """
        if radius is None:
            radius = 0.5*lsst.geom.arcseconds
        elif not isinstance(radius, lsst.geom.Angle):
            raise ValueError("'radius' argument must be an Angle")
        self.radius = radius
        self.mapper = SchemaMapper(schema)
        self.mapper.addMinimalSchema(schema, True)
        self.coordKey = CoordKey(schema[coordField])
        self.idKey = schema.find(idField).key
        self.dataIdKeys = {}
        outSchema = self.mapper.editOutputSchema()
        self.objectKey = outSchema.addField(
            "object", type=numpy.int64, doc="Unique ID for joined sources")
        for name, dataType in dataIdFormat.items():
            self.dataIdKeys[name] = outSchema.addField(
                name, type=dataType, doc="'%s' data ID component")
        # self.result will be a catalog containing the union of all matched records, with an 'object' ID
        # column that can be used to group matches.  Sources that have ambiguous matches may appear
        # multiple times.
        self.result = None
        # self.reference will be a subset of self.result, with exactly one record for each group of matches
        # (we'll use the one from the first catalog matched into this group)
        # We'll use this to match against each subsequent catalog.
        self.reference = None
        # A set of ambiguous objects that we may want to ultimately remove from
        # the final merged catalog.
        self.ambiguous = set()
        # Table used to allocate new records for the ouput catalog.
        self.table = RecordClass.Table.make(self.mapper.getOutputSchema())
        # Counter used to assign the next object ID
        self.nextObjId = 1

    def makeRecord(self, inputRecord, dataId, objId):
        """Create a new result record from the given input record, using the given data ID and object ID
        to fill in additional columns."""
        outputRecord = self.table.copyRecord(inputRecord, self.mapper)
        for name, key in self.dataIdKeys.items():
            outputRecord.set(key, dataId[name])
        outputRecord.set(self.objectKey, objId)
        return outputRecord

    def add(self, catalog, dataId):
        """Add a new catalog to the match, corresponding to the given data ID.
        """
        if self.result is None:
            self.result = self.table.Catalog(self.table)
            for record in catalog:
                self.result.append(self.makeRecord(
                    record, dataId, objId=self.nextObjId))
                self.nextObjId += 1
            self.reference = self.result.copy(deep=False)
            return
        catalog.sort(self.idKey)  # pre-sort for speedy by-id access later.
        # Will remove from this set as objects are matched.
        unmatchedIds = {record.get(self.idKey) for record in catalog}
        # Temporary dict mapping new source ID to a set of associated objects.
        newToObj = {}
        matches = lsst.afw.table.matchRaDec(self.reference, catalog, self.radius)
        matchedRefIds = set()
        matchedCatIds = set()
        for refRecord, newRecord, distance in matches:
            objId = refRecord.get(self.objectKey)
            if objId in matchedRefIds:
                # We've already matched this object against another new source,
                # mark it as ambiguous.
                self.ambiguous.add(objId)
            matchedRefIds.add(objId)
            if newRecord.get(self.idKey) in matchedCatIds:
                # We've already matched this new source to one or more other objects
                # Mark all involved objects as ambiguous
                self.ambiguous.add(objId)
                self.ambiguous |= newToObj.get(newRecord.get(self.idKey), set())
            matchedCatIds.add(newRecord.get(self.idKey))
            unmatchedIds.discard(newRecord.get(self.idKey))
            # Populate the newToObj dict (setdefault trick is an idiom for
            # appending to a dict-of-sets)
            newToObj.setdefault(newRecord.get(self.idKey), set()).add(objId)
            # Add a new result record for this match.
            self.result.append(self.makeRecord(newRecord, dataId, objId))
        # Add any unmatched sources from the new catalog as new objects to both
        # the joined result catalog and the reference catalog.
        for objId in unmatchedIds:
            newRecord = catalog.find(objId, self.idKey)
            resultRecord = self.makeRecord(newRecord, dataId, self.nextObjId)
            self.nextObjId += 1
            self.result.append(resultRecord)
            self.reference.append(resultRecord)

    def finish(self, removeAmbiguous=True):
        """Return the final match catalog, after sorting it by object, copying it to ensure contiguousness,
        and optionally removing ambiguous matches.

        After calling finish(), the in-progress state of the matcher is returned to the state it was
        just after construction, with the exception of the object ID counter (which is not reset).
        """
        if removeAmbiguous:
            result = self.table.Catalog(self.table)
            for record in self.result:
                if record.get(self.objectKey) not in self.ambiguous:
                    result.append(record)
        else:
            result = self.result
        result.sort(self.objectKey)
        result = result.copy(deep=True)
        self.result = None
        self.reference = None
        self.ambiguous = set()
        return result


class GroupView(collections.abc.Mapping):
    """A mapping (i.e. dict-like object) that provides convenient operations on the concatenated
    catalogs returned by a MultiMatch object.

    A GroupView provides access to a catalog of grouped objects, in which the grouping is indicated by
    a field for which all records in a group have the same value.  Once constructed, it allows operations
    similar to those supported by SQL "GROUP BY", such as filtering and aggregate calculation.
    """

    @classmethod
    def build(cls, catalog, groupField="object"):
        """!Construct a GroupView from a concatenated catalog.

        @param[in]  cls         (Class; omit this argument, but Doxygen wants it mentioned)
        @param[in]  catalog     Input catalog, containing records grouped by a field in which all records
                                in the same group have the same value.  Must be sorted by the group field.
        @param[in]  groupField  Name or Key for the field that indicates groups.
        """
        groupKey = catalog.schema.find(groupField).key
        ids, indices = numpy.unique(catalog.get(groupKey), return_index=True)
        groups = numpy.zeros(len(ids), dtype=object)
        ends = list(indices[1:]) + [len(catalog)]
        for n, (i1, i2) in enumerate(zip(indices, ends)):
            # casts are a work-around for DM-8557
            groups[n] = catalog[int(i1):int(i2)]
            assert (groups[n].get(groupKey) == ids[n]).all()
        return cls(catalog.schema, ids, groups)

    def __init__(self, schema, ids, groups):
        """Direct constructor; most users should call build() instead.

        This constructor takes the constituent arrays of the object directly, to allow multiple
        methods for construction.
        """
        self.schema = schema
        self.ids = ids
        self.groups = groups
        self.count = sum(len(cat) for cat in self.groups)

    def __len__(self):
        """Return the number of groups"""
        return len(self.ids)

    def __iter__(self):
        """Iterate over group field values"""
        return self.ids

    def __getitem__(self, key):
        """Return the catalog subset that corresponds to an group field value"""
        index = numpy.searchsorted(self.ids, key)
        if self.ids[index] != key:
            raise KeyError("Group with ID {0} not found".format(key))
        return self.groups[index]

    def where(self, predicate):
        """Return a new GroupView that contains only groups for which the given predicate function
        returns True.

        The predicate function is called once for each group, and passed a single argument: the subset
        catalog for that group.
        """
        mask = numpy.zeros(len(self), dtype=bool)
        for i in range(len(self)):
            mask[i] = predicate(self.groups[i])
        return type(self)(self.schema, self.ids[mask], self.groups[mask])

    def aggregate(self, function, field=None, dtype=float):
        """!Run an aggregate function on each group, returning an array with one element for each group.

        @param[in]  function      Callable object that computes the aggregate value.  If field is None,
                                  called with the entire subset catalog as an argument.  If field is not
                                  None, called with an array view into that field.
        @param[in]  field         A string name or Key object that indicates a single field the aggregate
                                  is computed over.
        @param[in]  dtype         Data type for the output array.
        """
        result = numpy.zeros(len(self), dtype=dtype)
        if field is not None:
            key = self.schema.find(field).key

            def f(cat):
                return function(cat.get(key))
        else:
            f = function
        for i in range(len(self)):
            result[i] = f(self.groups[i])
        return result

    def apply(self, function, field=None, dtype=float):
        """!Run a non-aggregate function on each group, returning an array with one element for each record.

        @param[in]  function      Callable object that computes the aggregate value.  If field is None,
                                  called with the entire subset catalog as an argument.  If field is not
                                  None, called with an array view into that field.
        @param[in]  field         A string name or Key object that indicates a single field the aggregate
                                  is computed over.
        @param[in]  dtype         Data type for the output array.
        """
        result = numpy.zeros(self.count, dtype=dtype)
        if field is not None:
            key = self.schema.find(field).key

            def f(cat):
                return function(cat.get(key))
        else:
            f = function
        last = 0
        for i in range(len(self)):
            next = last + len(self.groups[i])
            result[last:next] = f(self.groups[i])
            last = next
        return result
