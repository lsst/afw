# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import collections.abc

import numpy

import lsst.geom
from ._schemaMapper import SchemaMapper
from .aggregates import CoordKey
from .source import SourceRecord


class MultiMatch:
    """Initialize a multi-catalog match.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        Schema shared by all catalogs to be included in the match.
    dataIdFormat : `dict`
        Set of name: type for all data ID keys (e.g. {"visit":int,
        "ccd":int}).
    coordField : `str`, optional
        Prefix for _ra and _dec fields that contain the
        coordinates to use for the match.
    idField : `str`, optional
        Name of the field in schema that contains unique object
        IDs.
    radius : `lsst.geom.Angle`, optional
        Maximum separation for a match.  Defaults to 0.5 arcseconds.
    RecordClass : `lsst.afw.table.BaseRecord`
        Type of record to expect in catalogs to be matched.
    """

    def __init__(self, schema, dataIdFormat, coordField="coord", idField="id", radius=None,
                 RecordClass=SourceRecord):
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
        outSchema.setAliasMap(self.mapper.getInputSchema().getAliasMap())
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
        """Create a new result record from the given input record, using the
        given data ID and object ID to fill in additional columns.

        Parameters
        ----------
        inputRecord : `lsst.afw.table.source.sourceRecord`
            Record to use as the reference for the new result.
        dataId : `DataId` or `dict`
            Data id describing the data.
        objId : `int`
            Object id of the object to be added.

        Returns
        -------
        outputRecord : `lsst.afw.table.source.sourceRecord`
            Newly generated record.
        """
        outputRecord = self.table.copyRecord(inputRecord, self.mapper)
        for name, key in self.dataIdKeys.items():
            outputRecord.set(key, dataId[name])
        outputRecord.set(self.objectKey, objId)
        return outputRecord

    def add(self, catalog, dataId):
        """Add a new catalog to the match, corresponding to the given data ID.
        The new catalog is appended to the `self.result` and
        `self.reference` catalogs.

        Parameters
        ----------
        catalog : `lsst.afw.table.base.Catalog`
            Catalog to be added to the match result.
        dataId : `DataId` or `dict`
            Data id for the catalog to be added.
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
        """Return the final match catalog, after sorting it by object, copying
        it to ensure contiguousness, and optionally removing ambiguous
        matches.

        After calling finish(), the in-progress state of the matcher
        is returned to the state it was just after construction, with
        the exception of the object ID counter (which is not reset).

        Parameters
        ----------
        removeAmbiguous : `bool`, optional
            Should ambiguous matches be removed from the match
            catalog?  Defaults to True.

        Returns
        -------
        result : `lsst.afw.table.base.Catalog`
            Final match catalog, sorted by object.
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
    """A mapping (i.e. dict-like object) that provides convenient
    operations on the concatenated catalogs returned by a MultiMatch
    object.

    A GroupView provides access to a catalog of grouped objects, in
    which the grouping is indicated by a field for which all records
    in a group have the same value.  Once constructed, it allows
    operations similar to those supported by SQL "GROUP BY", such as
    filtering and aggregate calculation.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        Catalog schema to use for the grouped object catalog.
    ids : `List`
        List of identifying keys for the groups in the catalog.
    groups : `List`
        List of catalog subsets associated with each key in ids.
    """

    @classmethod
    def build(cls, catalog, groupField="object"):
        """Construct a GroupView from a concatenated catalog.

        Parameters
        ----------
        catalog : `lsst.afw.table.base.Catalog`
            Input catalog, containing records grouped by a field in
            which all records in the same group have the same value.
            Must be sorted by the group field.
        groupField : `str`, optional
            Name or Key for the field that indicates groups.  Defaults
            to "object".

        Returns
        -------
        groupCatalog : `lsst.afw.table.multiMatch.GroupView`
            Constructed GroupView from the input concatenated catalog.
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
        self.schema = schema
        self.ids = ids
        self.groups = groups
        self.count = sum(len(cat) for cat in self.groups)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return self.ids

    def __getitem__(self, key):
        index = numpy.searchsorted(self.ids, key)
        if self.ids[index] != key:
            raise KeyError("Group with ID {0} not found".format(key))
        return self.groups[index]

    def where(self, predicate):
        """Return a new GroupView that contains only groups for which the
        given predicate function returns True.

        The predicate function is called once for each group, and
        passed a single argument: the subset catalog for that group.

        Parameters
        ----------
        predicate :
            Function to identify which groups should be included in
            the output.

        Returns
        -------
        outGroupView : `lsst.afw.table.multiMatch.GroupView`
            Subset GroupView containing only groups that match the
            predicate.
        """
        mask = numpy.zeros(len(self), dtype=bool)
        for i in range(len(self)):
            mask[i] = predicate(self.groups[i])
        return type(self)(self.schema, self.ids[mask], self.groups[mask])

    def aggregate(self, function, field=None, dtype=float):
        """Run an aggregate function on each group, returning an array with
        one element for each group.

        Parameters
        ----------
        function :
            Callable object that computes the aggregate value.  If
            `field` is None, called with the entire subset catalog as an
            argument.  If `field` is not None, called with an array view
            into that field.
        field : `str`, optional
            A string name or Key object that indicates a single field the aggregate
            is computed over.
        dtype :
            Data type of the output array.

        Returns
        -------
        result : Array of `dtype`
            Aggregated values for each group.
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
        """Run a non-aggregate function on each group, returning an array with
        one element for each record.

        Parameters
        ----------
        function :
            Callable object that computes the aggregate value.  If field is None,
            called with the entire subset catalog as an argument.  If field is not
            None, called with an array view into that field.
        field : `str`
            A string name or Key object that indicates a single field the aggregate
            is computed over.
        dtype :
            Data type for the output array.

        Returns
        -------
        result : `numpy.array` of `dtype`
            Result of the function calculated on an element-by-element basis.
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
