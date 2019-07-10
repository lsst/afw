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

__all__ = ["makeMergedSchema", "copyIntoCatalog",
           "matchesToCatalog", "matchesFromCatalog", "copyAliasMapWithPrefix"]

import os.path

import numpy as np

import lsst.pex.exceptions as pexExcept
from ._schema import Schema
from ._schemaMapper import SchemaMapper
from ._base import BaseCatalog
from ._table import SimpleTable
from ._simple import SimpleCatalog
from ._source import SourceCatalog, SourceTable
from .match import ReferenceMatch

from lsst.utils import getPackageDir


def makeMapper(sourceSchema, targetSchema, sourcePrefix=None, targetPrefix=None):
    """Create a SchemaMapper between the input source and target schemas.

    Parameters
    ----------
    sourceSchema : :py:class:`lsst.afw.table.Schema`
        Input source schema that fields will be mapped from.
    targetSchema : :py:class:`lsst.afw.table.Schema`
        Target schema that fields will be mapped to.
    sourcePrefix : `str`, optional
        If set, only those keys with that prefix will be mapped.
    targetPrefix : `str`, optional
        If set, prepend it to the mapped (target) key name.

    Returns
    -------
    SchemaMapper : :py:class:`lsst.afw.table.SchemaMapper`
        Mapping between source and target schemas.
    """
    m = SchemaMapper(sourceSchema, targetSchema)
    for key, field in sourceSchema:
        keyName = field.getName()
        if sourcePrefix is not None:
            if not keyName.startswith(sourcePrefix):
                continue
            else:
                keyName = field.getName().replace(sourcePrefix, "", 1)
        m.addMapping(key, (targetPrefix or "") + keyName)
    return m


def makeMergedSchema(sourceSchema, targetSchema, sourcePrefix=None, targetPrefix=None):
    """Return a schema that is a deep copy of a mapping between source and target schemas.

    Parameters
    ----------
    sourceSchema : :py:class:`lsst.afw.table.Schema`
        Input source schema that fields will be mapped from.
    targetSchema : :py:class:`lsst.afw.atable.Schema`
        Target schema that fields will be mapped to.
    sourcePrefix : `str`, optional
        If set, only those keys with that prefix will be mapped.
    targetPrefix : `str`, optional
        If set, prepend it to the mapped (target) key name.

    Returns
    -------
    schema : :py:class:`lsst.afw.table.Schema`
        Schema that is the result of the mapping between source and target schemas.
    """
    return makeMapper(sourceSchema, targetSchema, sourcePrefix, targetPrefix).getOutputSchema()


def copyIntoCatalog(catalog, target, sourceSchema=None, sourcePrefix=None, targetPrefix=None):
    """Copy entries from one Catalog into another.

    Parameters
    ----------
    catalog : :py:class:`lsst.afw.table.base.Catalog`
        Source catalog to be copied from.
    target : :py:class:`lsst.afw.table.base.Catalog`
        Target catalog to be copied to (edited in place).
    sourceSchema : :py:class:`lsst.afw.table.Schema`, optional
        Schema of source catalog.
    sourcePrefix : `str`, optional
        If set, only those keys with that prefix will be copied.
    targetPrefix : `str`, optional
        If set, prepend it to the copied (target) key name

    Returns
    -------
    target : :py:class:`lsst.afw.table.base.Catalog`
        Target catalog that is edited in place.
    """
    if sourceSchema is None:
        sourceSchema = catalog.schema

    targetSchema = target.schema
    target.reserve(len(catalog))
    for i in range(len(target), len(catalog)):
        target.addNew()

    if len(catalog) != len(target):
        raise RuntimeError("Length mismatch: %d vs %d" %
                           (len(catalog), len(target)))

    m = makeMapper(sourceSchema, targetSchema, sourcePrefix, targetPrefix)
    for rFrom, rTo in zip(catalog, target):
        rTo.assign(rFrom, m)


def matchesToCatalog(matches, matchMeta):
    """Denormalise matches into a Catalog of "unpacked matches".

    Parameters
    ----------
    matches : `~lsst.afw.table.match.SimpleMatch`
        Unpacked matches, i.e. a list of Match objects whose schema
        has "first" and "second" attributes which, resepectively,
        contain the reference and source catalog entries, and a
        "distance" field (the measured distance between the reference
        and source objects).
    matchMeta : `~lsst.daf.base.PropertySet`
        Metadata for matches (must have .add attribute).

    Returns
    -------
    mergedCatalog : :py:class:`lsst.afw.table.BaseCatalog`
        Catalog of matches (with ref_ and src_ prefix identifiers for
        referece and source entries, respectively, including alias
        maps from reference and source catalogs)
    """
    if len(matches) == 0:
        raise RuntimeError("No matches provided.")

    refSchema = matches[0].first.getSchema()
    srcSchema = matches[0].second.getSchema()

    mergedSchema = makeMergedSchema(refSchema, Schema(), targetPrefix="ref_")
    mergedSchema = makeMergedSchema(
        srcSchema, mergedSchema, targetPrefix="src_")

    mergedSchema = copyAliasMapWithPrefix(refSchema, mergedSchema, prefix="ref_")
    mergedSchema = copyAliasMapWithPrefix(srcSchema, mergedSchema, prefix="src_")

    distKey = mergedSchema.addField(
        "distance", type=np.float64, doc="Distance between ref and src")

    mergedCatalog = BaseCatalog(mergedSchema)
    copyIntoCatalog([m.first for m in matches], mergedCatalog,
                    sourceSchema=refSchema, targetPrefix="ref_")
    copyIntoCatalog([m.second for m in matches], mergedCatalog,
                    sourceSchema=srcSchema, targetPrefix="src_")
    for m, r in zip(matches, mergedCatalog):
        r.set(distKey, m.distance)

    # obtain reference catalog name if one is setup
    try:
        catalogName = os.path.basename(getPackageDir("astrometry_net_data"))
    except pexExcept.NotFoundError:
        catalogName = "NOT_SET"
    matchMeta.add("REFCAT", catalogName)
    mergedCatalog.getTable().setMetadata(matchMeta)

    return mergedCatalog


def matchesFromCatalog(catalog, sourceSlotConfig=None):
    """Generate a list of ReferenceMatches from a Catalog of "unpacked matches".

    Parameters
    ----------
    catalog : :py:class:`lsst.afw.table.BaseCatalog`
        Catalog of matches.  Must have schema where reference entries
        are prefixed with "ref_" and source entries are prefixed with
        "src_".
    sourceSlotConfig : `lsst.meas.base.baseMeasurement.SourceSlotConfig`, optional
        Configuration for source slots.

    Returns
    -------
    matches : :py:class:`lsst.afw.table.ReferenceMatch`
        List of matches.
    """
    refSchema = makeMergedSchema(
        catalog.schema, SimpleTable.makeMinimalSchema(), sourcePrefix="ref_")
    refCatalog = SimpleCatalog(refSchema)
    copyIntoCatalog(catalog, refCatalog, sourcePrefix="ref_")

    srcSchema = makeMergedSchema(
        catalog.schema, SourceTable.makeMinimalSchema(), sourcePrefix="src_")
    srcCatalog = SourceCatalog(srcSchema)
    copyIntoCatalog(catalog, srcCatalog, sourcePrefix="src_")

    if sourceSlotConfig is not None:
        sourceSlotConfig.setupSchema(srcCatalog.schema)

    matches = []
    distKey = catalog.schema.find("distance").key
    for ref, src, cat in zip(refCatalog, srcCatalog, catalog):
        matches.append(ReferenceMatch(ref, src, cat[distKey]))

    return matches


def copyAliasMapWithPrefix(inSchema, outSchema, prefix=""):
    """Copy an alias map from one schema into another.

    This copies the alias map of one schema into another, optionally
    prepending a prefix to both the "from" and "to" names of the alias
    (the example use case here is for the "match" catalog created by
    `lsst.meas.astrom.denormalizeMatches` where prefixes "src_" and
    "ref_" are added to the source and reference field entries,
    respectively).

    Parameters
    ----------
    inSchema : `lsst.afw.table.Schema`
       The input schema whose `lsst.afw.table.AliasMap` is to be
       copied to ``outSchema``.
    outSchema : `lsst.afw.table.Schema`
       The output schema into which the `lsst.afw.table.AliasMap`
       from ``inSchema`` is to be copied (modified in place).
    prefix : `str`, optional
       An optional prefix to add to both the "from" and "to" names
       of the alias (default is an empty string).

    Returns
    -------
    outSchema : `lsst.afw.table.Schema`
       The output schema with the alias mappings from `inSchema`
       added.
    """
    for k, v in inSchema.getAliasMap().items():
        outSchema.getAliasMap().set(prefix + k, prefix + v)

    return outSchema
