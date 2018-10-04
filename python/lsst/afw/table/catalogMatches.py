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
           "matchesToCatalog", "matchesFromCatalog"]

import os.path

import numpy as np

import lsst.pex.exceptions as pexExcept
from .schema import Schema
from .schemaMapper import SchemaMapper
from .base import BaseCatalog
from .simple import SimpleCatalog, SimpleTable
from .source import SourceCatalog, SourceTable
from .match import ReferenceMatch

from lsst.utils import getPackageDir


def makeMapper(sourceSchema, targetSchema, sourcePrefix=None, targetPrefix=None):
    """Create a SchemaMapper between the input source and target schemas

    @param[in]  sourceSchema  input source schema that fields will be mapped from
    @param[in]  targetSchema  target schema that fields will be mapped to
    @param[in]  sourcePrefix  if set, only those keys with that prefix will be mapped
    @param[in]  targetPrefix  if set, prepend it to the mapped (target) key name

    @return     SchemaMapper between source and target schemas
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
    """Return a schema that is a deep copy of a mapping between source and target schemas
    @param[in]  sourceSchema  input source schema that fields will be mapped from
    @param[in]  targetSchema  target schema that fields will be mapped to
    @param[in]  sourcePrefix  if set, only those keys with that prefix will be mapped
    @param[in]  targetPrefix  if set, prepend it to the mapped (target) key name

    @return     schema        schema that is the result of the mapping between source and target schemas
    """
    return makeMapper(sourceSchema, targetSchema, sourcePrefix, targetPrefix).getOutputSchema()


def copyIntoCatalog(catalog, target, sourceSchema=None, sourcePrefix=None, targetPrefix=None):
    """Copy entries from one Catalog into another

    @param[in]     catalog       source catalog to be copied from
    @param[in/out] target        target catalog to be copied to (edited in place)
    @param[in]     souceSchema   schema of source catalog (optional)
    @param[in]     sourcePrefix  if set, only those keys with that prefix will be copied
    @param[in]     targetPrefix  if set, prepend it to the copied (target) key name
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
    """Denormalise matches into a Catalog of "unpacked matches"

    @param[in] matches    unpacked matches, i.e. a list of Match objects whose schema
                          has "first" and "second" attributes which, resepectively, contain the
                          reference and source catalog entries, and a "distance" field (the
                          measured distance between the reference and source objects)
    @param[in] matchMeta  metadata for matches (must have .add attribute)

    @return  lsst.afw.table.BaseCatalog of matches (with ref_ and src_ prefix identifiers
             for referece and source entries, respectively)
    """
    if len(matches) == 0:
        raise RuntimeError("No matches provided.")

    refSchema = matches[0].first.getSchema()
    srcSchema = matches[0].second.getSchema()

    mergedSchema = makeMergedSchema(refSchema, Schema(), targetPrefix="ref_")
    mergedSchema = makeMergedSchema(
        srcSchema, mergedSchema, targetPrefix="src_")
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
    """Generate a list of ReferenceMatches from a Catalog of "unpacked matches"

    @param[in] catalog           catalog of matches.  Must have schema where reference entries are
                                 prefixed with "ref_" and source entries are prefixed with "src_"
    @param[in] sourceSlotConfig  an lsst.meas.base.baseMeasurement.SourceSlotConfig configuration
                                 for source slots (optional)

    @returns   lsst.afw.table.ReferenceMatch of matches
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
