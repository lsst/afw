import os.path
import lsst.afw.table as afwTable
from lsst.utils import getPackageDir

def copySchema(schema, target, targetPrefix=None, sourcePrefix=None):
    """Return a deep copy the provided schema

    If sourcePrefix is set, only copy those keys that have that prefix.
    If targetPrefix is set, add that to the key name.
    """
    existing = set(target.getNames())
    for keyName in schema.getNames():
        if sourcePrefix is not None:
            if not keyName.startswith(sourcePrefix):
                continue
            keyNameFix = keyName[len(sourcePrefix):]
        else:
            keyNameFix = keyName
        if keyNameFix in existing:
            continue
        field = schema.find(keyName).field
        typeStr = field.getTypeString()
        fieldDoc = field.getDoc()
        fieldUnits = field.getUnits()
        if typeStr in ("ArrayF", "ArrayD", "ArrayI", "CovF", "CovD"):
            fieldSize = field.getSize()
        else:
            fieldSize = None

        target.addField((targetPrefix if targetPrefix is not None else "") + keyNameFix,
                        type=typeStr, doc=fieldDoc, units=fieldUnits, size=fieldSize)
    return target

def copyCatalog(catalog, target, sourceSchema=None, targetPrefix=None, sourcePrefix=None):
    """Copy entries from one Catalog to another.

    If sourcePrefix is set, only copy those keys that have that prefix.
    If targetPrefix is set, add that to the key name.
    """
    if sourceSchema is None:
        sourceSchema = catalog.schema

    targetSchema = target.schema
    target.reserve(len(catalog))
    for i in range(len(target), len(catalog)):
        target.addNew()

    if len(catalog) != len(target):
        raise RuntimeError("Length mismatch: %d vs %d" % (len(catalog), len(target)))

    sourceKeys = []
    targetKeys = []
    for k in sourceSchema.getNames():
        if sourcePrefix is not None:
            if not k.startswith(sourcePrefix):
                continue
            kFix = k[len(sourcePrefix):]
        else:
            kFix = k
        sourceKeys.append(sourceSchema.find(k).key)
        targetKeys.append(targetSchema.find((targetPrefix if targetPrefix is not None else "") + kFix).key)

    for rFrom, rTo in zip(catalog, target):
        for kFrom, kTo in zip(sourceKeys, targetKeys):
            try:
                rTo.set(kTo, rFrom.get(kFrom))
            except:
                print "Error setting: %s %s %s %s" % (type(rFrom), type(rTo), type(kFrom), type(kTo))

    return target

def matchesToCatalog(matches, matchMeta):
    """Denormalise matches into a Catalog of "unpacked matches" """

    if len(matches) == 0:
        raise RuntimeError("No matches provided.")

    refSchema = matches[0].first.getSchema()
    srcSchema = matches[0].second.getSchema()

    mergedSchema = copySchema(refSchema, afwTable.Schema(), targetPrefix="ref.")
    mergedSchema = copySchema(srcSchema, mergedSchema, targetPrefix="src.")
    distKey = mergedSchema.addField("distance", type=float, doc="Distance between ref and src")

    mergedCatalog = afwTable.BaseCatalog(mergedSchema)
    copyCatalog([m.first for m in matches], mergedCatalog, sourceSchema=refSchema, targetPrefix="ref.")
    copyCatalog([m.second for m in matches], mergedCatalog, sourceSchema=srcSchema, targetPrefix="src.")
    for m, r in zip(matches, mergedCatalog):
        r.set(distKey, m.distance)

    # obtaining reference catalog name
    catalogName = os.path.basename(getPackageDir("astrometry_net_data"))
    matchMeta.add('REFCAT', catalogName)
    mergedCatalog.getTable().setMetadata(matchMeta)

    return mergedCatalog

def matchesFromCatalog(catalog, sourceSlotConfig=None, prefix=""):
    """Generate a list of ReferenceMatches from a Catalog of "unpacked matches" """
    if catalog is None:
        # There are none
        return []
    refSchema = copySchema(catalog.schema, afwTable.SimpleTable.makeMinimalSchema(), sourcePrefix="ref.")
    refCatalog = afwTable.SimpleCatalog(refSchema)
    copyCatalog(catalog, refCatalog, sourcePrefix="ref.")

    srcSchema = copySchema(catalog.schema, afwTable.SourceTable.makeMinimalSchema(), sourcePrefix="src.")
    srcCatalog = afwTable.SourceCatalog(srcSchema)
    copyCatalog(catalog, srcCatalog, sourcePrefix="src.")

    if sourceSlotConfig is not None:
        sourceSlotConfig.setupTable(srcCatalog.table, prefix=prefix)

    matches = []
    distKey = catalog.schema.find("distance").key
    for ref, src, cat in zip(refCatalog, srcCatalog, catalog):
        matches.append(afwTable.ReferenceMatch(ref, src, cat[distKey]))

    return matches
