"""pyfic - A thin wrapper around PyFITS for working with
FITS Image Collections (FIC).
Contact David Fleming at dfleming@ncsa.uiuc.edu with questions.
"""

import pyfits

def open(ficName, mode = None):
    """Open the FITS Image Collection (FIC)."""
    (collection, target) = _parseFicName(ficName)
    if target is None:
        # Take this to mean the first target by default.
        target = 1
    targetFileName = _getTargetFileName(collection, target)
    if mode is None:
        return pyfits.open(targetFileName)
    else:
        return pyfits.open(targetFileName, mode)

def getProperties(collection):
    """Retrieve the properties dictionary of a FIC."""
    properName = _getCollectionProperName(collection)
    propertiesFileName = '%s/properties' % properName
    f = file(propertiesFileName, 'r')
    properties = {}
    for line in f:
        s = line.strip()
        if s == '':
            continue
        keyValue = s.split('=', 2)
        key = keyValue[0].strip()
        if len(keyValue) == 2:
            value = keyValue[1].strip()
            properties[key] = value
        else:
            properties[key] = True
    f.close()
    return properties

def setProperties(collection, properties):
    """Commit the properties dictionary of a FIC."""
    properName = _getCollectionProperName(collection)
    propertiesFileName = '%s/properties' % properName
    f = file(propertiesFileName, 'w')
    for k, v in properties.items():
        f.write('%s = %s\n' % (k, v))
    f.close()

def _parseFicName(ficName):
    """Parse a FIC name into collection and target parts."""
    parts = ficName.split('[', 2)
    collection = parts[0]
    if len(parts) == 2:
        # A target was specified.
        try: target = int(parts[1][:-1])
        except: raise 'FIC target must be an integer'
    else:
        # No target was given; only the collection name.
        target = None
    return (collection, target)

def _getTargetFileName(collection, target):
    """Construct the underlying FITS name from a FIC collection and target."""
    properName = _getCollectionProperName(collection)
    return '%s/target-%s.fits' % (properName, target)

def _getCollectionProperName(collection):
    """Construct the full proper name of a FIC collection."""
    if collection[-4:] == '.fic':
        return collection
    if collection[-5:] == '.fic/':
        return collection[:-1]
    return collection + '.fic'
