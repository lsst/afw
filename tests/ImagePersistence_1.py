#!/usr/bin/env python

import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy

# Create the additionalData DataProperty
additionalData = dafBase.DataProperty.createPropertyNode("root")
additionalData.addProperty(dafBase.DataProperty("sliceId", 0))
additionalData.addProperty(dafBase.DataProperty("visitId", "fov391"))
additionalData.addProperty(dafBase.DataProperty("universeSize", 100))
additionalData.addProperty(dafBase.DataProperty("itemName", "foo"))

# Create an empty Policy
policy = pexPolicy.PolicyPtr()

# Get a Persistence object
persistence = dafPers.Persistence.getPersistence(policy)

# Set up the LogicalLocation.  Assumes that previous tests have run, and
# Src_*.fits exists in the current directory.
logicalLocation = dafPers.LogicalLocation("Src_img.fits")

# Create a FitsStorage and put it in a StorageList.
storage = persistence.getRetrieveStorage("FitsStorage", logicalLocation)
storageList = dafPers.StorageList([storage])

# Let's do the retrieval!
persPtr = persistence.unsafeRetrieve("ImageF", storageList, additionalData)
image = afwImage.ImageF.swigConvert(persPtr)

# Check the resulting Image
# ...

# Persist the Image (under a different name, and in a different format)
logicalLocation = dafPers.LogicalLocation("image.boost")
storage = persistence.getPersistStorage("BoostStorage", logicalLocation)
storageList = dafPers.StorageList([storage])
persistence.persist(image, storageList, additionalData)

# Retrieve it again
storage = persistence.getRetrieveStorage("BoostStorage", logicalLocation)
storageList = dafPers.StorageList([storage])
pers2Ptr = persistence.unsafeRetrieve("ImageF", storageList, additionalData)
image2 = afwImage.ImageF.swigConvert(pers2Ptr)

# Check to make sure that we got the same data
assert image.getRows() == image2.getRows()
assert image.getCols() == image2.getCols()
assert image.getOffsetRows() == image2.getOffsetRows()
assert image.getOffsetCols() == image2.getOffsetCols()
for c in xrange(image.getCols()):
    for r in xrange(image.getRows()):
        pixel1 = image.getVal(c, r)
        pixel2 = image2.getVal(c, r)
        # Persisting through Boost text archives causes conversion error!
        # assert abs(pixel1 - pixel2) / pixel1 < 1e-7, \
        assert pixel1 == pixel2, \
                "Differing pixel2 at %d, %d: %f, %f" % (c, r, pixel1, pixel2)
