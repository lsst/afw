#!/usr/bin/env python

import lsst.afw.image as afwImage
import lsst.mwi.data as DATA
import lsst.mwi.persistence as PERS
import lsst.mwi.policy as POL
import os

# Create the additionalData DataProperty
additionalData = DATA.SupportFactory.createPropertyNode("root")
additionalData.addProperty(DATA.DataProperty("sliceId", 0))
additionalData.addProperty(DATA.DataProperty("visitId", "fov391"))
additionalData.addProperty(DATA.DataProperty("universeSize", 100))
additionalData.addProperty(DATA.DataProperty("itemName", "foo"))

# Create an empty Policy
policy = POL.PolicyPtr()

# Get a Persistence object
persistence = PERS.Persistence.getPersistence(policy)

# Set up the LogicalLocation.  Assumes that previous tests have run, and
# Src_*.fits exists in the current directory.
logicalLocation = PERS.LogicalLocation("Src")

# Create a FitsStorage and put it in a StorageList.
storage = persistence.getRetrieveStorage("FitsStorage", logicalLocation)
storageList = PERS.StorageList([storage])

print "Retrieving MaskedImage Src"

# Let's do the retrieval!
maskedImage = afwImage.MaskedImageF.swigConvert( \
    persistence.unsafeRetrieve("MaskedImageF", storageList, additionalData))

# Check the resulting MaskedImage
# ...

print "Persisting MaskedImage as FITS to Dest"

# Persist the MaskedImage (under a different name)
logicalLocation = PERS.LogicalLocation("Dest")
storage = persistence.getPersistStorage("FitsStorage", logicalLocation)
storageList = PERS.StorageList([storage])
try:
    persistence.persist(maskedImage, storageList, additionalData)
except POL.LsstInvalidParameter, e:
    print e.what()
    raise

# Ideally should do a cmp to make sure they are the same, but the persistence
# writes additional comments not present in the original file.
# assert os.system("cmp Src_img.fits Dest_img.fits") == 0
# assert os.system("cmp Src_msk.fits Dest_msk.fits") == 0
# assert os.system("cmp Src_var.fits Dest_var.fits") == 0
