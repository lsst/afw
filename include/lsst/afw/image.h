/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * \file
 * \brief An include file to include the header files for lsst::afw::image
 */
#ifndef LSST_IMAGE_H
#define LSST_IMAGE_H

#include "lsst/afw/geom.h"
#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/image/Filter.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/image/DistortedTanWcs.h"
#include "lsst/afw/image/Exposure.h"    // Exposure.h brings in almost everything
#include "lsst/afw/image/ImageAlgorithm.h"
#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/ImageSlice.h"
#include "lsst/afw/fits.h" // stuff here is forward-declared in headers in afw::image, but
                           // since we need it in SWIG (and that's the only place anyone
                           // should really be including image.h) we include it here.

#endif // LSST_IMAGE_H
