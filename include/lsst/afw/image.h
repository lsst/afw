/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * An include file to include the header files for lsst::afw::image
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
#include "lsst/afw/image/Exposure.h"  // Exposure.h brings in almost everything
#include "lsst/afw/image/ImageAlgorithm.h"
#include "lsst/afw/image/ImagePca.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/ImageSlice.h"
#include "lsst/afw/fits.h" /* stuff here is forward-declared in headers in afw::image, but
                            * since we need it in SWIG (and that's the only place anyone
                            * should really be including image.h) we include it here.
                            */

#endif  // LSST_IMAGE_H
