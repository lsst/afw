// -*- LSST-C++ -*-

#ifndef LSST_AFW_IMAGE_MAKEWCS_H
#define LSST_AFW_IMAGE_MAKEWCS_H


#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"

namespace lsst {
namespace afw {
namespace image {

Wcs::Ptr makeWcs(lsst::daf::base::PropertySet::Ptr fitsMetadata);

}}}

#endif
