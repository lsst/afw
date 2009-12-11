// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_H
#define LSST_AFW_GEOM_ELLIPSES_H

/**
 *  \file
 *  \brief Public header class for ellipse library.
 */

/// \defgroup EllipseGroup Ellipses

#include "lsst/afw/geom/ellipses/BaseEllipse.h"
#include "lsst/afw/geom/ellipses/EllipseImpl.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"
#include "lsst/afw/geom/ellipses/RadialFraction.h"
#include "lsst/afw/geom/ellipses/Moments.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/Parametric.h"

namespace lsst {
namespace afw {
namespace geom {
using ellipses::Ellipse;
}

#endif // !LSST_AFW_GEOM_ELLIPSES_H
