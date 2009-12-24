// -*- lsst-c++ -*-
/**
 *  \file
 *  \brief Conversions between afw::image:: and afw::geom:: Point objects.
 *
 *  Will be removed once afw::image::Point is no longer in use.
 */
#ifndef LSST_AFW_GEOM_DEPRECATED_H
#define LSST_AFW_GEOM_DEPRECATED_H

#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"

namespace lsst { namespace afw { namespace geom {

template <typename T>
inline geom::Point<T,2> convertToGeom(image::Point<T> const & other) {
    return geom::Point<T,2>::makeXY(other.getX(),other.getY());
}

template <typename T>
inline image::Point<T> convertToImage(geom::Point<T,2> const & other) {
    return image::Point<T>(other.getX(),other.getY());
}

}}}

#endif
