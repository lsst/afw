// -*- LSST-C++ -*-
#ifndef LSST_AFW_DETECTION_LOCAL_PSF_H
#define LSST_AFW_DETECTION_LOCAL_PSF_H

#include "boost/shared_ptr.hpp"
#include "lsst/base.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/shapelets.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/ndarray.h"

namespace lsst {
namespace afw {
namespace detection {

class LocalPsf {
public: 
    typedef boost::shared_ptr<LocalPsf> Ptr;
    typedef boost::shared_ptr<LocalPsf const> ConstPtr;

    typedef lsst::afw::math::shapelets::ShapeletFunction Shapelet;
    typedef std::list<Shapelet> MultiShapelet;

    typedef Psf::Pixel Pixel;
    typedef Psf::Image Image;

    virtual CONST_PTR(Image) asImage(geom::Extent2I const & size, bool normalize=true) const =0 ;
    virtual CONST_PTR(geom::ellipses::BaseCore) asGaussian() const = 0;
    virtual Shapelet asShapelet(math::shapelets::BasisTypeEnum basisType) const = 0;
    virtual MultiShapelet asMultiShapelet(math::shapelets::BasisTypeEnum basisType) const = 0;
    
    virtual void evaluatePointSource(
        Footprint const & fp, 
        geom::Point2D const & point, 
        ndarray::Array<Pixel, 1, 0> const & array
    ) const = 0;
    
    ndarray::Array<Pixel, 1,1> evaluatePointSource(
        Footprint const & fp, 
        geom::Point2D const & point
    ) const {
        ndarray::Array<Pixel, 1, 1> array = ndarray::allocate(ndarray::makeVector(fp.getArea()));
        evaluatePointSource(fp, point, array);
        return array;
    }

    virtual ~LocalPsf(){}
};

}}}

#endif
