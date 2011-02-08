// -*- LSST-C++ -*-
#ifndef LSST_AFW_DETECTION_LOCAL_PSF_H
#define LSST_AFW_DETECTION_LOCAL_PSF_H

#include "boost/shared_ptr.hpp"
#include "lsst/base.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/geom/geom.h"
#include "lsst/ndarray.hpp"

namespace lsst {
namespace afw {
namespace detection {

struct Shapelet {
    enum Definition {LAGUERRE_PQ, HERMITE_XY};

    typedef double Coefficient;
    ndarray::Array<Coefficient, 1, 1> coefficients;    
    lsst::afw::geom::ellipses::BaseCore::Ptr ellipse;
};

typedef std::list<Shapelet> MultiShapelet;

class LocalPsf {
public: 
    typedef boost::shared_ptr<LocalPsf> Ptr;
    typedef boost::shared_ptr<LocalPsf const> ConstPtr;

    typedef Psf::Pixel Pixel;
    typedef Psf::Image Image;

    virtual CONST_PTR(Image) asImage(lsst::afw::geom::Extent2I const & size, bool normalize=true) const =0 ;
    virtual CONST_PTR(lsst::afw::geom::ellipses::BaseCore) asGaussian() const = 0;
    virtual CONST_PTR(Shapelet) asShapelet(Shapelet::Definition definition) const = 0;
    virtual CONST_PTR(MultiShapelet) asMultiShapelet(Shapelet::Definition definition) const = 0;

    virtual ~LocalPsf(){}
};

}}}
