// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include "testSpatialCell.h"

/**
 * Constructor
 */
ExampleCandidate::ExampleCandidate(
        float const xCenter,            ///< The object's column-centre
        float const yCenter,            ///< The object's row-centre
        ExampleCandidate::MaskedImageT::ConstPtr parent, ///< the parent image
        lsst::afw::geom::Box2I bbox     ///< The object's bounding box
                                      ) :
    lsst::afw::math::SpatialCellMaskedImageCandidate<PixelT>(xCenter, yCenter), _parent(parent), _bbox(bbox) {
}

/**
 * Return candidates rating
 */
double ExampleCandidate::getCandidateRating() const {
    return (*_parent->getImage())(getXCenter(), getYCenter());
}

/**
 * Return the %image
 */
ExampleCandidate::MaskedImageT::ConstPtr ExampleCandidate::getMaskedImage() const {
    if (_image.get() == NULL) {
        _image = MaskedImageT::Ptr(new MaskedImageT(*_parent, _bbox, lsst::afw::image::LOCAL));
    }
    
    return _image;
}
