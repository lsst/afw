// -*- lsst-c++ -*-
#include "testSpatialCell.h"

/**
 * Constructor
 */
ExampleCandidate::ExampleCandidate(
        float const xCenter,            ///< The object's column-centre
        float const yCenter,            ///< The object's row-centre
        ExampleCandidate::ImageT::ConstPtr parent, ///< the parent image
        lsst::afw::image::BBox bbox     ///< The object's bounding box
                                      ) :
    lsst::afw::math::SpatialCellImageCandidate<ImageT>(xCenter, yCenter), _parent(parent), _bbox(bbox) {
}

/**
 * Return candidates rating
 */
double ExampleCandidate::getCandidateRating() const {
    return (*_parent)(getXCenter(), getYCenter());
}

/**
 * Return the %image
 */
ExampleCandidate::ImageT::ConstPtr ExampleCandidate::getImage() const {
    if (_image.get() == NULL) {
        _image = ImageT::Ptr(new ImageT(*_parent, _bbox));
    }
    
    return _image;
}
