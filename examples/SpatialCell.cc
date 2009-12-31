// -*- lsst-c++ -*-
#include "SpatialCell.h"

/**
 * constructor
 */
TestCandidate::TestCandidate(
        float const xCenter, ///< The object's column-centre
        float const yCenter, ///< The object's row-centre
        float const flux     ///< The object's flux
                            ) :
    SpatialCellCandidate(xCenter, yCenter), _flux(flux) {
}

/**
 * Return candidates rating
 */
double TestCandidate::getCandidateRating() const {
    return _flux;
}

/************************************************************************************************************/
/**
 * Constructor
 */
TestImageCandidate::TestImageCandidate(
        float const xCenter,            ///< The object's column-centre
        float const yCenter,            ///< The object's row-centre
        float const flux                ///< The object's flux
                                      ) :
    lsst::afw::math::SpatialCellImageCandidate<ImageT>(xCenter, yCenter), _flux(flux) {
}

/**
 * Return candidates rating
 */
double TestImageCandidate::getCandidateRating() const {
    return _flux;
}

/**
 * Return the %image
 */
TestImageCandidate::ImageT::ConstPtr TestImageCandidate::getImage() const {
    if (_image.get() == NULL) {
        _image = ImageT::Ptr(new ImageT(getWidth(), getHeight()));
        *_image = _flux;
    }
    
    return _image;
}
