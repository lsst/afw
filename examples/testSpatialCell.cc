// -*- lsst-c++ -*-

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
