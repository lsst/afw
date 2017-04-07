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

ExampleCandidate::ExampleCandidate(
        float const xCenter,
        float const yCenter,
        std::shared_ptr<MaskedImageT const> parent,
        lsst::afw::geom::Box2I bbox
                                      ) :
    lsst::afw::math::SpatialCellImageCandidate(xCenter, yCenter), _parent(parent), _bbox(bbox) {
}

double ExampleCandidate::getCandidateRating() const {
    return (*_parent->getImage())(getXCenter(), getYCenter());
}

std::shared_ptr<ExampleCandidate::MaskedImageT const> ExampleCandidate::getMaskedImage() const {
    if (!_image) {
        _image = std::make_shared<MaskedImageT>(*_parent, _bbox, lsst::afw::image::LOCAL);
    }
    return _image;
}
