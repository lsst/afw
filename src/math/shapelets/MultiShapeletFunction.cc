// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

#include "lsst/afw/math/shapelets/MultiShapeletFunction.h"
#include "lsst/afw/math/shapelets/ConversionMatrix.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray/eigen.h"
#include <boost/format.hpp>

namespace shapelets = lsst::afw::math::shapelets;
namespace nd = lsst::ndarray;

void shapelets::MultiShapeletFunction::convolve(shapelets::ShapeletFunction const & other) {
    for (ElementList::iterator i = _elements.begin(); i != _elements.end(); ++i) {
        i->convolve(other);
    }
}

void shapelets::MultiShapeletFunction::convolve(shapelets::MultiShapeletFunction const & other) {
    ElementList newElements;
    for (ElementList::const_iterator j = other.getElements().begin(); j != other.getElements().end(); ++j) {
        for (ElementList::iterator i = _elements.begin(); i != _elements.end(); ++i) {
            newElements.push_back(*i);
            newElements.back().convolve(*j);
            
        }
    }
    newElements.swap(_elements);
}

void shapelets::MultiShapeletFunctionEvaluator::update(shapelets::MultiShapeletFunction const & function) {
    _elements.clear();
    for (
        MultiShapeletFunction::ElementList::const_iterator i = function.getElements().begin(); 
        i != function.getElements().end();
        ++i
    ) {
        _elements.push_back(i->evaluate());
    }
}

shapelets::Pixel shapelets::MultiShapeletFunctionEvaluator::operator()(geom::Point2D const & point) const {
    Pixel r = 0.0;
    for (ElementList::const_iterator i = _elements.begin(); i != _elements.end(); ++i) {
        r += (*i)(point);
    }
    return r;
}

shapelets::Pixel shapelets::MultiShapeletFunctionEvaluator::operator()(geom::Extent2D const & point) const {
    Pixel r = 0.0;
    for (ElementList::const_iterator i = _elements.begin(); i != _elements.end(); ++i) {
        r += (*i)(point);
    }
    return r;
}


shapelets::Pixel shapelets::MultiShapeletFunctionEvaluator::integrate() const {
    Pixel r = 0.0;
    for (ElementList::const_iterator i = _elements.begin(); i != _elements.end(); ++i) {
        r += i->integrate();
    }
    return r;
}

shapelets::MultiShapeletFunctionEvaluator::MultiShapeletFunctionEvaluator(
    shapelets::MultiShapeletFunction const & function
) {
    update(function);
}
