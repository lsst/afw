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
 
#if !defined(LSST_DETECTION_FOOTPRINT_FUNCTOR_H)
#define LSST_DETECTION_FOOTPRINT_FUNCTOR_H

#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw { 
namespace detection {
/************************************************************************************************************/
/**
 * \brief A functor class to allow users to process all the pixels in a Footprint
 *
 * There's an annotated example of a FootprintFunctor in action
 * \link FootprintFunctorsExample FootprintFunctors here\endlink
 */
template <typename ImageT>
class FootprintFunctor {
public:
    FootprintFunctor(ImageT const& image    ///< The image that the Footprint lives in
                    ) : _image(image) {}

    virtual ~FootprintFunctor() = 0;

    /**
     * A function that's called at the beginning of apply; useful if apply
     * calculates a per-footprint quantity
     */
    virtual void reset() {}
    virtual void reset(Footprint const&) {}

    /**
     * \brief Apply operator() to each pixel in the Footprint
     */
    void apply(Footprint const& foot,   ///< The Footprint in question
               int const margin=0       ///< The required margin from the edge of the image
              ) {
        reset();
        reset(foot);

        if (foot.getSpans().empty()) {
            return;
        }

        geom::Box2I const bbox = foot.getBBox();
        geom::Box2I region = foot.getRegion();
        if (!region.isEmpty() &&
            (!region.contains(bbox.getMin()) || !region.contains(bbox.getMax()))) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                (boost::format("Footprint with BBox (%d,%d) -- (%dx%d)"
                               "doesn't fit in image with BBox (%d,%d) -- (%dx%d)"
                               ) % bbox.getMinX() % bbox.getMinY() 
                               % bbox.getMaxX() % bbox.getMaxY() 
                               % region.getMinX() % region.getMinY() 
                               % region.getMaxX() % region.getMaxY()
                ).str()
            );
        }

        // Current position of the locator (in the SpanList loop)
        int ox1 = 0, oy = 0;            
        
        typename ImageT::xy_locator loc = _image.xy_at(
            -_image.getX0(), -_image.getY0()
        ); // Origin of the Image's pixels

        int const width = _image.getWidth();
        int const height = _image.getHeight();
        for (Footprint::SpanList::const_iterator siter = foot.getSpans().begin();
             siter != foot.getSpans().end(); siter++) {
            Span::Ptr const span = *siter;

            int const y = span->getY();
            if (y < margin || y >= height - margin) {
                continue;
            }
            int x0 = span->getX0();
            int x1 = span->getX1();
            if (x0 < margin) {
                x0 = margin;
            }
            if (x1 >= width - margin) {
                x1 = width - margin - 1;
            }

            loc += lsst::afw::image::pair2I(x0 - ox1, y - oy);

            for (int x = x0; x <= x1; ++x, ++loc.x()) {
                operator()(loc, x, y);
            }

            ox1 = x1 + 1; oy = y;
        }
    }
    /// Return the image
    ImageT const& getImage() const { return _image; }    

    /// The operator to be applied to each pixel in the Footprint.
    ///
    /// N.b. the coordinates (x, y) are relative to the origin of the image's parent
    /// if it exists (i.e. they obey getX0/getY0)
    virtual void operator()(typename ImageT::xy_locator loc, int x, int y) = 0;
private:
    ImageT const& _image;               // The image that the Footprints live in
};

///
/// Although FootprintFunctor is pure virtual, this is needed by subclasses
///
/// It wasn't defined in the class body as I want swig to know that the class is pure virtual
///
template <typename ImageT>
FootprintFunctor<ImageT>::~FootprintFunctor() {}
            
}}}
#endif
