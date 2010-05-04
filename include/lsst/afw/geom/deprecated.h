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

/** 
    \page image2geom Transitioning from image::Point and image::BBox to geom

    The convertToGeom() and convertToImage() functions declared in deprecated.h provide conversions
    between geometry objects defined in image and their counterparts in geom.

    In cases where std::pair was used in place of Extent in the past,
    @link CoordinateBase<Derived,T,2>::asPair Extent::asPair @endlink 
    and 
    @link CoordinateBase<Derived,T,2>::make Extent::make @endlink 
    may be useful.

    \section point image::Point to geom::Point
    
    In the table below,
    - x and y are variables of type T
    - v is a two-element array of T
    - ip1 and ip2 are objects of type image::Point<T>
    - gp1 and gp2 are objects of type geom::Point<T>
    
    image::Point operations not addressed below have not changed in geom::Point.
    
    <table>
    <tr> <td></td> <th>image</th> <th>geom</th> <th>Notes</th> </tr>
    <tr> 
      <th rowspan=3>Types</th> 
      <td>@code image::Point<T> @endcode</td>
      <td>@code geom::Point<T> @endcode</td> 
      <td>The second template parameter of geom::Point defaults to 2.</td>
    </tr>
    <tr>
      <td rowspan=2>@code image::PointD @endcode</td>
      <td>@code geom::PointD @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code geom::Point2D @endcode</td>
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th rowspan=2>Construction</th> 
      <td>@code image::Point<T>(x,y) @endcode</td>
      <td>@code geom::Point<T>(x,y) @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::Point<T>(v) @endcode</td>
      <td>@code geom::Point<T>(v) @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th rowspan=2>Comparison</th> 
      <td>@code ip1 == ip2 @endcode</td>
      <td>@code all(gp1 == gp2) @endcode</td> 
      <td rowspan=2>Comparison operators return a 2-element CoordinateExpr object, not a scalar bool.</td>
    </tr>
    <tr>
      <td>@code ip1 != ip2 @endcode</td>
      <td>@code any(gp1 != gp2) @endcode</td> 
    </tr>
    <tr>
      <th rowspan=4>Operators</th> 
      <td rowspan=3>@code ip1 + ip2 @endcode</td>
      <td>@code gp1 + Extent<T>(gp2) @endcode</td>
      <td rowspan=3>
       Two objects of type geom::Point cannot be added, but Point can be added to Extent, and
       Extent can be added to Extent.
      </td>
    </tr>
    <tr>
      <td>@code Extent<T>(gp1) + gp2 @endcode</td>
    </tr>
    <tr>
      <td>@code Point<T>(Extent<T>(gp1) + Extent<T>(gp2)) @endcode</td>
    </tr>
    <tr>
      <td>@code ip1 - ip2 @endcode</td>
      <td>@code Point<T>(gp1 - gp2) @endcode </td> 
      <td>The return type of geom::Point subtraction is geom::Extent, not geom::Point.</td>
    </tr>
    </table>

    \section box image::BBox to geom::BoxI

    Unlike geom::BoxD, both image::BBox and geom::BoxI are inclusive; the
    minimum and maximum points are contained by the box.  However, image::BBox
    involves no special handling for empty boxes; an image::BBox may have zero
    or negative size in either or both dimensions, and while this will produce
    sensible results in containment tests, it may produce unexpected results in
    other cases.  geom::BoxI treats all empty boxes as the same, and does not
    permit boxes that are empty in only one dimension.

    geom::BoxI does not allow corners or edges to be set
    individually; this avoids some empty-box-semantics headaches.
    
    <table>
    <tr> <td></td> <th>image</th> <th>geom</th> <th>Notes</th> </tr>
    <tr> 
      <th>Types</th> 
      <td>@code image::BBox @endcode</td>
      <td>@code geom::BoxI @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th rowspan=3>Construction</th> 
      <td>
@code
image::BBox::BBox(image::PointI llc, 
                  int width, 
                  int height)
@endcode
      </td>
      <td>
@code
geom::BoxI::BoxI(geom::Point2I min, 
                 geom::ExtentI dimensions, 
                 bool invert=true)
@endcode
      </td> 
      <td rowspan=2>
        image::BBox's "LLC" and "URC" have generally been replaced by "min" and "max" 
        in geom::BoxI.  The BoxI constructor also has an optional "invert" parameter that specifies
        how negative dimensions should be handled; the default is to interpret negative dimensions
        as indicating the minimum value is actually a maximum (passing negative dimensions to
        the image::BBox constructor produces a box that is empty in one or both dimensions).
      </td>
    </tr>
    <tr>
      <td>
@code
image::BBox::BBox(image::PointI llc, 
                  image::PointI urc)
@endcode
     </td>
      <td>
@code
geom::BoxI::BoxI(geom::PointI min, 
                 geom::PointI max, 
                 bool invert=true)
@endcode
    </td> 
    </tr>
    <tr>
      <td>@code image::BBox::BBox() @endcode</td>
      <td>@code geom::BoxI::BoxI() @endcode</td> 
      <td>
        Both default constructors produce an empty box, though only with geom::BoxI will this
        box be equivalent to any other empty box.
      </td>
    </tr>
    <tr>
      <th rowspan=9>Accessors</th> 
      <td>@code image::BBox::getX0() @endcode</td>
      <td>@code geom::BoxI::getMinX() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getX1() @endcode</td>
      <td>@code geom::BoxI::getMaxX() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getY0() @endcode</td>
      <td>@code geom::BoxI::getMinY() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getY1() @endcode</td>
      <td>@code geom::BoxI::getMaxY() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getLLC() @endcode</td>
      <td>@code geom::BoxI::getMin() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getURC() @endcode</td>
      <td>@code geom::BoxI::getMax() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getWidth() @endcode</td>
      <td>@code geom::BoxI::getWidth() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getHeight() @endcode</td>
      <td>@code geom::BoxI::getHeight() @endcode</td> 
      <td>&nbsp;</td>
    </tr>
    <tr>
      <td>@code image::BBox::getDimensions() @endcode</td>
      <td>@code geom::BoxI::getDimensions() @endcode</td> 
      <td>geom::BoxI returns geom::ExtentI, while image::BBox return std::pair<int,int></td>
    </tr>
    <tr>
      <th rowspan=3>Comparison</th> 
      <td>@code image::BBox::operator==(image::BBox) @endcode</td>
      <td>@code geom::BoxI::operator==(geom::BoxI) @endcode</td> 
      <td rowspan=2>Comparison operators are the same, but in geom all empty boxes are considered equal.</td>
    </tr>
    <tr>
      <td>@code image::BBox::operator!=(image::BBox) @endcode</td>
      <td>@code geom::BoxI::operator!=(geom::BoxI) @endcode</td> 
    </tr>
    <tr>
      <td>@code image::BBox::operator bool() @endcode</td>
      <td>@code geom::BoxI::isEmpty() @endcode</td> 
      <td>
        geom::BoxI is not implicitly converible to bool.  Note that an image::BBox that is
        empty in only one dimension evaluates to true.
      </td>
    </tr>
    <tr>
      <th rowspan=4>Spatial Relations</th> 
      <td>@code image::BBox::contains(image::PointI) @endcode</td>
      <td>@code geom::BoxI::contains(geom::Point2I) @endcode</td>
      <td>
        Containment of points is identical (aside from the data type switch 
        from image::PointI to geom::Point2I).
      </td>
    </tr>
    <tr>
      <td>@code image::BBox::grow(image::PointI) @endcode</td>
      <td>@code geom::BoxI::include(geom::Point2I) @endcode</td>
      <td>
        These expand the box to include the given point.  Behavior is identical,
        except for point data type and a difference in handling boxes with only one
        empty dimension by image::BBox (a situation that is impossible with geom::BoxI).
        <br>
        <em>Note that geom::BoxI::grow is unrelated to image::BBox::grow.</em>
      </td>
    </tr>
    <tr>
      <td>@code image::BBox::shift(int x, int y) @endcode</td>
      <td>@code geom::BoxI::shift(geom::ExtentI) @endcode</td>
      <td>
        Translation is identical for non-empty boxes aside from the signature.  Shifting
        an image::BBox moves the box, while shifting a geom::BoxI is a no-op.
      </td>
    </tr>
    <tr>
      <td>@code image::BBox::clip(image::BBox) @endcode</td>
      <td>@code geom::BoxI::clip(geom::BoxI) @endcode</td>
      <td>
        Box intersection is identical for non-empty boxes, and if an empty box is involved,
        the result will be empty in both cases (though image::BBox may allow it to be empty
        in only one dimension).
      </td>
    </tr>
    </table>

*/

/// \brief Convert an image::Point object to the equivalent geom::Point object.
template <typename T>
inline geom::Point<T,2> convertToGeom(image::Point<T> const & other) {
    return geom::Point<T,2>(other.getX(),other.getY());
}

/// \brief Convert a geom::Point object to the equivalent image::Point object.
template <typename T>
inline image::Point<T> convertToImage(geom::Point<T,2> const & other) {
    return image::Point<T>(other.getX(),other.getY());
}

/// \brief Convert an image::BBox object to the equivalent geom::BoxI object.
inline geom::BoxI convertToGeom(image::BBox const & other) {
    return geom::BoxI(convertToGeom(other.getLLC()), convertToGeom(other.getURC()), false);
}

/// \brief Convert a geom::BoxI object to the equivalent image::BBox object.
inline image::BBox convertToImage(geom::BoxI const & other) {
    return image::BBox(convertToImage(other.getMin()), convertToImage(other.getMax()));
}

}}}

#endif
