//
// LSST Data Management System
// Copyright 2015-2016 LSST/AURA
//
// This product includes software developed by the
// LSST Project (http://www.lsst.org/).
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the LSST License Statement and
// the GNU General Public License along with this program.  If not,
// see <https://www.lsstcorp.org/LegalNotices/>.
//

/*
 * Support RGB and grey-scale images
 */
#if !defined(LSST_AFW_DISPLAY_RGB_H)
#define LSST_AFW_DISPLAY_RGB_H 1

namespace lsst {
namespace afw {
namespace display {

template <typename ImageT>
void replaceSaturatedPixels(
        ImageT& rim,                       //< R image (e.g. i)
        ImageT& gim,                       //< G image (e.g. r)
        ImageT& bim,                       //< B image (e.g. g)
        int borderWidth = 2,               //< width of border used to estimate colour of saturated regions
        float saturatedPixelValue = 65535  //< the brightness of a saturated pixel, once fixed
        );

/**
 * Calculate an IRAF/ds9-style zscaling.
 *
 * To quote Frank Valdes (http://iraf.net/forum/viewtopic.php?showtopic=134139)
 * <blockquote>
ZSCALE ALGORITHM

    The zscale algorithm is designed to display the  image  values  near
    the  median  image  value  without  the  time  consuming  process of
    computing a full image histogram.  This is particularly  useful  for
    astronomical  images  which  generally  have a very peaked histogram
    corresponding to  the  background  sky  in  direct  imaging  or  the
    continuum in a two dimensional spectrum.

    The  sample  of pixels, specified by values greater than zero in the
    sample mask zmask or by an  image  section,  is  selected  up  to  a
    maximum  of nsample pixels.  If a bad pixel mask is specified by the
    bpmask parameter then any pixels with mask values which are  greater
    than  zero  are not counted in the sample.  Only the first pixels up
    to the limit are selected where the order is by line beginning  from
    the  first line.  If no mask is specified then a grid of pixels with
    even spacing along lines and columns that  make  up  a  number  less
    than or equal to the maximum sample size is used.

    If  a  contrast of zero is specified (or the zrange flag is used and
    the image does not have a  valid  minimum/maximum  value)  then  the
    minimum  and maximum of the sample is used for the intensity mapping
    range.

    If the contrast  is  not  zero  the  sample  pixels  are  ranked  in
    brightness  to  form  the  function  I(i) where i is the rank of the
    pixel and I is its value.  Generally the midpoint of  this  function
    (the  median) is very near the peak of the image histogram and there
    is a well defined slope about the midpoint which is related  to  the
    width  of the histogram.  At the ends of the I(i) function there are
    a few very bright and dark pixels due to objects and defects in  the
    field.   To  determine  the  slope  a  linear  function  is fit with
    iterative rejection;

    <code>
            I(i) = intercept + slope * (i - midpoint)
    </code>

    If more than half of the points are rejected then there is  no  well
    defined  slope  and  the full range of the sample defines z1 and z2.
    Otherwise the endpoints of the linear function  are  used  (provided
    they are within the original range of the sample):

    <code>
            z1 = I(midpoint) + (slope / contrast) * (1 - midpoint)
            z2 = I(midpoint) + (slope / contrast) * (npoints - midpoint)
    </code>

    As  can  be  seen,  the parameter contrast may be used to adjust the
    contrast produced by this algorithm.
 * </blockquote>
 */
template <class T>
std::pair<double, double> getZScale(image::Image<T> const& image,  ///< The image we wish to stretch
                                    int const nSamples = 1000,     ///< Number of samples to use
                                    double const contrast = 0.25   ///< Stretch parameter; see description
                                    );
}
}
}

#endif
