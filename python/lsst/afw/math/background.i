
%{
#include "lsst/afw/math/Background.h"
%}

%include "lsst/afw/math/Background.h"

%template(BackgroundD) lsst::afw::math::Background::Background<lsst::afw::image::Image<double> >;
%template(getImageD) lsst::afw::math::Background::getImage<double>;

%template(BackgroundF) lsst::afw::math::Background::Background<lsst::afw::image::Image<float> >;
%template(getImageF) lsst::afw::math::Background::getImage<float>;

%template(BackgroundI) lsst::afw::math::Background::Background<lsst::afw::image::Image<int> >;
%template(getImageI) lsst::afw::math::Background::getImage<int>;
