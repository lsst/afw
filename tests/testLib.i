// -*- lsst-c++ -*-
%define testLib_DOCSTRING
"
Various swigged-up C++ classes for testing
"
%enddef

%feature("autodoc", "1");
%module(package="testLib", docstring=testLib_DOCSTRING) testLib

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

%include "lsst/p_lsstSwig.i"

%lsst_exceptions()

%{
#include "lsst/afw/math.h"
%}

%import "lsst/afw/math/mathLib.i"

SWIG_SHARED_PTR_DERIVED(TestCandidate, lsst::afw::math::SpatialCellCandidate, TestCandidate);
SWIG_SHARED_PTR_DERIVED(TestImageCandidate,
                        lsst::afw::math::SpatialCellImageCandidate<lsst::afw::image::Image<float> >,
                        TestImageCandidate);

%inline %{
    /*
     * Test class for SpatialCellCandidate
     */
    class TestCandidate : public lsst::afw::math::SpatialCellCandidate {
    public:
        TestCandidate(float const xCenter, ///< The object's column-centre
                      float const yCenter, ///< The object's row-centre
                      float const flux     ///< The object's flux
                    ) :
            SpatialCellCandidate(xCenter, yCenter), _flux(flux) {
        }

        /// Return candidates rating
        double getCandidateRating() const {
            return _flux;
        }
    private:
        double _flux;
    };

    /************************************************************************************************************/
    /*
     * Test class for SpatialCellImageCandidate
     */
    class TestImageCandidate : public lsst::afw::math::SpatialCellImageCandidate<lsst::afw::image::Image<float> > {
    public:
        typedef lsst::afw::image::Image<float> ImageT;

        TestImageCandidate(float const xCenter, ///< The object's column-centre
                           float const yCenter, ///< The object's row-centre
                           float const flux     ///< The object's flux
                    ) :
            lsst::afw::math::SpatialCellImageCandidate<ImageT>(xCenter, yCenter), _flux(flux) {
        }

        /// Return candidates rating
        double getCandidateRating() const {
            return _flux;
        }

        /// Return the %image
        ImageT::ConstPtr getImage() const {
            if (_image.get() == NULL) {
                _image = ImageT::Ptr(new ImageT(11, 11));
                *_image = _flux;
            }

            return _image;
        }
    private:
        double _flux;
    };
%}
