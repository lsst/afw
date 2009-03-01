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

%inline %{

    /*
     * Test class for SpatialCell
     */
    class TestCandidate : public lsst::afw::math::SpatialCellCandidate {
    public:
        TestCandidate(float const xCenter, ///< The object's column-centrew
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
%}


