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
#include "lsst/log/Log.h"
#include "lsst/pex/policy.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/cameraGeom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math/SpatialCell.h"
%}

%import "lsst/afw/math/mathLib.i"

%shared_ptr(TestCandidate);
%shared_ptr(TestCandidateVisitor);
%shared_ptr(TestMaskedImageCandidate);

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
        virtual double getCandidateRating() const {
            return _flux;
        }
        virtual void setCandidateRating(double flux) {
            _flux = flux;
        }
    private:
        double _flux;
    };

    /// A class to pass around to all our TestCandidates
    class TestCandidateVisitor : public lsst::afw::math::CandidateVisitor {
    public:
        TestCandidateVisitor() : lsst::afw::math::CandidateVisitor(), _n(0) {}

        // Called by SpatialCellSet::visitCandidates before visiting any Candidates
        void reset() { _n = 0; }

        // Called by SpatialCellSet::visitCandidates for each Candidate
        void processCandidate(lsst::afw::math::SpatialCellCandidate *candidate) {
            ++_n;
        }

        int getN() const { return _n; }
    private:
        int _n;                         // number of TestCandidates
    };

    /************************************************************************************************************/
    /*
     * Test class for SpatialCellMaskedImageCandidate
     */
    class TestMaskedImageCandidate : public lsst::afw::math::SpatialCellMaskedImageCandidate<float> {
    public:
        typedef lsst::afw::image::MaskedImage<float> MaskedImageT;

        TestMaskedImageCandidate(float const xCenter, ///< The object's column-centre
                                 float const yCenter, ///< The object's row-centre
                                 float const flux     ///< The object's flux
                                ) :
            lsst::afw::math::SpatialCellMaskedImageCandidate<float>(xCenter, yCenter), _flux(flux) {
        }

        /// Return candidates rating
        double getCandidateRating() const {
            return _flux;
        }

        /// Return the %image
        MaskedImageT::ConstPtr getMaskedImage() const {
            if (_image.get() == NULL) {
                _image = MaskedImageT::Ptr(new MaskedImageT(lsst::afw::geom::ExtentI(getWidth(), getHeight())));
                *_image->getImage() = _flux;
            }

            return _image;
        }
    private:
        double _flux;
    };
%}
