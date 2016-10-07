/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/Box.h"

#include "lsst/afw/math/SpatialCell.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::math;

PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

void declareTestClasses(py::module &mod){
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
    
    
    py::class_<TestCandidate, std::shared_ptr<TestCandidate>, SpatialCellCandidate> 
        clsTestCandidate(mod, ("TestCandidate"));
    clsTestCandidate.def(py::init<float const, float const, float const>());
    clsTestCandidate.def("getCandidateRating", &TestCandidate::getCandidateRating);
    clsTestCandidate.def("setCandidateRating", &TestCandidate::setCandidateRating);
    
    py::class_<TestCandidateVisitor, std::shared_ptr<TestCandidateVisitor>, CandidateVisitor> 
        clsTestCandidateVisitor(mod, ("TestCandidateVisitor"));
    clsTestCandidateVisitor.def(py::init<>());
    clsTestCandidateVisitor.def("getN", &TestCandidateVisitor::getN);
    //clsTestCandidateVisitor.def("", &TestCandidateVisitor::);
    
    py::class_<TestMaskedImageCandidate,
               std::shared_ptr<TestMaskedImageCandidate>,
               SpatialCellMaskedImageCandidate<float>> 
        clsTestMaskedImageCandidate(mod, ("TestMaskedImageCandidate"));
    clsTestMaskedImageCandidate.def(py::init<float const, float const, float const>());
    clsTestMaskedImageCandidate.def("getMaskedImage", &TestMaskedImageCandidate::getMaskedImage);
    
};

template<typename PixelT>
void declareSpatialCellMaskedImageCandidate(py::module &mod, const std::string & suffix){
    py::class_<SpatialCellMaskedImageCandidate<PixelT>,
               std::shared_ptr<SpatialCellMaskedImageCandidate<PixelT>>,
               SpatialCellCandidate>
        clsSpatialCellMaskedImageCandidate(mod, ("SpatialCellMaskedImageCandidate"+suffix).c_str());
    clsSpatialCellMaskedImageCandidate.def_static("setWidth", &SpatialCellMaskedImageCandidate<PixelT>::setWidth);
    clsSpatialCellMaskedImageCandidate.def_static("getWidth", &SpatialCellMaskedImageCandidate<PixelT>::getWidth);
    clsSpatialCellMaskedImageCandidate.def_static("setHeight", &SpatialCellMaskedImageCandidate<PixelT>::setHeight);
    clsSpatialCellMaskedImageCandidate.def_static("getHeight", &SpatialCellMaskedImageCandidate<PixelT>::getHeight);
};

PYBIND11_PLUGIN(_spatialCell) {
    py::module mod("_spatialCell", "Python wrapper for afw _spatialCell library");
    
    /* SpatialCellCandidate */
    py::class_<SpatialCellCandidate, std::shared_ptr<SpatialCellCandidate>>
        clsSpatialCellCandidate(mod, "SpatialCellCandidate");
    /* SpatialCellCandidate Member Types and Enums */
    py::enum_<SpatialCellCandidate::Status>(clsSpatialCellCandidate, "Status")
        .value("BAD", SpatialCellCandidate::Status::BAD)
        .value("GOOD",SpatialCellCandidate::Status::GOOD)
        .value("UNKNOWN",SpatialCellCandidate::Status::UNKNOWN)
        .export_values();
    typedef std::vector<PTR(SpatialCellCandidate)> CandidateList;
    /* SpatialCellCandidate Operators */
    py::class_<SpatialCellCandidateIterator>
        clsSpatialCellCandidateIterator(mod, "SpatialCellCandidateIterator");
    clsSpatialCellCandidateIterator.def("__incr__", [](SpatialCellCandidateIterator &it) -> void{
        ++it;
        return;
    });
    clsSpatialCellCandidateIterator.def("__deref__", [](SpatialCellCandidateIterator &it) ->
        PTR(SpatialCellCandidate){
            return *it;
    });
    clsSpatialCellCandidateIterator.def(py::self == py::self);
    clsSpatialCellCandidateIterator.def(py::self != py::self);
    clsSpatialCellCandidateIterator.def(py::self - py::self);
    /* SpatialCellCandidate Members */
    clsSpatialCellCandidate.def("getXCenter", &SpatialCellCandidate::getXCenter);
    clsSpatialCellCandidate.def("getYCenter", &SpatialCellCandidate::getYCenter);
    clsSpatialCellCandidate.def("setStatus", &SpatialCellCandidate::setStatus);
    clsSpatialCellCandidate.def("getId", &SpatialCellCandidate::getId);
    
    declareSpatialCellMaskedImageCandidate<double>(mod, "D");
    declareSpatialCellMaskedImageCandidate<float>(mod, "F");
    
    /* SpatialCell */
    py::class_<SpatialCell, std::shared_ptr<SpatialCell>>
        clsSpatialCell(mod, "SpatialCell");
    /* SpatialCell Constructors */
    clsSpatialCell.def(py::init<std::string const&,
                                lsst::afw::geom::Box2I const&,
                                CandidateList const&>(),
                       "label"_a,
                       "bbox"_a=lsst::afw::geom::Box2I(),
                       "candidateList"_a=CandidateList());
    /* SpatialCell Members */
    clsSpatialCell.def("getLabel", &SpatialCell::getLabel);
    clsSpatialCell.def("begin", (SpatialCellCandidateIterator (SpatialCell::*)()) &SpatialCell::begin);
    clsSpatialCell.def("begin", (SpatialCellCandidateIterator (SpatialCell::*)(bool)) &SpatialCell::begin);
    clsSpatialCell.def("end", (SpatialCellCandidateIterator (SpatialCell::*)()) &SpatialCell::end);
    clsSpatialCell.def("end", (SpatialCellCandidateIterator (SpatialCell::*)(bool)) &SpatialCell::end);
    clsSpatialCell.def("insertCandidate", &SpatialCell::insertCandidate);
    clsSpatialCell.def("size", &SpatialCell::size);
    clsSpatialCell.def("setIgnoreBad", &SpatialCell::setIgnoreBad);
    clsSpatialCell.def("getCandidateById",
                        &SpatialCell::getCandidateById,
                        "id"_a,
                        "noThrow"_a=false);
    clsSpatialCell.def("sortCandidates", &SpatialCell::sortCandidates);
    clsSpatialCell.def("empty", &SpatialCell::empty);
    clsSpatialCell.def("getBBox", &SpatialCell::getBBox);
    
    /* SpatialCellSet */
    py::class_<SpatialCellSet, std::shared_ptr<SpatialCellSet>>
        clsSpatialCellSet(mod, "SpatialCellSet");
    clsSpatialCellSet.def(py::init<lsst::afw::geom::Box2I const&, int, int>(),
                          "region"_a, "xSize"_a, "ySize"_a=0);
    clsSpatialCellSet.def("getCellList", &SpatialCellSet::getCellList);
    clsSpatialCellSet.def("insertCandidate", &SpatialCellSet::insertCandidate);
    clsSpatialCellSet.def("visitCandidates", 
                          (void (SpatialCellSet::*)(CandidateVisitor *, int const, bool const))
                              &SpatialCellSet::visitCandidates,
                          "visitor"_a, "nMaxPerCell"_a=-1, "ignoreExceptions"_a=false);
    clsSpatialCellSet.def("visitCandidates",
                          (void (SpatialCellSet::*)(CandidateVisitor *, int const, bool const) const)
                              &SpatialCellSet::visitCandidates,
                          "visitor"_a, "nMaxPerCell"_a=-1, "ignoreExceptions"_a=false);
    clsSpatialCellSet.def("getCandidateById", &SpatialCellSet::getCandidateById, "id"_a, "noThrow"_a=false);
    clsSpatialCellSet.def("sortCandidates", &SpatialCellSet::sortCandidates);
    //clsSpatialCellSet.def("", &SpatialCellSet::);
    
    /* CandidateVisitor */
    py::class_<CandidateVisitor, std::shared_ptr<CandidateVisitor>>
        clsCandidateVisitor(mod, "CandidateVisitor");
    
    
    
    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    
    /* Test Members */
    declareTestClasses(mod);

    return mod.ptr();
}