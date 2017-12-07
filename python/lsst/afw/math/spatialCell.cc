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

#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/math/SpatialCell.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {

using CandidateList = std::vector<std::shared_ptr<SpatialCellCandidate>>;

// Wrap SpatialCellCandidate (an abstract class so no constructor is wrapped)
void wrapSpatialCellCandidate(py::module &mod) {
    py::class_<SpatialCellCandidate, std::shared_ptr<SpatialCellCandidate>> cls(mod, "SpatialCellCandidate");

    py::enum_<SpatialCellCandidate::Status>(cls, "Status")
            .value("BAD", SpatialCellCandidate::Status::BAD)
            .value("GOOD", SpatialCellCandidate::Status::GOOD)
            .value("UNKNOWN", SpatialCellCandidate::Status::UNKNOWN)
            .export_values();

    cls.def("getXCenter", &SpatialCellCandidate::getXCenter);
    cls.def("getYCenter", &SpatialCellCandidate::getYCenter);
    cls.def("instantiate", &SpatialCellCandidate::instantiate);
    cls.def("getCandidateRating", &SpatialCellCandidate::getCandidateRating);
    cls.def("setCandidateRating", &SpatialCellCandidate::setCandidateRating);
    cls.def("getId", &SpatialCellCandidate::getId);
    cls.def("getStatus", &SpatialCellCandidate::getStatus);
    cls.def("setStatus", &SpatialCellCandidate::setStatus);
    cls.def("isBad", &SpatialCellCandidate::isBad);
}

// Wrap SpatialCellCandidateIterator
void wrapSpatialCellCandidateIterator(py::module &mod) {
    py::class_<SpatialCellCandidateIterator> cls(mod, "SpatialCellCandidateIterator");
    cls.def("__incr__", &SpatialCellCandidateIterator::operator++, py::is_operator());
    cls.def("__deref__",
            [](SpatialCellCandidateIterator &it) -> std::shared_ptr<SpatialCellCandidate> { return *it; },
            py::is_operator());
    cls.def("__eq__", &SpatialCellCandidateIterator::operator==, py::is_operator());
    cls.def("__ne__", &SpatialCellCandidateIterator::operator!=, py::is_operator());
    cls.def("__sub__", &SpatialCellCandidateIterator::operator-, py::is_operator());
}

// Wrap SpatialCell
void wrapSpatialCell(py::module &mod) {
    py::class_<SpatialCell, std::shared_ptr<SpatialCell>> cls(mod, "SpatialCell");

    cls.def(py::init<std::string const &, geom::Box2I const &, CandidateList const &>(), "label"_a,
            "bbox"_a = geom::Box2I(), "candidateList"_a = CandidateList());

    cls.def("empty", &SpatialCell::empty);
    cls.def("size", &SpatialCell::size);
    cls.def("__len__", &SpatialCell::size);
    cls.def("getLabel", &SpatialCell::getLabel);
    cls.def("begin", (SpatialCellCandidateIterator (SpatialCell::*)()) & SpatialCell::begin);
    cls.def("begin", (SpatialCellCandidateIterator (SpatialCell::*)(bool)) & SpatialCell::begin);
    cls.def("end", (SpatialCellCandidateIterator (SpatialCell::*)()) & SpatialCell::end);
    cls.def("end", (SpatialCellCandidateIterator (SpatialCell::*)(bool)) & SpatialCell::end);
    cls.def("insertCandidate", &SpatialCell::insertCandidate);
    cls.def("removeCandidate", &SpatialCell::removeCandidate);
    cls.def("setIgnoreBad", &SpatialCell::setIgnoreBad, "ignoreBad"_a);
    cls.def("getIgnoreBad", &SpatialCell::getIgnoreBad);
    cls.def("getCandidateById", &SpatialCell::getCandidateById, "id"_a, "noThrow"_a = false);
    cls.def("getLabel", &SpatialCell::getLabel);
    cls.def("getBBox", &SpatialCell::getBBox);
    cls.def("sortCandidates", &SpatialCell::sortCandidates);
    cls.def("visitCandidates",
            (void (SpatialCell::*)(CandidateVisitor *, int const, bool const, bool const)) &
                    SpatialCell::visitCandidates,
            "visitor"_a, "nMaxPerCell"_a = -1, "ignoreExceptions"_a = false, "reset"_a = true);
    cls.def("visitAllCandidates", (void (SpatialCell::*)(CandidateVisitor *, bool const, bool const)) &
                                          SpatialCell::visitAllCandidates,
            "visitor"_a, "ignoreExceptions"_a = false, "reset"_a = true);
}

// Wrap SpatialCellSet
void wrapSpatialCellSet(py::module &mod) {
    py::class_<SpatialCellSet, std::shared_ptr<SpatialCellSet>> cls(mod, "SpatialCellSet");

    cls.def(py::init<geom::Box2I const &, int, int>(), "region"_a, "xSize"_a, "ySize"_a = 0);

    cls.def("getCellList", &SpatialCellSet::getCellList);
    cls.def("getBBox", &SpatialCellSet::getBBox);
    cls.def("insertCandidate", &SpatialCellSet::insertCandidate);
    cls.def("sortCandidates", &SpatialCellSet::sortCandidates);
    cls.def("visitCandidates", (void (SpatialCellSet::*)(CandidateVisitor *, int const, bool const)) &
                                       SpatialCellSet::visitCandidates,
            "visitor"_a, "nMaxPerCell"_a = -1, "ignoreExceptions"_a = false);
    cls.def("visitAllCandidates",
            (void (SpatialCellSet::*)(CandidateVisitor *, bool const)) & SpatialCellSet::visitAllCandidates,
            "visitor"_a, "ignoreExceptions"_a = false);
    cls.def("getCandidateById", &SpatialCellSet::getCandidateById, "id"_a, "noThrow"_a = false);
    cls.def("setIgnoreBad", &SpatialCellSet::setIgnoreBad, "ignoreBad"_a);
}

// Wrap CandidateVisitor
void wrapCandidateVisitor(py::module &mod) {
    py::class_<CandidateVisitor, std::shared_ptr<CandidateVisitor>> cls(mod, "CandidateVisitor");

    cls.def(py::init<>());

    cls.def("reset", &CandidateVisitor::reset);
    cls.def("processCandidate", &CandidateVisitor::processCandidate);
}

// Wrap class SpatialCellImageCandidate (an abstract class, so no constructor is wrapped)
void wrapSpatialCellImageCandidate(py::module &mod) {
    py::class_<SpatialCellImageCandidate, std::shared_ptr<SpatialCellImageCandidate>, SpatialCellCandidate>
            cls(mod, "SpatialCellImageCandidate");

    cls.def_static("setWidth", &SpatialCellImageCandidate::setWidth, "width"_a);
    cls.def_static("getWidth", &SpatialCellImageCandidate::getWidth);
    cls.def_static("setHeight", &SpatialCellImageCandidate::setHeight, "height"_a);
    cls.def_static("getHeight", &SpatialCellImageCandidate::getHeight);
    cls.def("setChi2", &SpatialCellImageCandidate::setChi2, "chi2"_a);
    cls.def("getChi2", &SpatialCellImageCandidate::getChi2);
}

}  // namespace lsst::afw::math::<anonymous>

// PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

void wrapTestClasses(py::module &mod) {
    /*
     * Test class for SpatialCellCandidate
     */
    class TestCandidate : public SpatialCellCandidate {
    public:
        TestCandidate(float const xCenter,  ///< @internal The object's column-centre
                      float const yCenter,  ///< @internal The object's row-centre
                      float const flux      ///< @internal The object's flux
                      )
                : SpatialCellCandidate(xCenter, yCenter), _flux(flux) {}

        /// @internal Return candidates rating
        double getCandidateRating() const override { return _flux; }
        void setCandidateRating(double flux) override { _flux = flux; }

    private:
        double _flux;
    };

    /// @internal A class to pass around to all our TestCandidates
    class TestCandidateVisitor : public CandidateVisitor {
    public:
        TestCandidateVisitor() : CandidateVisitor(), _n(0) {}

        // Called by SpatialCellSet::visitCandidates before visiting any Candidates
        void reset() override { _n = 0; }

        // Called by SpatialCellSet::visitCandidates for each Candidate
        void processCandidate(SpatialCellCandidate *candidate) override { ++_n; }

        int getN() const { return _n; }

    private:
        int _n;  // number of TestCandidates
    };

    class TestImageCandidate : public SpatialCellImageCandidate {
    public:
        typedef image::MaskedImage<float> MaskedImageT;

        TestImageCandidate(float const xCenter,  ///< @internal The object's column-centre
                           float const yCenter,  ///< @internal The object's row-centre
                           float const flux      ///< @internal The object's flux
                           )
                : SpatialCellImageCandidate(xCenter, yCenter), _flux(flux) {}

        /// @internal Return candidates rating
        double getCandidateRating() const override { return _flux; }

        /// @internal Return the %image
        std::shared_ptr<MaskedImageT const> getMaskedImage() const {
            if (!_image) {
                _image = std::make_shared<MaskedImageT>(geom::ExtentI(getWidth(), getHeight()));
                *_image->getImage() = _flux;
            }
            return _image;
        }

    private:
        mutable std::shared_ptr<MaskedImageT> _image;
        double _flux;
    };

    py::class_<TestCandidate, std::shared_ptr<TestCandidate>, SpatialCellCandidate> clsTestCandidate(
            mod, ("TestCandidate"));
    clsTestCandidate.def(py::init<float const, float const, float const>());
    clsTestCandidate.def("getCandidateRating", &TestCandidate::getCandidateRating);
    clsTestCandidate.def("setCandidateRating", &TestCandidate::setCandidateRating);

    py::class_<TestCandidateVisitor, std::shared_ptr<TestCandidateVisitor>, CandidateVisitor>
            clsTestCandidateVisitor(mod, ("TestCandidateVisitor"));
    clsTestCandidateVisitor.def(py::init<>());
    clsTestCandidateVisitor.def("getN", &TestCandidateVisitor::getN);

    py::class_<TestImageCandidate, std::shared_ptr<TestImageCandidate>, SpatialCellImageCandidate>
            clsTestImageCandidate(mod, "TestImageCandidate");
    clsTestImageCandidate.def(py::init<float const, float const, float const>(), "xCenter"_a, "yCenter"_a,
                              "flux"_a);
    clsTestImageCandidate.def("getCandidateRating", &TestImageCandidate::getCandidateRating);
    clsTestImageCandidate.def("getMaskedImage", &TestImageCandidate::getMaskedImage);
};

PYBIND11_PLUGIN(_spatialCell) {
    py::module mod("_spatialCell", "Python wrapper for afw _spatialCell library");

    wrapSpatialCellCandidate(mod);
    wrapSpatialCellCandidateIterator(mod);
    wrapSpatialCell(mod);
    wrapSpatialCellSet(mod);
    wrapCandidateVisitor(mod);
    wrapSpatialCellImageCandidate(mod);

    /* Test Members */
    wrapTestClasses(mod);

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::math
