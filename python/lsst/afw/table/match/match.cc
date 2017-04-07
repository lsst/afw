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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Match.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

/// @internal Declare match code templated on two types of catalog
template <typename Catalog1, typename Catalog2>
void declareMatch2(py::module & mod, std::string const & prefix) {
    typedef typename Catalog1::Record Record1;
    typedef typename Catalog2::Record Record2;
    typedef std::vector<Match<typename Catalog1::Record, typename Catalog2::Record>> MatchList;

    using Class = Match<Record1, Record2>;
    py::class_<Class, std::shared_ptr<Class>> clsMatch(mod, (prefix + "Match").c_str());
    clsMatch.def(py::init<>());
    clsMatch.def(py::init<PTR(Record1) const &, PTR(Record2) const &, double>(),
                 "first"_a, "second"_a, "distance"_a);

    // struct fields
    clsMatch.def_readwrite("first", &Match<Record1, Record2>::first);
    clsMatch.def_readwrite("second", &Match<Record1, Record2>::second);
    clsMatch.def_readwrite("distance", &Match<Record1, Record2>::distance);

    // Free Functions
    mod.def("unpackMatches", &unpackMatches<Catalog1, Catalog2>, "matches"_a, "cat1"_a, "cat2"_a);

    mod.def("matchRaDec",
            (MatchList (*)(Catalog1 const &, Catalog2 const &, geom::Angle, MatchControl const &))
             matchRaDec<Catalog1, Catalog2>, "cat1"_a, "cat2"_a, "radius"_a, "mc"_a=MatchControl());
    // The following is deprecated; consider changing the code instead of wrapping it:
    // mod.def("matchRaDec",
    //         (MatchList (*)(Catalog1 const &, Catalog2 const &, geom::Angle, bool))
    //          matchRaDec<Catalog1, Catalog2>, "cat1"_a, "cat2"_a, "radius"_a, "closest"_a);
};

/// @internal Declare match code templated on one type of catalog
template <typename Catalog>
void declareMatch1(py::module &mod) {
    typedef std::vector<Match<typename Catalog::Record, typename Catalog::Record>> MatchList;
    mod.def("matchRaDec",
            (MatchList (*)(Catalog const &, geom::Angle, MatchControl const &))
             matchRaDec<Catalog>, "cat"_a, "radius"_a, "mc"_a=MatchControl());
    // The following is deprecated; consider changing the code instead of wrapping it:
    // mod.def("matchRaDec",
    //         (MatchList (*)(Catalog const &, geom::Angle, bool))
    //          &matchRaDec<Catalog1>, "cat"_a, "radius"_a, "symmetric"_a);
}

}  // <anonymous>

PYBIND11_PLUGIN(match) {
    py::module mod("match", "Python wrapper for afw _match library");

    py::class_<MatchControl> clsMatchControl(mod, "MatchControl");
    clsMatchControl.def(py::init<>());
    LSST_DECLARE_CONTROL_FIELD(clsMatchControl, MatchControl, findOnlyClosest);
    LSST_DECLARE_CONTROL_FIELD(clsMatchControl, MatchControl, symmetricMatch);
    LSST_DECLARE_CONTROL_FIELD(clsMatchControl, MatchControl, includeMismatches);

    declareMatch2<SimpleCatalog, SimpleCatalog>(mod, "Simple");
    declareMatch2<SimpleCatalog, SourceCatalog>(mod, "Reference");
    declareMatch2<SourceCatalog, SourceCatalog>(mod, "Source");
    declareMatch1<SimpleCatalog>(mod);
    declareMatch1<SourceCatalog>(mod);

    mod.def("matchXy",
            (SourceMatchVector (*)(SourceCatalog const &, SourceCatalog const &, double,
                                   MatchControl const &))
             matchXy, "cat1"_a, "cat2"_a, "radius"_a, "mc"_a=MatchControl());
    mod.def("matchXy",
            (SourceMatchVector (*)(SourceCatalog const &, double, MatchControl const &))
             matchXy, "cat"_a, "radius"_a, "mc"_a=MatchControl());
    // The following are deprecated; consider changing the code instead of wrapping them:
    // mod.def("matchXy",
    //         (SourceMatchVector (*)(SourceCatalog const &, SourceCatalog const &, double, bool))
    //          matchXy, "cat1"_a, "cat2"_a, "radius"_a, "closest"_a);
    // mod.def("matchXy",
    //         (SourceMatchVector (*)(SourceCatalog const &, double, bool))
    //          &matchXy, "cat"_a, "radius"_a, "symmetric"_a);


    return mod.ptr();
}

}}}  // namespace lsst::afw::table::<anonymous>
