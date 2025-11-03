/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/native_enum.h>

#include "lsst/cpputils/python.h"

#include "ndarray/pybind11.h"

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/pex/exceptions/python/Exception.h"
#include "lsst/daf/base.h"
#include "lsst/afw/image/Image.h"

#include "lsst/afw/fits.h"

namespace py = pybind11;

using namespace pybind11::literals;
namespace lsst {
namespace afw {
namespace fits {
namespace {
void declareCompression(lsst::cpputils::python::WrapperCollection &wrappers) {
    py::native_enum<CompressionAlgorithm>(wrappers.module, "CompressionAlgorithm", "enum.Enum")
        .value("GZIP_1", CompressionAlgorithm::GZIP_1_)
        .value("GZIP_2", CompressionAlgorithm::GZIP_2_)
        .value("RICE_1", CompressionAlgorithm::RICE_1_)
        .finalize();
    py::native_enum<DitherAlgorithm>(wrappers.module, "DitherAlgorithm", "enum.Enum")
        .value("NO_DITHER", DitherAlgorithm::NO_DITHER_)
        .value("SUBTRACTIVE_DITHER_1", DitherAlgorithm::SUBTRACTIVE_DITHER_1_)
        .value("SUBTRACTIVE_DITHER_2", DitherAlgorithm::SUBTRACTIVE_DITHER_2_)
        .finalize();
    py::native_enum<ScalingAlgorithm>(wrappers.module, "ScalingAlgorithm", "enum.Enum")
        .value("RANGE", ScalingAlgorithm::RANGE)
        .value("STDEV_MASKED", ScalingAlgorithm::STDEV_MASKED)
        .value("STDEV_CFITSIO", ScalingAlgorithm::STDEV_CFITSIO)
        .value("MANUAL", ScalingAlgorithm::MANUAL)
        .finalize();
    wrappers.wrapType(
        py::class_<QuantizationOptions>(wrappers.module, "QuantizationOptions"),
        [](auto &mod, auto &cls) {
            cls.def(
                py::init<DitherAlgorithm, ScalingAlgorithm, std::vector<std::string>, float, int>(),
                py::kw_only(),
                "dither"_a = DitherAlgorithm::NO_DITHER_,
                "scaling"_a = ScalingAlgorithm::STDEV_MASKED,
                "mask_planes"_a = py::list(),
                "level"_a = 0.0,
                "seed"_a = 0
            );
            cls.def_readwrite("dither", &QuantizationOptions::dither);
            cls.def_readwrite("scaling", &QuantizationOptions::scaling);
            cls.def_readwrite("mask_planes", &QuantizationOptions::mask_planes);
            cls.def_readwrite("level", &QuantizationOptions::level);
            cls.def_readwrite("seed", &QuantizationOptions::seed);
        });
    wrappers.wrapType(
        py::class_<CompressionOptions>(wrappers.module, "CompressionOptions"),
        [](auto &mod, auto &cls) {
            cls.def(
                py::init<
                    CompressionAlgorithm,
                    std::size_t,
                    std::size_t,
                    std::optional<QuantizationOptions>
                >(),
                py::kw_only(),
                "algorithm"_a = CompressionAlgorithm::GZIP_2_,
                "tile_width"_a = 0,
                "tile_height"_a = 1,
                "quantization"_a = py::none()
            );
            cls.def_readwrite("algorithm", &CompressionOptions::algorithm);
            cls.def_readwrite("tile_width", &CompressionOptions::tile_width);
            cls.def_readwrite("tile_height", &CompressionOptions::tile_height);
            cls.def_readwrite("quantization", &CompressionOptions::quantization);
        });
}

// Wrapping for lsst::afw::fits::Fits
//
// Not every feature is wrapped, only those that we guess might be useful.
// In particular, the header keyword read/write and table read/write are not wrapped.
void declareFits(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<Fits>(wrappers.module, "Fits"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::string const &, std::string const &, int>(), "filename"_a, "mode"_a,
                "behavior"_a = Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
        cls.def(py::init<MemFileManager &, std::string const &, int>(), "manager"_a, "mode"_a,
                "behavior"_a = Fits::AUTO_CLOSE | Fits::AUTO_CHECK);

        cls.def("closeFile", &Fits::closeFile);
        cls.def("getFileName", &Fits::getFileName);
        cls.def("getHdu", &Fits::getHdu);
        cls.def("setHdu", py::overload_cast<int, bool>(&Fits::setHdu), "hdu"_a, "relative"_a = false);
        cls.def(
                "setHdu", [](Fits &self, std::string const &name) { self.setHdu(name); }, "name"_a);
        cls.def("countHdus", &Fits::countHdus);

        cls.def("writeMetadata", &Fits::writeMetadata);
        cls.def(
                "readMetadata", [](Fits &self, bool strip = false) { return readMetadata(self, strip); },
                "strip"_a = false);
        cls.def("createEmpty", &Fits::createEmpty);

        cls.def("readImageI", [](Fits &self) {
            ndarray::Vector<int, 2> const offset;  // initialized to zero by default
            ndarray::Vector<ndarray::Size, 2> shape = self.getImageShape<2>();
            ndarray::Array<int, 2, 2> result = ndarray::allocate(shape[0], shape[1]);
            self.readImage(result, offset);
            return result;
        });

        cls.def("gotoFirstHdu", [](Fits &self) { self.setHdu(DEFAULT_HDU); });

        cls.def("checkCompressedImagePhu", &Fits::checkCompressedImagePhu);

        cls.def_readonly("status", &Fits::status);
    });
}

void declareFitsModule(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrap([](auto &mod) {
        py::class_<MemFileManager> clsMemFileManager(mod, "MemFileManager");

        clsMemFileManager.def(py::init<>());
        clsMemFileManager.def(py::init<size_t>());

        /* TODO: We should really revisit persistence and pickling as this is quite ugly.
         * But it is what Swig did (sort of, it used the cdata.i extension), so I reckon this
         * is cleaner because it does not expose casting to the Python side. */
        clsMemFileManager.def("getLength", &MemFileManager::getLength);
        clsMemFileManager.def("getData", [](MemFileManager &m) {
            return py::bytes(static_cast<char *>(m.getData()), m.getLength());
        });
        clsMemFileManager.def("setData", [](MemFileManager &m, py::bytes const &d, size_t size) {
            memcpy(m.getData(), PyBytes_AsString(d.ptr()), size);
        });
        clsMemFileManager.def(
                "readMetadata",
                [](MemFileManager &self, int hdu = DEFAULT_HDU, bool strip = false) {
                    return readMetadata(self, hdu, strip);
                },
                "hdu"_a = DEFAULT_HDU, "strip"_a = false);
        mod.attr("DEFAULT_HDU") = DEFAULT_HDU;
        mod.def(
            "combineMetadata",
            py::overload_cast<daf::base::PropertyList const&, daf::base::PropertyList const &>(
                combineMetadata
            ),
            "first"_a, "second"_a
        );
        mod.def("makeLimitedFitsHeader", &makeLimitedFitsHeader, "metadata"_a,
                "excludeNames"_a = std::set<std::string>());
        mod.def(
                "readMetadata",
                [](std::string const &filename, int hdu = DEFAULT_HDU, bool strip = false) {
                    return readMetadata(filename, hdu, strip);
                },
                "fileName"_a, "hdu"_a = DEFAULT_HDU, "strip"_a = false);

        mod.def(
                "readMetadata",
                [](std::string const &filename, std::string const &hduname, bool strip = false) {
                    return readMetadata(filename, hduname, HduType::ANY, 0, strip);
                },
                "fileName"_a, "hduName"_a, "strip"_a = false);
    });
}
}  // namespace
PYBIND11_MODULE(_fits, mod) {
    lsst::cpputils::python::WrapperCollection wrappers(mod, "lsst.afw.fits");
    wrappers.addInheritanceDependency("lsst.pex.exceptions");
    wrappers.addSignatureDependency("lsst.daf.base");
    // FIXME: after afw.image pybind wrappers are converted
    //wrappers.addSignatureDependency("lsst.afw.image");
    auto cls = wrappers.wrapException<FitsError, lsst::pex::exceptions::IoError>("FitsError", "IoError");
    cls.def(py::init<std::string const &>());
    declareCompression(wrappers);
    declareFits(wrappers);
    declareFitsModule(wrappers);
    wrappers.finish();
}
}  // namespace fits
}  // namespace afw
}  // namespace lsst
