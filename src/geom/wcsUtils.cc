/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
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

#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <limits>

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"
#include "lsst/afw/geom/wcsUtils.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {
/*
 * Get the name of a SIP matrix coefficient card
 *
 * This works for matrix coefficients of the form: <name>_i_j where i and j are zero-based.
 * Thus it works for SIP matrix terms but not CD matrix terms (because CD terms use 1-based indexing
 * and have no underscore between the name and the first index).
 */
std::string getSipCoeffCardName(std::string const& name, int i, int j) {
    return name + std::to_string(i) + "_" + std::to_string(j);
}

}  // namespace

std::shared_ptr<daf::base::PropertyList> createTrivialWcsMetadata(std::string const& wcsName,
                                                                  lsst::geom::Point2I const& xy0) {
    std::shared_ptr<daf::base::PropertyList> wcsMetaData(new daf::base::PropertyList);

    wcsMetaData->set("CTYPE1" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CTYPE2" + wcsName, "LINEAR", "Type of projection");
    wcsMetaData->set("CRPIX1" + wcsName, static_cast<double>(1), "Column Pixel Coordinate of Reference");
    wcsMetaData->set("CRPIX2" + wcsName, static_cast<double>(1), "Row Pixel Coordinate of Reference");
    wcsMetaData->set("CRVAL1" + wcsName, static_cast<double>(xy0[0]), "Column pixel of Reference Pixel");
    wcsMetaData->set("CRVAL2" + wcsName, static_cast<double>(xy0[1]), "Row pixel of Reference Pixel");
    wcsMetaData->set("CUNIT1" + wcsName, "PIXEL", "Column unit");
    wcsMetaData->set("CUNIT2" + wcsName, "PIXEL", "Row unit");

    return wcsMetaData;
}

void deleteBasicWcsMetadata(daf::base::PropertySet& metadata, std::string const& wcsName) {
    std::vector<std::string> const names = {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1",
                                            "CTYPE2", "CUNIT1", "CUNIT2", "CD1_1",  "CD1_2",
                                            "CD2_1",  "CD2_2",  "WCSAXES"};
    for (auto const& name : names) {
        if (metadata.exists(name + wcsName)) {
            metadata.remove(name + wcsName);
        }
    }
}

Eigen::Matrix2d getCdMatrixFromMetadata(daf::base::PropertySet& metadata) {
    Eigen::Matrix2d matrix;
    bool found{false};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::string const cardName = "CD" + std::to_string(i + 1) + "_" + std::to_string(j + 1);
            if (metadata.exists(cardName)) {
                matrix(i, j) = metadata.getAsDouble(cardName);
                found = true;
            } else {
                matrix(i, j) = 0.0;
            }
        }
    }
    if (!found) {
        throw LSST_EXCEPT(pex::exceptions::TypeError, "No CD matrix coefficients found");
    }
    return matrix;
}

lsst::geom::Point2I getImageXY0FromMetadata(daf::base::PropertySet& metadata, std::string const& wcsName,
                                      bool strip) {
    int x0 = 0;  // Our value of X0
    int y0 = 0;  // Our value of Y0

    if (metadata.exists("CRPIX1" + wcsName) && metadata.exists("CRPIX2" + wcsName) &&
        metadata.exists("CRVAL1" + wcsName) && metadata.exists("CRVAL2" + wcsName) &&
        (metadata.getAsDouble("CRPIX1" + wcsName) == 1.0) &&
        (metadata.getAsDouble("CRPIX2" + wcsName) == 1.0)) {
        x0 = static_cast<int>(std::lround(metadata.getAsDouble("CRVAL1" + wcsName)));
        y0 = static_cast<int>(std::lround(metadata.getAsDouble("CRVAL2" + wcsName)));
        if (strip) {
            deleteBasicWcsMetadata(metadata, wcsName);
        }
    }
    return lsst::geom::Point2I(x0, y0);
}

Eigen::MatrixXd getSipMatrixFromMetadata(daf::base::PropertySet const& metadata, std::string const& name) {
    std::string cardName = name + "_ORDER";
    if (!metadata.exists(cardName)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          "metadata does not contain SIP matrix " + name + ": " + cardName + " not found");
    };
    int order = metadata.getAsInt(cardName);
    if (order < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          "matrix order = " + std::to_string(order) + " < 0");
    }
    Eigen::MatrixXd matrix(order + 1, order + 1);
    auto const coeffName = name + "_";
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            std::string const cardName = getSipCoeffCardName(coeffName, i, j);
            if (metadata.exists(cardName)) {
                matrix(i, j) = metadata.getAsDouble(cardName);
            } else {
                matrix(i, j) = 0.0;
            }
        }
    }
    return matrix;
}

bool hasSipMatrix(daf::base::PropertySet const& metadata, std::string const& name) {
    std::string cardName = name + "_ORDER";
    if (!metadata.exists(cardName)) {
        return false;
    }
    return metadata.getAsInt(cardName) >= 0;
}

std::shared_ptr<daf::base::PropertyList> makeSipMatrixMetadata(Eigen::MatrixXd const& matrix,
                                                               std::string const& name) {
    if (matrix.rows() != matrix.cols() || matrix.rows() < 1) {
        throw LSST_EXCEPT(lsst::pex::exceptions::TypeError,
                          "Matrix must be square and at least 1 x 1; dimensions = " +
                                  std::to_string(matrix.rows()) + " x " + std::to_string(matrix.cols()));
    }
    int const order = matrix.rows() - 1;
    auto metadata = std::make_shared<daf::base::PropertyList>();
    std::string cardName = name + "_ORDER";
    metadata->set(cardName, order);
    auto const coeffName = name + "_";
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            if (matrix(i, j) != 0.0) {
                metadata->set(getSipCoeffCardName(coeffName, i, j), matrix(i, j));
            }
        }
    }
    return metadata;
}

std::shared_ptr<daf::base::PropertyList> makeSimpleWcsMetadata(lsst::geom::Point2D const& crpix,
                                                               lsst::geom::SpherePoint const& crval,
                                                               Eigen::Matrix2d const& cdMatrix,
                                                               std::string const& projection) {
    auto pl = std::make_shared<daf::base::PropertyList>();
    pl->add("RADESYS", "ICRS");
    pl->add("EQUINOX", 2000);  // not needed, but may help some older code
    pl->add("CTYPE1", "RA---" + projection);
    pl->add("CTYPE2", "DEC--" + projection);
    pl->add("CRPIX1", crpix[0] + 1);
    pl->add("CRPIX2", crpix[1] + 1);
    pl->add("CRVAL1", crval[0].asDegrees());
    pl->add("CRVAL2", crval[1].asDegrees());
    pl->add("CUNIT1", "deg");
    pl->add("CUNIT2", "deg");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (cdMatrix(i, j) != 0.0) {
                std::string name = "CD" + std::to_string(i + 1) + "_" + std::to_string(j + 1);
                pl->add(name, cdMatrix(i, j));
            }
        }
    }
    return pl;
}

std::shared_ptr<daf::base::PropertyList> makeTanSipMetadata(lsst::geom::Point2D const& crpix,
                                                            lsst::geom::SpherePoint const& crval,
                                                            Eigen::Matrix2d const& cdMatrix,
                                                            Eigen::MatrixXd const& sipA,
                                                            Eigen::MatrixXd const& sipB) {
    auto metadata = makeSimpleWcsMetadata(crpix, crval, cdMatrix, "TAN-SIP");
    metadata->combine(makeSipMatrixMetadata(sipA, "A"));
    metadata->combine(makeSipMatrixMetadata(sipB, "B"));
    return metadata;
}

std::shared_ptr<daf::base::PropertyList> makeTanSipMetadata(
        lsst::geom::Point2D const& crpix, lsst::geom::SpherePoint const& crval, Eigen::Matrix2d const& cdMatrix,
        Eigen::MatrixXd const& sipA, Eigen::MatrixXd const& sipB, Eigen::MatrixXd const& sipAp,
        Eigen::MatrixXd const& sipBp) {
    auto metadata = makeTanSipMetadata(crpix, crval, cdMatrix, sipA, sipB);
    metadata->combine(makeSipMatrixMetadata(sipAp, "AP"));
    metadata->combine(makeSipMatrixMetadata(sipBp, "BP"));
    return metadata;
}

}  // namespace geom
}  // namespace afw
}  // namespace lsst
