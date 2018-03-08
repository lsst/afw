// -*- LSST-C++ -*-
#include <limits>
#include <typeinfo>
#include <cmath>

#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/math/offsetImage.h"

namespace lsst {
namespace afw {
namespace detection {

namespace {

// Comparison function that determines when we used the cached image instead of recomputing it.
// We'll probably want a tolerance for colors someday too, but they're just a placeholder right now
// so it's not worth the effort.
bool comparePsfEvalPoints(geom::Point2D const &a, geom::Point2D const &b) {
    // n.b. desired tolerance is actually sqrt(eps), so tolerance squared is eps.
    return (a - b).computeSquaredNorm() < std::numeric_limits<double>::epsilon();
}

bool isPointNull(geom::Point2D const &p) { return std::isnan(p.getX()) && std::isnan(p.getY()); }

}  // namespace

Psf::Psf(bool isFixed) : daf::base::Citizen(typeid(this)), _isFixed(isFixed) {}

std::shared_ptr<image::Image<double>> Psf::recenterKernelImage(std::shared_ptr<Image> im,
                                                               geom::Point2D const &position,
                                                               std::string const &warpAlgorithm,
                                                               unsigned int warpBuffer) {
    // "ir" : (integer, residual)
    std::pair<int, double> const irX = image::positionToIndex(position.getX(), true);
    std::pair<int, double> const irY = image::positionToIndex(position.getY(), true);

    if (irX.second != 0.0 || irY.second != 0.0) {
        im = math::offsetImage(*im, irX.second, irY.second, warpAlgorithm, warpBuffer);
    }

    im->setXY0(irX.first + im->getX0(), irY.first + im->getY0());
    return im;
}

std::shared_ptr<Psf::Image> Psf::computeImage(geom::Point2D position, image::Color color,
                                              ImageOwnerEnum owner) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    std::shared_ptr<Psf::Image> result;
    if (_cachedImage && color == _cachedImageColor && comparePsfEvalPoints(position, _cachedImagePosition)) {
        result = _cachedImage;
    } else {
        result = doComputeImage(position, color);
        _cachedImage = result;
        _cachedImageColor = color;
        _cachedImagePosition = position;
    }
    if (owner == COPY) {
        result = std::make_shared<Image>(*result, true);
    }
    return result;
}

std::shared_ptr<Psf::Image> Psf::computeKernelImage(geom::Point2D position, image::Color color,
                                                    ImageOwnerEnum owner) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    std::shared_ptr<Psf::Image> result;
    if (_cachedKernelImage && (_isFixed || (color == _cachedKernelImageColor &&
                                            comparePsfEvalPoints(position, _cachedKernelImagePosition)))) {
        result = _cachedKernelImage;
    } else {
        result = doComputeKernelImage(position, color);
        _cachedKernelImage = result;
        _cachedKernelImageColor = color;
        _cachedKernelImagePosition = position;
    }
    if (owner == COPY) {
        result = std::make_shared<Image>(*result, true);
    }
    return result;
}

geom::Box2I Psf::computeBBox(geom::Point2D position, image::Color color) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    return doComputeBBox(position, color);
}

std::shared_ptr<math::Kernel const> Psf::getLocalKernel(geom::Point2D position, image::Color color) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    // FixedKernel ctor will deep copy image, so we can use INTERNAL.
    std::shared_ptr<Image> image = computeKernelImage(position, color, INTERNAL);
    return std::make_shared<math::FixedKernel>(*image);
}

double Psf::computePeak(geom::Point2D position, image::Color color) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    std::shared_ptr<Image> image = computeKernelImage(position, color, INTERNAL);
    return (*image)(-image->getX0(), -image->getY0());
}

double Psf::computeApertureFlux(double radius, geom::Point2D position, image::Color color) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    return doComputeApertureFlux(radius, position, color);
}

geom::ellipses::Quadrupole Psf::computeShape(geom::Point2D position, image::Color color) const {
    if (isPointNull(position)) position = getAveragePosition();
    if (color.isIndeterminate()) color = getAverageColor();
    return doComputeShape(position, color);
}

std::shared_ptr<Psf::Image> Psf::doComputeImage(geom::Point2D const &position,
                                                image::Color const &color) const {
    std::shared_ptr<Psf::Image> im = computeKernelImage(position, color, COPY);
    return recenterKernelImage(im, position);
}

geom::Point2D Psf::getAveragePosition() const { return geom::Point2D(); }
}  // namespace detection
}  // namespace afw
}  // namespace lsst
