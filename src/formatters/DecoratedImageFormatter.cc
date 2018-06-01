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

/*
 * Implementation of DecoratedImageFormatter class
 */

#ifndef __GNUC__
#define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include <cstdint>
#include <memory>
#include <string>
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include "boost/serialization/nvp.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/log/Log.h"
#include "lsst/afw/formatters/DecoratedImageFormatter.h"
#include "lsst/afw/image/Image.h"

namespace {
LOG_LOGGER _log = LOG_GET("afw.DecoratedImageFormatter");
}

using boost::serialization::make_nvp;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::XmlStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::FormatterStorage;
using lsst::afw::image::DecoratedImage;

namespace lsst {
namespace afw {
namespace formatters {

template <typename ImagePixelT>
class DecoratedImageFormatterTraits {
public:
    static std::string name();
};

template <>
std::string DecoratedImageFormatterTraits<std::uint16_t>::name() {
    static std::string name = "DecoratedImageU";
    return name;
}
template <>
std::string DecoratedImageFormatterTraits<int>::name() {
    static std::string name = "DecoratedImageI";
    return name;
}
template <>
std::string DecoratedImageFormatterTraits<float>::name() {
    static std::string name = "DecoratedImageF";
    return name;
}
template <>
std::string DecoratedImageFormatterTraits<double>::name() {
    static std::string name = "DecoratedImageD";
    return name;
}
template <>
std::string DecoratedImageFormatterTraits<std::uint64_t>::name() {
    static std::string name = "DecoratedImageL";
    return name;
}

template <typename ImagePixelT>
lsst::daf::persistence::FormatterRegistration DecoratedImageFormatter<ImagePixelT>::registration(
        DecoratedImageFormatterTraits<ImagePixelT>::name(), typeid(DecoratedImage<ImagePixelT>),
        createInstance);

template <typename ImagePixelT>
DecoratedImageFormatter<ImagePixelT>::DecoratedImageFormatter(std::shared_ptr<lsst::pex::policy::Policy>)
        : lsst::daf::persistence::Formatter(typeid(this)) {}

template <typename ImagePixelT>
DecoratedImageFormatter<ImagePixelT>::~DecoratedImageFormatter() = default;

template <typename ImagePixelT>
void DecoratedImageFormatter<ImagePixelT>::write(Persistable const* persistable,
                                                 std::shared_ptr<FormatterStorage> storage,
                                                 std::shared_ptr<lsst::daf::base::PropertySet>) {
    LOGL_DEBUG(_log, "DecoratedImageFormatter write start");
    DecoratedImage<ImagePixelT> const* ip = dynamic_cast<DecoratedImage<ImagePixelT> const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-DecoratedImage");
    }

    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter write BoostStorage");
        boost->getOArchive() & *ip;
        LOGL_DEBUG(_log, "DecoratedImageFormatter write end");
        return;
    }
    auto xml = std::dynamic_pointer_cast<XmlStorage>(storage);
    if (xml) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter write XmlStorage");
        xml->getOArchive() & make_nvp("img", *ip);
        LOGL_DEBUG(_log, "DecoratedImageFormatter write end");
        return;
    }
    auto fits = std::dynamic_pointer_cast<FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter write FitsStorage");

        ip->writeFits(fits->getPath());
        // @todo Do something with these fields?
        // int _X0;
        // int _Y0;
        LOGL_DEBUG(_log, "DecoratedImageFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized FormatterStorage for DecoratedImage");
}

template <typename ImagePixelT>
Persistable* DecoratedImageFormatter<ImagePixelT>::read(
        std::shared_ptr<FormatterStorage> storage,
        std::shared_ptr<lsst::daf::base::PropertySet> additionalData) {
    LOGL_DEBUG(_log, "DecoratedImageFormatter read start");
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter read BoostStorage");
        DecoratedImage<ImagePixelT>* ip = new DecoratedImage<ImagePixelT>;
        boost->getIArchive() & *ip;
        LOGL_DEBUG(_log, "DecoratedImageFormatter read end");
        return ip;
    }
    auto xml = std::dynamic_pointer_cast<XmlStorage>(storage);
    if (xml) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter read XmlStorage");
        DecoratedImage<ImagePixelT>* ip = new DecoratedImage<ImagePixelT>;
        boost->getIArchive() & make_nvp("img", *ip);
        LOGL_DEBUG(_log, "DecoratedImageFormatter read end");
        return ip;
    }
    auto fits = std::dynamic_pointer_cast<FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "DecoratedImageFormatter read FitsStorage");
        lsst::geom::Box2I box;
        if (additionalData->exists("llcX")) {
            int llcX = additionalData->get<int>("llcX");
            int llcY = additionalData->get<int>("llxY");
            int width = additionalData->get<int>("width");
            int height = additionalData->get<int>("height");
            box = lsst::geom::Box2I(lsst::geom::Point2I(llcX, llcY), lsst::geom::Extent2I(width, height));
        }
        DecoratedImage<ImagePixelT>* ip =
                new DecoratedImage<ImagePixelT>(fits->getPath(), fits->getHdu(), box);
        LOGL_DEBUG(_log, "DecoratedImageFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized FormatterStorage for DecoratedImage");
}

template <typename ImagePixelT>
void DecoratedImageFormatter<ImagePixelT>::update(lsst::daf::base::Persistable*,
                                                  std::shared_ptr<lsst::daf::persistence::FormatterStorage>,
                                                  std::shared_ptr<lsst::daf::base::PropertySet>) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for DecoratedImage");
}

template <typename ImagePixelT>
template <class Archive>
void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(Archive&, int const, Persistable* persistable) {
    LOGL_DEBUG(_log, "DecoratedImageFormatter delegateSerialize start");
    DecoratedImage<ImagePixelT>* ip = dynamic_cast<DecoratedImage<ImagePixelT>*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-DecoratedImage");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "DecoratedImage serialization not yet implemented");
}

template <typename ImagePixelT>
std::shared_ptr<lsst::daf::persistence::Formatter> DecoratedImageFormatter<ImagePixelT>::createInstance(
        std::shared_ptr<lsst::pex::policy::Policy> policy) {
    return std::shared_ptr<lsst::daf::persistence::Formatter>(
            new DecoratedImageFormatter<ImagePixelT>(policy));
}

#define InstantiateFormatter(ImagePixelT)                                                                   \
    template class DecoratedImageFormatter<ImagePixelT>;                                                    \
    template void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(boost::archive::text_oarchive&,   \
                                                                          int const, Persistable*);         \
    template void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(boost::archive::text_iarchive&,   \
                                                                          int const, Persistable*);         \
    template void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(boost::archive::binary_oarchive&, \
                                                                          int const, Persistable*);         \
    template void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(boost::archive::binary_iarchive&, \
                                                                          int const, Persistable*);

InstantiateFormatter(std::uint16_t);
InstantiateFormatter(int);
InstantiateFormatter(float);
InstantiateFormatter(double);
InstantiateFormatter(std::uint64_t);

#undef InstantiateFormatter
}  // namespace formatters
}  // namespace afw
}  // namespace lsst
