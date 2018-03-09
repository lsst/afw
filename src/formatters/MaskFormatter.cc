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
 * Implementation of MaskFormatter class
 */

#ifndef __GNUC__
#define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include <string>

#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "lsst/afw/formatters/MaskFormatter.h"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/fits.h"

#include "lsst/afw/image/LsstImageTypes.h"

namespace {
LOG_LOGGER _log = LOG_GET("afw.MaskFormatter");
}

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::FormatterStorage;
using lsst::afw::image::Mask;
using lsst::afw::image::MaskPixel;

namespace lsst {
namespace afw {
namespace formatters {

template <typename imagePixelT>
class MaskFormatterTraits {
public:
    static std::string name();
};

template <>
std::string MaskFormatterTraits<MaskPixel>::name() {
    static std::string name = "Mask";
    return name;
}

template <typename MaskPixelT>
lsst::daf::persistence::FormatterRegistration MaskFormatter<MaskPixelT>::registration(
        MaskFormatterTraits<MaskPixelT>::name(), typeid(Mask<MaskPixelT>), createInstance);

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::MaskFormatter(std::shared_ptr<lsst::pex::policy::Policy>)
        : lsst::daf::persistence::Formatter(typeid(this)) {}

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::~MaskFormatter() = default;

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::write(Persistable const* persistable, std::shared_ptr<FormatterStorage> storage,
                                      std::shared_ptr<lsst::daf::base::PropertySet> additionalData) {
    LOGL_DEBUG(_log, "MaskFormatter write start");
    Mask<MaskPixelT> const* ip = dynamic_cast<Mask<MaskPixelT> const*>(persistable);
    if (ip == nullptr) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-Mask");
    }
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "MaskFormatter write BoostStorage");
        boost->getOArchive() & *ip;
        LOGL_DEBUG(_log, "MaskFormatter write end");
        return;
    }
    auto fits = std::dynamic_pointer_cast<FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "MaskFormatter write FitsStorage");
        // Need to cast away const because writeFits modifies the metadata.

        fits::ImageWriteOptions options;
        if (additionalData) {
            try {
                options = fits::ImageWriteOptions(*additionalData->getAsPropertySetPtr("mask"));
            } catch (std::exception const& exc) {
                LOGLS_WARN(_log, "Unable to construct mask write options (" << exc.what() <<
                           "); writing with default options");
            }
        }

        Mask<MaskPixelT>* vip = const_cast<Mask<MaskPixelT>*>(ip);
        vip->writeFits(fits->getPath(), options);
        LOGL_DEBUG(_log, "MaskFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for Mask");
}

template <typename MaskPixelT>
Persistable* MaskFormatter<MaskPixelT>::read(std::shared_ptr<FormatterStorage> storage,
                                             std::shared_ptr<lsst::daf::base::PropertySet>) {
    LOGL_DEBUG(_log, "MaskFormatter read start");
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "MaskFormatter read BoostStorage");
        Mask<MaskPixelT>* ip = new Mask<MaskPixelT>;
        boost->getIArchive() & *ip;
        LOGL_DEBUG(_log, "MaskFormatter read end");
        return ip;
    }
    auto fits = std::dynamic_pointer_cast<FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "MaskFormatter read FitsStorage");
        Mask<MaskPixelT>* ip = new Mask<MaskPixelT>(fits->getPath(), fits->getHdu());
        LOGL_DEBUG(_log, "MaskFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for Mask");
}

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::update(Persistable*, std::shared_ptr<FormatterStorage>,
                                       std::shared_ptr<lsst::daf::base::PropertySet>) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for Mask");
}

template <typename MaskPixelT>
template <class Archive>
void MaskFormatter<MaskPixelT>::delegateSerialize(Archive& ar, int const version, Persistable* persistable) {
    LOGL_DEBUG(_log, "MaskFormatter delegateSerialize start");
    Mask<MaskPixelT>* ip = dynamic_cast<Mask<MaskPixelT>*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-Mask");
    }
    ar & ip->_offsetRows & ip->_offsetCols;
    ar & ip->_maskPlaneDict;
    unsigned int cols;
    unsigned int rows;
    unsigned int planes;
    if (Archive::is_saving::value) {
        cols = ip->_vwImagePtr->cols();
        rows = ip->_vwImagePtr->rows();
        planes = ip->_vwImagePtr->planes();
    }
    ar& cols& rows& planes;
    if (Archive::is_loading::value) {
        ip->_vwImagePtr->set_size(cols, rows, planes);
    }
    unsigned int pixels = cols * rows * planes;
    MaskPixelT* data = ip->_vwImagePtr->data();
    ar& boost::serialization::make_array(data, pixels);
    LOGL_DEBUG(_log, "MaskFormatter delegateSerialize end");
}

template <typename MaskPixelT>
std::shared_ptr<lsst::daf::persistence::Formatter> MaskFormatter<MaskPixelT>::createInstance(
        std::shared_ptr<lsst::pex::policy::Policy> policy) {
    return std::shared_ptr<lsst::daf::persistence::Formatter>(new MaskFormatter<MaskPixelT>(policy));
}

template class MaskFormatter<MaskPixel>;
// The followings fails
// because the function template `delegateSerialize' is obsolete(?)
// template void MaskFormatter<MaskPixel>::delegateSerialize(
//    boost::archive::binary_oarchive&, int const, Persistable*);
// template void MaskFormatter<MaskPixel>::delegateSerialize(
//    boost::archive::binary_iarchive&, int const, Persistable*);
}  // namespace formatters
}  // namespace afw
}  // namespace lsst
