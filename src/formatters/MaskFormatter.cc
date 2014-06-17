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
 

/** @file
 * @brief Implementation of MaskFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2151 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * @ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "lsst/afw/formatters/MaskFormatter.h"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/Mask.h"

#include "lsst/afw/image/LsstImageTypes.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.MaskFormatter", level, s);
}

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::Storage;
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

template<> std::string MaskFormatterTraits<MaskPixel>::name() {
    static std::string name = "Mask";
    return name;
}

template <typename MaskPixelT>
lsst::daf::persistence::FormatterRegistration MaskFormatter<MaskPixelT>::registration(
    MaskFormatterTraits<MaskPixelT>::name(),
    typeid(Mask<MaskPixelT>),
    createInstance);

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::MaskFormatter(
    lsst::pex::policy::Policy::Ptr) :
    lsst::daf::persistence::Formatter(typeid(this)) {
}

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::~MaskFormatter(void) {
}

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr) {
    execTrace("MaskFormatter write start");
    Mask<MaskPixelT> const* ip =
        dynamic_cast<Mask<MaskPixelT> const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-Mask");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("MaskFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        // Need to cast away const because writeFits modifies the metadata.
        Mask<MaskPixelT>* vip = const_cast<Mask<MaskPixelT>*>(ip);
        vip->writeFits(fits->getPath());
        execTrace("MaskFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Mask");
}

template <typename MaskPixelT>
Persistable* MaskFormatter<MaskPixelT>::read(
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr) {
    execTrace("MaskFormatter read start");
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        Mask<MaskPixelT>* ip = new Mask<MaskPixelT>;
        boost->getIArchive() & *ip;
        execTrace("MaskFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        Mask<MaskPixelT>* ip = new Mask<MaskPixelT>(fits->getPath(), fits->getHdu());
        execTrace("MaskFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Mask");
}

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::update(
    Persistable*,
    Storage::Ptr,
    lsst::daf::base::PropertySet::Ptr) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for Mask");
}

template <typename MaskPixelT> template <class Archive>
void MaskFormatter<MaskPixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("MaskFormatter delegateSerialize start");
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
    ar & cols & rows & planes;
    if (Archive::is_loading::value) {
        ip->_vwImagePtr->set_size(cols, rows, planes);
    }
    unsigned int pixels = cols * rows * planes;
    MaskPixelT* data = ip->_vwImagePtr->data();
    ar & boost::serialization::make_array(data, pixels);
    execTrace("MaskFormatter delegateSerialize end");
}

template <typename MaskPixelT>
lsst::daf::persistence::Formatter::Ptr MaskFormatter<MaskPixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new MaskFormatter<MaskPixelT>(policy));
}

template class MaskFormatter<MaskPixel>;
//// The followings fails
//// because the function template `delegateSerialize' is obsolete(?)
//template void MaskFormatter<MaskPixel>::delegateSerialize(
//    boost::archive::binary_oarchive&, int const, Persistable*);
//template void MaskFormatter<MaskPixel>::delegateSerialize(
//    boost::archive::binary_iarchive&, int const, Persistable*);


}}} // namespace lsst::afw::formatters
