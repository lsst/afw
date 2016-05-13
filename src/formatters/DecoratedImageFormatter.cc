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
 * @brief Implementation of DecoratedImageFormatter class
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
static char const* SVNid __attribute__((unused)) =
    "$Id$";

#include <cstdint>
#include <memory>
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include "boost/serialization/nvp.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/DecoratedImageFormatter.h"
#include "lsst/afw/image/Image.h"


#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.DecoratedImageFormatter", level, s);
}

using boost::serialization::make_nvp;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::XmlStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::Storage;
using lsst::afw::image::DecoratedImage;

namespace lsst {
namespace afw {
namespace formatters {

template <typename ImagePixelT>
class DecoratedImageFormatterTraits {
public:
    static std::string name();
};

template<> std::string DecoratedImageFormatterTraits<std::uint16_t>::name() {
    static std::string name = "DecoratedImageU";
    return name;
}
template<> std::string DecoratedImageFormatterTraits<int>::name() {
    static std::string name = "DecoratedImageI";
    return name;
}
template<> std::string DecoratedImageFormatterTraits<float>::name() {
    static std::string name = "DecoratedImageF";
    return name;
}
template<> std::string DecoratedImageFormatterTraits<double>::name() {
    static std::string name = "DecoratedImageD";
    return name;
}
template<> std::string DecoratedImageFormatterTraits<std::uint64_t>::name() {
    static std::string name = "DecoratedImageL";
    return name;
}

template <typename ImagePixelT>
lsst::daf::persistence::FormatterRegistration DecoratedImageFormatter<ImagePixelT>::registration(
    DecoratedImageFormatterTraits<ImagePixelT>::name(),
    typeid(DecoratedImage<ImagePixelT>),
    createInstance);

template <typename ImagePixelT>
DecoratedImageFormatter<ImagePixelT>::DecoratedImageFormatter(
        lsst::pex::policy::Policy::Ptr
                                                             )
    : lsst::daf::persistence::Formatter(typeid(this))
{
}

template <typename ImagePixelT>
DecoratedImageFormatter<ImagePixelT>::~DecoratedImageFormatter(void) {
}

template <typename ImagePixelT>
void DecoratedImageFormatter<ImagePixelT>::write(
        Persistable const* persistable,
        Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr
                                                )
{
    execTrace("DecoratedImageFormatter write start");
    DecoratedImage<ImagePixelT> const* ip = dynamic_cast<DecoratedImage<ImagePixelT> const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-DecoratedImage");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("DecoratedImageFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("DecoratedImageFormatter write end");
        return;
    } else if (typeid(*storage) == typeid(XmlStorage)) {
        execTrace("DecoratedImageFormatter write XmlStorage");
        XmlStorage* boost = dynamic_cast<XmlStorage*>(storage.get());
        boost->getOArchive() & make_nvp("img", *ip);
        execTrace("DecoratedImageFormatter write end");
        return;
    } else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("DecoratedImageFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        typedef DecoratedImage<ImagePixelT> DecoratedImage;

        ip->writeFits(fits->getPath());
        // \todo Do something with these fields?
        // int _X0;
        // int _Y0;
        execTrace("DecoratedImageFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized Storage for DecoratedImage");
}

template <typename ImagePixelT>
Persistable* DecoratedImageFormatter<ImagePixelT>::read(
        Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr
                                                       )
{
    execTrace("DecoratedImageFormatter read start");
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("DecoratedImageFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        DecoratedImage<ImagePixelT>* ip = new DecoratedImage<ImagePixelT>;
        boost->getIArchive() & *ip;
        execTrace("DecoratedImageFormatter read end");
        return ip;
    } else if (typeid(*storage) == typeid(XmlStorage)) {
        execTrace("DecoratedImageFormatter read XmlStorage");
        XmlStorage* boost = dynamic_cast<XmlStorage*>(storage.get());
        DecoratedImage<ImagePixelT>* ip = new DecoratedImage<ImagePixelT>;
        boost->getIArchive() & make_nvp("img", *ip);
        execTrace("DecoratedImageFormatter read end");
        return ip;
    } else if(typeid(*storage) == typeid(FitsStorage)) {

        execTrace("DecoratedImageFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        
        DecoratedImage<ImagePixelT>* ip = new DecoratedImage<ImagePixelT>(fits->getPath(), fits->getHdu());
        // \todo Do something with these fields?
        // int _X0;
        // int _Y0;
        execTrace("DecoratedImageFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized Storage for DecoratedImage");
}

template <typename ImagePixelT>
void DecoratedImageFormatter<ImagePixelT>::update(
        lsst::daf::base::Persistable*,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
                                                 )
{
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unexpected call to update for DecoratedImage");
}

template <typename ImagePixelT> template <class Archive>
void DecoratedImageFormatter<ImagePixelT>::delegateSerialize(
        Archive&,
        int const,
        Persistable* persistable
                                                            )
{
    execTrace("DecoratedImageFormatter delegateSerialize start");
    DecoratedImage<ImagePixelT>* ip = dynamic_cast<DecoratedImage<ImagePixelT>*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-DecoratedImage");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "DecoratedImage serialization not yet implemented");
}

template <typename ImagePixelT>
lsst::daf::persistence::Formatter::Ptr DecoratedImageFormatter<ImagePixelT>::createInstance(
        lsst::pex::policy::Policy::Ptr policy
                                                                                           )
{
    return lsst::daf::persistence::Formatter::Ptr(new DecoratedImageFormatter<ImagePixelT>(policy));
}

#define InstantiateFormatter(ImagePixelT) \
    template class DecoratedImageFormatter<ImagePixelT >; \
    template void DecoratedImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::text_oarchive&, int const, Persistable*); \
    template void DecoratedImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::text_iarchive&, int const, Persistable*); \
    template void DecoratedImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::binary_oarchive&, int const, Persistable*); \
    template void DecoratedImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::binary_iarchive&, int const, Persistable*);

InstantiateFormatter(std::uint16_t);
InstantiateFormatter(int);
InstantiateFormatter(float);
InstantiateFormatter(double);
InstantiateFormatter(std::uint64_t);

#undef InstantiateFormatter


}}} // namespace lsst::afw::formatters
