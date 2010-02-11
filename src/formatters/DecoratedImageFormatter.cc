// -*- lsst-c++ -*-

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

#include "boost/scoped_ptr.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include "boost/serialization/nvp.hpp"

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
    static std::string name;
};

template<> std::string DecoratedImageFormatterTraits<boost::uint16_t>::name("DecoratedImageU");
template<> std::string DecoratedImageFormatterTraits<int>::name("DecoratedImageI");
template<> std::string DecoratedImageFormatterTraits<float>::name("DecoratedImageF");
template<> std::string DecoratedImageFormatterTraits<double>::name("DecoratedImageD");


template <typename ImagePixelT>
lsst::daf::persistence::FormatterRegistration DecoratedImageFormatter<ImagePixelT>::registration(
    DecoratedImageFormatterTraits<ImagePixelT>::name,
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
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Persisting non-DecoratedImage");
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
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
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
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                      "Unrecognized Storage for DecoratedImage");
}

template <typename ImagePixelT>
void DecoratedImageFormatter<ImagePixelT>::update(
        lsst::daf::base::Persistable*,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
                                                 )
{
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
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
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Serializing non-DecoratedImage");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException,
                      "DecoratedImage serialization not yet implemented");
}

template <typename ImagePixelT>
lsst::daf::persistence::Formatter::Ptr DecoratedImageFormatter<ImagePixelT>::createInstance(
        lsst::pex::policy::Policy::Ptr policy
                                                                                           )
{
    return lsst::daf::persistence::Formatter::Ptr(new DecoratedImageFormatter<ImagePixelT>(policy));
}

template class DecoratedImageFormatter<boost::uint16_t>;
template class DecoratedImageFormatter<int>;
template class DecoratedImageFormatter<float>;
template class DecoratedImageFormatter<double>;

}}} // namespace lsst::afw::formatters
