#if !defined(LSST_DETECTION_DGPSF_H)
#define LSST_DETECTION_DGPSF_H
//!
// Describe an image's PSF
//
#include "lsst/base.h"
#include "lsst/afw/detection/Psf.h"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/void_cast.hpp"

// Forward declarations

namespace lsst { namespace afw {
    namespace detection {
        class dgPsf;
    }
    namespace math {
        class Kernel;
    }
}}

namespace boost {
namespace serialization {
    template <class Archive>
    void save_construct_data(
        Archive& ar, lsst::afw::detection::dgPsf const* p,
        unsigned int const file_version);
}}

namespace lsst { namespace afw { namespace detection {
            
/*!
 * \brief Represent a Psf as a circularly symmetrical double Gaussian
 */
class dgPsf : public KernelPsf {
public:
    typedef PTR(dgPsf) Ptr;
    typedef CONST_PTR(dgPsf) ConstPtr;

    /**
     * @brief constructors for a dgPsf
     *
     * Parameters:
     */
    explicit dgPsf(int width, int height, double sigma1, double sigma2=1, double b=0);
private:
    double _sigma1;                     ///< Width of inner Gaussian
    double _sigma2;                     ///< Width of outer Gaussian
    double _b;                          ///< Central amplitude of outer Gaussian (inner amplitude == 1)

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive&, unsigned int const) {
        boost::serialization::void_cast_register<dgPsf, Psf>(
            static_cast<dgPsf*>(0), static_cast<Psf*>(0));
    }
    template <class Archive>
    friend void boost::serialization::save_construct_data(
            Archive& ar, dgPsf const* p, unsigned int const file_version);
};

}}}

namespace boost {
namespace serialization {

template <class Archive>
inline void save_construct_data(
        Archive& ar, lsst::afw::detection::dgPsf const* p,
        unsigned int const
                               )
{
    int width = p->getKernel()->getWidth();
    int height = p->getKernel()->getHeight();
    ar << make_nvp("width", width);
    ar << make_nvp("height", height);
    ar << make_nvp("sigma1", p->_sigma1);
    ar << make_nvp("sigma2", p->_sigma2);
    ar << make_nvp("b", p->_b);
}

template <class Archive>
inline void load_construct_data(
        Archive& ar, lsst::afw::detection::dgPsf* p,
        unsigned int const
                               )
{
    int width;
    int height;
    double sigma1;
    double sigma2;
    double b;
    ar >> make_nvp("width", width);
    ar >> make_nvp("height", height);
    ar >> make_nvp("sigma1", sigma1);
    ar >> make_nvp("sigma2", sigma2);
    ar >> make_nvp("b", b);
    ::new(p) lsst::afw::detection::dgPsf(width, height, sigma1, sigma2, b);
}

}}


#endif
