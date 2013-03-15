// -*- lsst-c++ -*-

#if !defined(LSST_DETECTION_DoubleGaussianPsf_H)
#define LSST_DETECTION_DoubleGaussianPsf_H
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
        class DoubleGaussianPsf;
    }
    namespace math {
        class Kernel;
    }
}}

namespace boost {
namespace serialization {
    template <class Archive>
    void save_construct_data(
        Archive& ar, lsst::afw::detection::DoubleGaussianPsf const* p,
        unsigned int const file_version);
}}

namespace lsst { namespace afw { namespace detection {
            
/*!
 * \brief Represent a Psf as a circularly symmetrical double Gaussian
 */
class DoubleGaussianPsf : public afw::table::io::PersistableFacade<DoubleGaussianPsf>, public KernelPsf {
public:
    typedef PTR(DoubleGaussianPsf) Ptr;
    typedef CONST_PTR(DoubleGaussianPsf) ConstPtr;

    /**
     * Constructor for a DoubleGaussianPsf
     */
    DoubleGaussianPsf(int width,                         ///< Number of columns in realisations of Psf
          int height,                        ///< Number of rows in realisations of Psf
          double sigma1,                     ///< Width of inner Gaussian
          double sigma2=0.0,                     ///< Width of outer Gaussian
          double b=0.0                   ///< Central amplitude of outer Gaussian (inner amplitude == 1)
    );

    virtual PTR(Psf) clone() const {
        return boost::make_shared<DoubleGaussianPsf>(
            getKernel()->getWidth(),
            getKernel()->getHeight(),
            _sigma1, _sigma2, _b
        );
    }

    double getSigma1() const { return _sigma1; }

    double getSigma2() const { return _sigma2; }

    double getB() const { return _b; }

    virtual bool isPersistable() const { return true; }

protected:

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:
    double _sigma1;                     ///< Width of inner Gaussian
    double _sigma2;                     ///< Width of outer Gaussian
    double _b;                          ///< Central amplitude of outer Gaussian (inner amplitude == 1)

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive&, unsigned int const) {
        boost::serialization::void_cast_register<DoubleGaussianPsf, Psf>(
            static_cast<DoubleGaussianPsf*>(0), static_cast<Psf*>(0));
    }
    template <class Archive>
    friend void boost::serialization::save_construct_data(
            Archive& ar, DoubleGaussianPsf const* p, unsigned int const file_version);
};

}}}

namespace boost {
namespace serialization {

template <class Archive>
inline void save_construct_data(
        Archive& ar, lsst::afw::detection::DoubleGaussianPsf const* p,
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
        Archive& ar, lsst::afw::detection::DoubleGaussianPsf* p,
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
    ::new(p) lsst::afw::detection::DoubleGaussianPsf(width, height, sigma1, sigma2, b);
}

}}


#endif
