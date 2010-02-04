#ifndef LSST_AFW_MATH_LOCAL_KERNEL_H
#define LSST_AFW_MATH_LOCAL_KERNEL_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include "lsst/afw/math/FourierCutout.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math { 

/** 
 *  All model convolution goes through a visitor that defines one of
 *  the (we hope very few) different ways to convolve a model.  We can
 *  imagine no more than five of these at present (only one of which
 *  will see action in DC3b).
 *
 *  Both a model and PSF will be asked which convolution methods they
 *  support; a PSF may support many, but a particular model class will
 *  only ever support one.  The free function setModelPsf, defined
 *  below, acts as a kind of visitor factory, checking if any of the
 *  convolution methods supported by an instance of a psf are
 *  compatible with a model, and asking the Psf to create a
 *  convolution visitor if possible.  This visitor is then passed to
 *  the Model, providing it all the information it needs to convolve
 *  itself.
 *
 *  The "local linear combination of functions" discussed at the last
 *  meeting is a combination of special cases of two of these visitors
 *  (IMAGE and FOURIER).  A linear combination is less fundamental
 *  than we thought (we actually just need to compute the derivative
 *  of a kernel with respect to its local parameters).  Splitting
 *  regular image convolution from Fourier-space convolution also
 *  seemed like a good idea.
 */
class LocalKernel {
public:
    typedef Kernel::Pixel Pixel; 
    typedef boost::shared_ptr<LocalKernel> Ptr;
    typedef boost::shared_ptr<LocalKernel const> ConstPtr;
    typedef boost::shared_ptr<Eigen::MatrixXd> CovariancePtr;
   
    virtual ~LocalKernel(){}

    /**
     * Retrieve the number of kernel Parameters
     */
    virtual int getNParameters() const = 0;
  
    /**
     * Retrieve the kernel Parameters
     */
    virtual std::vector<double> const & getParameters() const = 0;
    /**
     * Determine if the LocalKernel has derivatives with respect to each of its
     * kernel parameters
     */
    virtual bool hasDerivatives() const = 0;

    /**
     *  This allows Psf to set a covariance matrix for a visitor after
     *  it has been constructed by a Kernel (which has no knowledge of
     *  uncertainty, and hence initializes the covariance to zero).
     */
    void setCovariance(CovariancePtr covariance) {
        int nRows = getNParameters();
        int nCols = nRows - 1;
        if(covariance->rows() != nRows && covariance->cols() != nCols) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, 
                    (boost::format("Covariance dimensions must be (%1%, %2%)") 
                    % nRows % nCols).str()
            );
        }

        _covariance = covariance;
    }

    /**
     * @brief Retrieve a shared_ptr to the covariance matrix
     */
    CovariancePtr getCovariance() const {
        return _covariance;
    }

private:
    CovariancePtr _covariance;    
};

/**
 *  LocalKernel corresponding to models that construct their
 *  unconvolved selves as a regular images and convolve in real space.
 */
class ImageLocalKernel : public LocalKernel {
public:
    typedef boost::shared_ptr<ImageLocalKernel> Ptr;
    typedef boost::shared_ptr<ImageLocalKernel const> ConstPtr;

    typedef lsst::afw::image::Image<Pixel> Image;
    typedef std::vector<boost::shared_ptr<Image> > ImagePtrList;
   
    ImageLocalKernel(
        lsst::afw::geom::Point2I const & center,
        std::vector<double> const & parameters, 
        Image::Ptr const & image, 
        ImagePtrList const & derivatives = ImagePtrList()
    ) : _center(center),
        _parameters(parameters), 
        _image(image), 
        _derivatives(derivatives) 
    {
        if(!image) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "NULL Kernel Image::Ptr.");
        }
        else if(image->getHeight() == 0 || image->getWidth() == 0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                "Kerenl image has zero size");
        }
        validateDerivatives(); 
    }
        
    virtual ~ImageLocalKernel() { }

    virtual int getNParameters() const {return _parameters.size();}
    virtual std::vector<double> const & getParameters() const {
        return _parameters;
    }

    virtual bool hasDerivatives() const {return !_derivatives.empty();}

    /**
     * @brief Retrieve an image of the kernel.
     */
    Image::Ptr getImage() const {return _image;}
    
    /**
     *  Retrieve an image list of the derivatives of the kernel 
     *  
     *  These are the derivatives of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    ImagePtrList getDerivatives() const {
        return _derivatives;
    }

    /**
     * @brief Retrieve the height of the kernel
     */
    int getHeight() const {return _image->getHeight();}
    
    /**
     *@brief Retrieve the width of the kernel
     */
    int getWidth() const {return _image->getWidth();}
   
    /**
     * Retrive the Kernel center on the image
     */
    lsst::afw::geom::Point2I const & getCenter() const {return _center;}
private:
    void validateDerivatives() {
        unsigned int nDerivatives = _derivatives.size();
        if(_parameters.size() != nDerivatives && nDerivatives != 0 ) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "Parameter list must have same size as list of derivatives");
        }

        //remove any null pointers, and zero sized images to the derivative list
        ImagePtrList::iterator i(_derivatives.begin());
        ImagePtrList::iterator end(_derivatives.end());
        Image::Ptr derivative;
        for( ; i != end; ++i) {
            derivative = *i;
            if(!derivative) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    "NULL pointer in vector of derivatives"
                );
            }            
            else if(derivative->getWidth() == 0 || 
                derivative->getHeight() == 0
            ) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    "Zero size image in vector of derivatives"
                );
            }
            else if(derivative->getWidth() != getWidth() || 
                derivative->getHeight() != getHeight()
            ) {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::InvalidParameterException,
                    "derivative dimensions do not match the image dimensions"
                );
            }
        }    
    }
    
    lsst::afw::geom::Point2I _center;
    std::vector<double> _parameters;
    Image::Ptr _image;
    ImagePtrList _derivatives;
};

class FourierLocalKernel : public LocalKernel {
public:
    typedef boost::shared_ptr<FourierLocalKernel> Ptr;
    typedef boost::shared_ptr<FourierLocalKernel const> ConstPtr;

    /**
     * @brief retrieve the image of the kernel evaluated in fourier space, 
     *      shifted such that the center of the kernel is at the origin
     */
    virtual FourierCutout::Ptr getFourierImage() const = 0;        

    /** 
     * @brief Retrieve a list of kernel derivatives evaluated in fourier space
     *
     * These are the derivatives of the kernel w.r.t. its local parameters
     * The returned list of fourier images are shifted such that the center
     * of the kernel is at the origin
     */
    virtual std::vector<FourierCutout::Ptr> getFourierDerivatives() const {
        return std::vector<FourierCutout::Ptr>();
    }; 

    virtual bool hasDerivatives() const {return false;}
    /**
     * @brief Set the dimensions of the image of the fourier-space evaluation
     * 
     * Ths operation may ask to generate a fourier image that is larger than the
     * kernel. However, calling setDimensions with dimensions smaller than the 
     * kernel's native size, will result in an exception.
     *
     * When transforming to dimensions, the extra rows/cols are zeroed, and act
     * as padding
     *
     * @param width requested width of the fourier-space image
     * @param height requested height of the fourier-space image
     * @param normalize request that the fourier-space image be normalized
     * @throws lsst::pex::exceptions if the dimensions are smaller than the
     *      kernels naitive size
     */
    virtual void setDimensions(
        int const & width, int const & height, 
        bool normalize = false
    ) = 0;


    virtual ~FourierLocalKernel() {}
};

/**
 *  FourierLocalKernel subclass constructed by performing an fft of the kernel 
 *  image
 */
class FftLocalKernel : public FourierLocalKernel{
public:
    typedef boost::shared_ptr<FftLocalKernel> Ptr;
    typedef boost::shared_ptr<FftLocalKernel const> ConstPtr;
    
    typedef lsst::afw::image::Image<Pixel> Image;    
    typedef std::vector<boost::shared_ptr<Image> > ImagePtrList;
    
    explicit FftLocalKernel(
        lsst::afw::geom::Point2I const & center,
        std::vector<double> const & parameters, 
        Image::Ptr const & image, 
        ImagePtrList const & derivatives = ImagePtrList()
    ) : _imageKernel(center, parameters, image, derivatives) 
    {}
    
    explicit FftLocalKernel(
        ImageLocalKernel const & imageKernel
    ) : _imageKernel(imageKernel) 
    {}

    /**
     * @brief Retrieve the real-space image representation of the kernel
     */
    Image::Ptr getImage() const {return _imageKernel.getImage();} 

    /**
     *  @brief Retrieve an image list of the derivatives of the kernel 
     *  
     *  These are the derivatives of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    ImagePtrList getDerivatives() const {return _imageKernel.getDerivatives();}

    virtual FourierCutout::Ptr getFourierImage() const;        
    virtual std::vector<FourierCutout::Ptr> getFourierDerivatives() const; 
    virtual ~FftLocalKernel(){}

    virtual int getNParameters() const {return _imageKernel.getNParameters();}
    virtual bool hasDerivatives() const {return _imageKernel.hasDerivatives();}
    virtual std::vector<double> const & getParameters() const {
        return _imageKernel.getParameters();
    }
    int getImageWidth() const {return _imageKernel.getWidth();}
    int getImageHeight() const {return _imageKernel.getHeight();}

    virtual void setDimensions(
        int const & width, int const & height, 
        bool normalize = false
    );

protected:
    ImageLocalKernel _imageKernel;
    FourierCutoutStack _fourierStack;  

private:
    void copyImage(Pixel * dest, Image::Ptr image, int const & destWidth);
    void fillImageStack(
        Pixel * imageStack, 
        int const & imageSize, int const & imageWidth
    );    
};

#if 0

/**
 *  LocalKernel corresponding to models that are sums of
 *  Gaussians and want to convolve themselves with Kernels that are
 *  sums of Gaussians.
 *
 *  We probably won't implement this until it's actually needed; it's
 *  only necessary for Gaussian-based models, not Gaussian-based PSFs.
 */
class MultiGaussianLocalKernel : public LocalKernel {
public:
    
    /**
     *  Get the local parameters of the kernel.
     *
     *  In this case that would be a flattened sequence of
     *  (amplitude,xx,yy,xy) tuples or something.
     */
    void getParameters(std::vector<double> & parameters);

    /**
     *  Get the covariance of the local parameters of the kernel.
     */
    void getCovariance(CovarianceMatrix & covariance);

    virtual void visit(Model & model) { model.accept(*this); }
};


/**
 *  LocalKernel corresponding to models that are represented
 *  in shapelet space.
 *
 *  We probably won't implement this until it's actually needed; it's
 *  only necessary for shapelet based models, not shapelet-based PSFs.
 */
class ShapeletLocalKernel : public LocalKernel {
public:

    /**
     *  Get the scale factor of the shapelet expansion.
     */
    double getScale() const;

    /**
     *  Get the local parameters of the kernel.
     *
     *  In this case that would be the shapelet coefficients.
     */
    void getParameters(std::vector<double> & parameters);

    /**
     *  Get the covariance of the local parameters of the kernel.
     */
    void getCovariance(CovarianceMatrix & covariance);

    virtual void visit(Model & model) { model.accept(*this); }
};
#endif

}}} //end namespace lsst::afw::math

#endif
