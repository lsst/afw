#ifndef LSST_AFW_MATH_CONVOLUTION_VISITOR_H
#define LSST_AFW_MATH_CONVOLUTION_VISITOR_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <lsst/afw/math/FourierCutout.h>
#include <lsst/afw/image/Image.h>

namespace lsst {
namespace afw {
namespace math { 

class ImageConvolutionVisitor;
class FourierConvolutionVisitor;
//class MultiGaussianConvolutionVisitor;
//class ShapeletConvolutionVisitor;

class Convolvable {
public:
    typedef boost::shared_ptr<Convolvable> Ptr;
    typedef boost::shared_ptr<Convolvable const> ConstPtr;

    virtual ~Convolvable(){}
    
    virtual void convolve(ImageConvolutionVisitor const &) const {assert(false);}
    virtual void convolve(FourierConvolutionVisitor const &) const {assert(false);}
    //virtual void convolve(MultiGaussianConvolutionVisitor const &) const {assert(false);}
    //virtual void convolve(ShapeletConvolutionVisitor const &) const {assert(false);}        
};

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
class ConvolutionVisitor {
public:
    typedef double Pixel; 
    typedef boost::shared_ptr<Eigen::MatrixXd> CovariancePtr;
    typedef boost::shared_ptr<ConvolutionVisitor> Ptr;
    typedef boost::shared_ptr<ConvolutionVisitor const> ConstPtr;

    virtual ~ConvolutionVisitor(){}

    /**
     * Note: the number of types of convolution is expected to remain small.
     * We may find the types defined below are already more than we will 
     * ever need.    
     */
    enum TypeFlag { 
        IMAGE=0x1, 
        FOURIER=0x2, 
        //MULTIGAUSSIAN=0x4, 
        //SHAPELET=0x8 
    };

    /**
     * @brief visit a convolvable model
     *
     * This is the first step of the double-dispatch.
     * Every subclass of ConvolutionVisitor must implement this method
     */
    virtual void visit(Convolvable & model) = 0;

    /**
     * @brief Retrieve the number of kernel Parameters
     */
    virtual int getNParameters() const = 0;
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
 *  ConvolutionVisitor corresponding to models that construct their
 *  unconvolved selves as a regular images and convolve in real space.
 */
class ImageConvolutionVisitor : public ConvolutionVisitor {
public:
    typedef boost::shared_ptr<ImageConvolutionVisitor> Ptr;
    typedef boost::shared_ptr<ImageConvolutionVisitor const> ConstPtr;

    typedef lsst::afw::image::Image<Pixel> Image;
    typedef std::vector<boost::shared_ptr<Image> > ImagePtrList;
   
    ImageConvolutionVisitor(
            std::pair<int, int> center,
            std::vector<double> const & parameterList, 
            Image::Ptr image, 
            ImagePtrList derivativeImageList = ImagePtrList()
    ) : _center(center), _parameterList(parameterList), _image(image), _derivativeImageList(derivativeImageList) {
        if(!image) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "NULL Kernel Image::Ptr.");
        }
        else if(image->getHeight() == 0 || image->getWidth() == 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "Kerenl image has zero size");
        }
        validateDerivatives(); 

        unsigned int nDerivatives = _derivativeImageList.size();
        if(parameterList.size() != nDerivatives && nDerivatives != 0 ) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                "Parameter list must have same size as list of derivatives");
        }
    }
        
    virtual ~ImageConvolutionVisitor() { }

    /**
     * @brief Retrieve the center point of the Kernel
     */
    std::pair<int, int> getCenter() const {return _center;}

    /**
     * @brief Retrieve the number of Kernel Parameters
     */
    virtual int getNParameters() const {return _parameterList.size();}
   
    /**
     * @brief Determine if visitor has list of derivative images
     */
    bool hasDerivatives() const {return !_derivativeImageList.empty();}

    /**
     * @brief Retrieve an image of the kernel.
     */
    Image::Ptr getImage() const {return _image;}
    
    /**
     *  @brief Retrieve an image list of the derivatives of the kernel 
     *  
     *  These are the derivatives of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    ImagePtrList getDerivativeImageList() const {
        return _derivativeImageList;
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
     * @brief Retrieve the kernel parameters
     * A ConvolutionVisitor represents a 
     */
    std::vector<double> const & getParameterList() const {return _parameterList;}

    virtual void visit(Convolvable & model) { model.convolve(*this); }
private:
    void validateDerivatives() {
        //remove any null pointers, and zero sized images to the derivative list
        ImagePtrList::iterator i(_derivativeImageList.begin());
        ImagePtrList::iterator end(_derivativeImageList.end());
        Image::Ptr derivative;
        for( ; i != end; ++i) {
            derivative = *i;
            if(!derivative) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                    "Vector of kernel derivative images contains null pointer(s)");
            }            
            else if(derivative->getWidth() == 0 || derivative->getHeight() == 0) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                    "Vector of kernel derivative images contains zero size image(s)");
            }
            else if(derivative->getWidth() != getWidth() || derivative->getHeight() != getHeight()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                    "kernel derivative image(s) do not match the dimensions of the kernel image");
            }
        }    
    }
    
    std::pair<int, int> _center;
    std::vector<double> _parameterList;
    Image::Ptr _image;
    ImagePtrList _derivativeImageList;

};

/**
 *  ConvolutionVisitor corresponding to models that construct
 *  their unconvolved selves in Fourier space.
 */
class FourierConvolutionVisitor : public ConvolutionVisitor {
public:
    typedef boost::shared_ptr<FourierConvolutionVisitor> Ptr;
    typedef boost::shared_ptr<FourierConvolutionVisitor const> ConstPtr;
    
    typedef lsst::afw::image::Image<Pixel> Image;    
    typedef std::vector<boost::shared_ptr<Image> > ImagePtrList;
    
    explicit FourierConvolutionVisitor(
            std::pair<int, int> const & center,
            std::vector<double> const & parameterList, 
            Image::Ptr & image, 
            ImagePtrList const & derivative = ImagePtrList()
    ) : _imageVisitor(center, parameterList, image, derivative) 
    {}
    
    explicit FourierConvolutionVisitor(ImageConvolutionVisitor const & imageVisitor) 
            : _imageVisitor(imageVisitor) 
    { }

    /**
     * @brief Retrieve the real-space image representation of the kernel
     */
    Image::Ptr getImage() const {return _imageVisitor.getImage();} 

    /**
     *  @brief Retrieve an image list of the derivatives of the kernel 
     *  
     *  These are the derivatives of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    ImagePtrList getDerivativeImageList() const {return _imageVisitor.getDerivativeImageList();}

    /**
     * @brief retrieve the fourier transform of the kernel image, shifted
     *      such that the center of the kernel is at the origin
     */
    FourierCutout::Ptr getFourierImage() const;        

    /** 
     * @brief Retrieve a list of fourier-transformed kernel derivatives
     *
     * These are the derivatives of the kernel w.r.t. its local parameters
     * The returned list of fourier images are shifted such that the center
     * of the kernel is at the origin
     */
    std::vector<FourierCutout::Ptr> getFourierDerivativeImageList() const; 
    
    virtual ~FourierConvolutionVisitor(){}

    /**
     * @brief Retrieve the center of the kernel
     */
    std::pair<int, int> getCenter() const {return _imageVisitor.getCenter();}

    /**
     * @brief retrieve the number of kernel parameters
     */
    virtual int getNParameters() const {return _imageVisitor.getNParameters();}

    /**
     * @brief determine if the visitor has list of kernel derivatives
     */
    bool hasDerivatives() const {return _imageVisitor.hasDerivatives();}

    /**
     * @brief retrieve kernel's local parameters
     */
    std::vector<double> getParameterList() const {return _imageVisitor.getParameterList();}

    /**
     * @brief Retrieve kernel's width
     */
    int getWidth() const {return _imageVisitor.getWidth();}

    /**
     * @brief Retrieve kernel's height
     */
    int getHeight() const {return _imageVisitor.getHeight();}

    virtual void visit(Convolvable & model) { model.convolve(*this); }

    /**
     * @brief fourier transform the kernel image
     * 
     * the fft operation may ask to generate a fourier image that is larger than the kernel
     * However, calling fft with dimensions smaller than the kernel's, will result in 
     *   an lsst::pex::exceptions::InvalidParameterException being thrown
     *
     * When transforming to dimensions, the extra rows/cols are zeroed, act as padding
     */
    void fft(int const & width, int const & height, bool normalizeFft = false);

protected:
    ImageConvolutionVisitor _imageVisitor;
    FourierCutoutStack _fourierStack;  

private:
    void copyImage(Pixel * dest, Image::Ptr image, int const & destWidth);
    void fillImageStack(Pixel * imageStack, int const & imageSize, int const & imageWidth);    
};

#if 0

/**
 *  ConvolutionVisitor corresponding to models that are sums of
 *  Gaussians and want to convolve themselves with Kernels that are
 *  sums of Gaussians.
 *
 *  We probably won't implement this until it's actually needed; it's
 *  only necessary for Gaussian-based models, not Gaussian-based PSFs.
 */
class MultiGaussianConvolutionVisitor : public ConvolutionVisitor {
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
 *  ConvolutionVisitor corresponding to models that are represented
 *  in shapelet space.
 *
 *  We probably won't implement this until it's actually needed; it's
 *  only necessary for shapelet based models, not shapelet-based PSFs.
 */
class ShapeletConvolutionVisitor : public ConvolutionVisitor {
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
