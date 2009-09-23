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
    typedef double PixelT; 
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

    virtual void visit(Convolvable & model) = 0;

    /**
     *  This allows Psf to set a covariance matrix for a visitor after
     *  it has been constructed by a Kernel (which has no knowledge of
     *  uncertainty, and hence initializes the covariance to zero).
     */
    void setCovarianceMatrix(CovariancePtr covariance) {
        _covariance = covariance;
    }
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

    typedef lsst::afw::image::Image<PixelT> Image;
    typedef std::vector<Image::Ptr> ImagePtrList;
   
    ImageConvolutionVisitor(Image::Ptr image) : _image(image) {}
    ImageConvolutionVisitor(Image::Ptr image, ImagePtrList derivative) : 
        _image(image), 
        _derivativeImageList(derivative)
    {}
        
    virtual ~ImageConvolutionVisitor() {
        _image.reset();
        ImagePtrList::iterator const end(_derivativeImageList.end());
        ImagePtrList::iterator i(_derivativeImageList.begin());
        for (; i != end; ++i) {
            i->reset();
        }
    }
    /**
     *  Create an image of the kernel.
     */
    Image::Ptr getImage() const {return _image;}

    /**
     *  Create images of the derivative of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    ImagePtrList getDerivative() const {
        return _derivativeImageList;
    }
    int getNParameters() const {return _derivativeImageList.size();}

    virtual void visit(Convolvable & model) { model.convolve(*this); }
private:
    friend class FourierConvolutionVisitor;
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
    
    typedef FourierCutoutStack::FourierCutoutVector FourierCutoutVector;
    typedef lsst::afw::image::Image<PixelT> Image;    
    typedef std::vector<Image::Ptr> ImagePtrList;
    
    explicit FourierConvolutionVisitor(Image::Ptr & image, ImagePtrList derivative) :
        _imageVisitor(image, derivative) {
    }
    explicit FourierConvolutionVisitor(Image::Ptr image) :
        _imageVisitor(image) {           
    }
    explicit FourierConvolutionVisitor(ImageConvolutionVisitor const & imageVisitor) :
        _imageVisitor(imageVisitor) {
    }
    
    /**
     *  Create a Fourier image of the kernel.
     */
    FourierCutout::Ptr getFourierImage() const {return _fourierStack.getCutout(0);}

    /**
     *  Create FourierCutout images of the derivative of the kernel w.r.t. its local
     *  parameters (evaluated at the values of those parameters).
     *
     */
    FourierCutoutVector getFourierDerivative() const {
        if(_fourierStack.getStackDepth() > 1)
            return _fourierStack.getCutoutVector(1);
        else return FourierCutoutVector();
    }
    virtual ~FourierConvolutionVisitor(){}

    int getNParameters() const {return _imageVisitor.getNParameters();}
    virtual void visit(Convolvable & model) { model.convolve(*this); }

    void fft(std::pair<int,int> const & imageDimensions);
    
private:
    void copyImage(PixelT * dest, Image::Ptr image, int const & destWidth);
    void fillImageStack(PixelT * imageStack, std::pair<int,int> const & imageDimensions);
    
protected:
    ImageConvolutionVisitor _imageVisitor;
    FourierCutoutStack _fourierStack; 
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
