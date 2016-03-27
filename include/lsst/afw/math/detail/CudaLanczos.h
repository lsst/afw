// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Declaration of a GPU kernel for image warping
 *        and declarations of requred datatypes
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifdef NVCC_COMPILING
    #define CPU_GPU __device__ __host__
#else
    #define CPU_GPU
#endif

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

typedef lsst::afw::image::VariancePixel VarPixel;
typedef lsst::afw::image::MaskPixel     MskPixel;

int const SIZE_X_WARPING_BLOCK=16;
int const SIZE_Y_WARPING_BLOCK=16;
int const SIZE_MAX_WARPING_KERNEL=100;

/// Simple 2D point (suitable for use on a GPU)
struct SPoint2
{
    double x;
    double y;

    CPU_GPU SPoint2(double par_x, double par_y) : x(par_x), y(par_y) {}

    #ifndef NVCC_COMPILING
    SPoint2(lsst::afw::geom::Point2D p) : x(p.getX()), y(p.getY()) {}
    #endif
};

/// Simple 2D vector (suitable for use on a GPU)
struct SVec2
{
    double x;
    double y;

    CPU_GPU SVec2(double par_x, double par_y) : x(par_x), y(par_y) {}
    CPU_GPU SVec2(SPoint2 a, SPoint2 b) : x(b.x-a.x), y(b.y-a.y) {}

    #ifndef NVCC_COMPILING
    SVec2(lsst::afw::geom::Extent2D e) : x(e.getX()), y(e.getY()) {}
    #endif
};

CPU_GPU inline SVec2 VecAdd(SVec2 a, SVec2 b)
{
    return SVec2(a.x+b.x, a.y+b.y);
}
CPU_GPU inline SVec2 VecSub(SVec2 a, SVec2 b)
{
    return SVec2(a.x-b.x, a.y-b.y);
}
CPU_GPU inline SVec2 VecMul(SVec2 v, double m)
{
    return SVec2(m*v.x, m*v.y);
}
CPU_GPU inline SPoint2 MovePoint(SPoint2 p, SVec2 v)
{
    return SPoint2(p.x+v.x, p.y+v.y);
}


/// defines a 2D range of integer values begX <= x < endX, begY <= y < endY
struct SBox2I
{
    int begX;
    int begY;
    int endX;
    int endY;

    SBox2I() {};

    CPU_GPU SBox2I(int par_begX, int par_begY, int par_endX, int par_endY)
        : begX(par_begX), begY(par_begY), endX(par_endX), endY(par_endY) {}

    CPU_GPU bool isInsideBox(gpu::SPoint2 p)
    {
        return      begX <= p.x && p.x < endX
                 && begY <= p.y && p.y < endY;
    }
};

/** Used for linear interpolation of a 2D function Z -> R*R

    This class just defines a line which can be used to interpolate
    a segment of a function.

    It does not specify which part of the function is interpolated.
*/
struct LinearInterp
{
    SPoint2 o;    /// defines the value at the origin
    SVec2 deltaX; /// difference of neighboring values of the function (the gradient)

    CPU_GPU LinearInterp(SPoint2 par_o, SVec2 par_deltaX) : o(par_o), deltaX(par_deltaX) {};

    /// Calculates a value of the interpolation function at subX
    CPU_GPU SPoint2 Interpolate(int subX)
    {
        return MovePoint(o, VecMul(deltaX,subX) );
    }
};


/** Used for bilinear interpolation of a 2D function Z*Z -> R*R

    This class just defines a 2D surface which can be used to interpolate
    a segment of a 2D function.

    It does not specify which segment of the function is interpolated.
*/
struct BilinearInterp
{
    SPoint2 o;  /// defines the value at origin
    SVec2 d0X;  /// difference of neighboring values in the first row (the gradient of a line at y=0)
    /// difference of difference of neighboring values in two neighbouring rows (diff. of gradients at x=0 and x=1)
    SVec2 ddX;
    SVec2 deltaY; /// difference of neighboring values in the first column (gradient of a line at x=0)

    BilinearInterp() : o(0,0), d0X(0,0), ddX(0,0), deltaY(0,0) {};

    CPU_GPU BilinearInterp(SPoint2 par_o, SVec2 par_d0X, SVec2 par_ddX, SVec2 par_deltaY)
        : o(par_o), d0X(par_d0X), ddX(par_ddX), deltaY(par_deltaY) {}


    /// intersects the interpolation surface with a const-y plane
    CPU_GPU LinearInterp GetLinearInterp(int subY)
    {
        SVec2 deltaX=VecAdd(d0X, VecMul(ddX, subY));
        SPoint2 lineBeg= MovePoint(o, VecMul(deltaY,subY) );
        return LinearInterp(lineBeg, deltaX);
    }

    /// Calculates a value of the interpolation surface at a point (subX,subY)
    CPU_GPU SPoint2 Interpolate(int subX, int subY)
    {
        LinearInterp lineY=GetLinearInterp(subY);
        return lineY.Interpolate(subX);
    }
};

/// defines a pixel having image, variance and mask planes
template<typename T>
struct PixelIVM
{
    T img;
    VarPixel var;
    MskPixel msk;
};

enum KernelType
{ KERNEL_TYPE_LANCZOS, KERNEL_TYPE_BILINEAR, KERNEL_TYPE_NEAREST_NEIGHBOR };

/// defines memory region containing image data
template<typename T>
struct ImageDataPtr
{
    T* img;
    VarPixel* var;
    MskPixel* msk;
    int strideImg;
    int strideVar;
    int strideMsk;
    int width;
    int height;
};

/**
    @brief Calls the GPU kernel for lanczos resampling

    @arg isMaskedImage - if false, only the image plane is calculated, mask and variance planes are ignored
    @arg destImageGpu - (output) defines memory region (on GPU) containing allocated buffer for output data
    @arg srcImageGpu - defines memory region (on GPU) containing source image data
    @arg srcGoodBox - valid source pixel centers ffor Lanczos kernels
    @arg kernelCenterX, kernelCenterY - offset of Lanczos kernel center, in pixels
    @arg edgePixel - set this for all dest. image output pixels mapped from outside of bounds of the source image
    @arg srcPosInterp - a 2D array defining a piecewise bilinear interpolation of a coordinate transform function over
                   input image. Each element defines coordinate transform of one part of the input image.
                   The size of each part is defined by the interpLength parameter.
    @arg interpLength - defines width and height of parts of the input image (for interpolation)
*/
template<typename DestPixelT, typename SrcPixelT>
void WarpImageGpuCallKernel(bool isMaskedImage,
                            ImageDataPtr<DestPixelT> destImageGpu,
                            ImageDataPtr<SrcPixelT>  srcImageGpu,
                            int mainKernelSize,
                            KernelType maskKernelType,
                            int maskKernelSize,
                            SBox2I srcGoodBox,
                            PixelIVM<DestPixelT> edgePixel,
                            BilinearInterp* srcPosInterp,
                            int interpLength
                            );

}}}}} //namespace lsst::afw::math::detail::gpu ends
