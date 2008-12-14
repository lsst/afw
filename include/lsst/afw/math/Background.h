#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/**
 * \file
 * \brief ImageT Background
 */

#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"


namespace lsst { namespace afw { namespace math {

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}
    
/// \brief Pass parameters to a Background object
class BackgroundControl {
public:
    BackgroundControl(interpolate::Style style, int const nxSample=10, int const nySample=10)
        : _nxSample(nxSample), _nySample(nySample) {
        assert(nxSample > 0);
        assert(nySample > 0);
        sctrl = StatisticsControl();
        ictrl = interpolate::InterpControl(style);
    }
    ~BackgroundControl() {}
    void setNxSample (int nxSample) { assert(nxSample > 0); _nxSample = nxSample; }
    void setNySample (int nySample) { assert(nySample > 0); _nySample = nySample; }
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    StatisticsControl sctrl;
    interpolate::InterpControl ictrl;
private:
    int _nxSample;
    int _nySample;
};
    
/**
 * A class to evaluate %image background levels
 *
 * The basic strategy is to construct a Background object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Background methods:
 * \code
        math::Background<ImageT> stats = math::make_Background(*img, math::NPOINT | math::MEAN);
        
        double const n = stats.getValue(math::NPOINT);
        std::pair<double, double> const mean = stats.getResult(math::MEAN); // Returns (value, error)
        double const meanError = stats.getError(math::MEAN);                // just the error
 * \endcode
 *
 * (Note that we used a helper function, \c make_Background, rather that the constructor directly so that
 * the compiler could deduce the types -- cf. \c std::make_pair)
 */
template<typename ImageT>
class Background {
public:

    explicit Background(ImageT const& img, BackgroundControl const& bgCtrl);
    ~Background() {}
    typename ImageT::Pixel getPixel(int const x, int const y) const;
    ImageT getFrame() const;
    //~Background() { delete _grid; };
    
private:
    int _n;                             // number of pixels in the image
    double _meanclip;                   // n-sigma clipped mean
    ImageT _img;
    int _nxSample;
    int _nySample;
    int _subimgWidth;
    int _subimgHeight;
    std::vector<int> _xcen;
    std::vector<int> _ycen;
    std::vector<int> _xorig;
    std::vector<int> _yorig;
    std::vector<std::vector<typename ImageT::Pixel> > _grid;
    std::vector<std::vector<typename ImageT::Pixel> > _gridcolumns;
    BackgroundControl _bctrl;
};

/// A convenience function that uses function overloading to make the correct type of Background
/// cf. std::make_pair()
template<typename ImageT>
Background<ImageT> make_Background(ImageT const& img, BackgroundControl const& bgCtrl=BackgroundControl()) { ///< ImageT (or MaskedImage) whose properties we want
    return Background<ImageT>(img, bgCtrl);
};
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
