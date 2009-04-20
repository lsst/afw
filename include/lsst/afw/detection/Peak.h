#if !defined(LSST_AFW_DETECTION_PEAK_H)
#define LSST_AFW_DETECTION_PEAK_H
/*!
 * \file
 * \brief Support for peaks in images
 */
    
#include <list>
#include <cmath>
#include <boost/shared_ptr.hpp>
#include "lsst/daf/data/LsstBase.h"

namespace lsst { namespace afw { namespace detection {
/// A peak in an %image
class Peak : public lsst::daf::base::Citizen {
public:
    typedef boost::shared_ptr<Peak> Ptr;

    typedef std::vector<Peak::Ptr> List; ///< A collection of Peaks
    /// A peak at the pixel <tt>(ix, iy)</tt>
    explicit Peak(int ix,               //!< column pixel
                  int iy)               //!< row pixel
        : lsst::daf::base::Citizen(typeid(*this)),
          _id(++id),
          _ix(ix), _iy(iy), _fx(ix), _fy(iy) {};
    /// A peak at the floating-point position <tt>(fx, fy)</tt>
    explicit Peak(float fx=NAN,       //!< column centre
                  float fy=NAN)       //!< row centre
        : lsst::daf::base::Citizen(typeid(*this)),
          _id(++id),
          _ix(fx > 0 ? static_cast<int>(fx) : -static_cast<int>(-fx) - 1),
          _iy(fy > 0 ? static_cast<int>(fy) : -static_cast<int>(-fy) - 1),
          _fx(fx), _fy(fy) {};
    ~Peak() {};

    int getId() const { return _id; }   //!< Return the Peak's unique ID

    int getIx() const { return _ix; }         //!< Return the column pixel position
    int getIy() const { return _iy; }         //!< Return the row pixel position
    float getFx() const { return _fx; }       //!< Return the column centroid
    float getFy() const { return _fy; }       //!< Return the row centroid

    std::string toString();    
private:
    //Peak(const Peak &) {}             // XXX How do we manage Citizen's copy constructor?
    static int id;
    mutable int _id;                    //!< unique ID for this peak
    int _ix;                            //!< column-position of peak pixel
    int _iy;                            //!< row-position of peak pixel
    float _fx;                          //!< column-position of peak
    float _fy;                          //!< row-position of peak
};

}}}
#endif
