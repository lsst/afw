// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   MovingObjectPrediction.h
//! \brief  Persistable C++ data object for moving object predictions
//
//##====----------------                                ----------------====##/

#ifndef LSST_FW_MOVING_OBJECT_PREDICTION_H
#define LSST_FW_MOVING_OBJECT_PREDICTION_H

#include <boost/cstdint.hpp>

#include <lsst/mwi/data/Citizen.h>
#include <lsst/mwi/persistence/Persistable.h>


namespace boost {
namespace serialization {
    class access;
}}


namespace lsst {
namespace fw {

#ifndef SWIG
using boost::int64_t;
#endif

// forward declarations for formatters
namespace formatters {
    class MovingObjectPredictionVectorFormatter;
}


/*!
    Contains predicted attributes of a moving object at a sepcific time. This class is useful
    when an unadorned data structure is required (e.g. for placement into shared memory) or
    is all that is necessary.
 */
class MovingObjectPrediction {

public :

    MovingObjectPrediction();

    // Getters required by association pipeline
    int64_t getId()                  const { return _orbitId; }
    double  getRa()                  const { return _ra;      }
    double  getDec()                 const { return _dec;     }
    double  getSemiMinorAxisLength() const { return _smia;    }
    double  getSemiMajorAxisLength() const { return _smaa;    }
    double  getPositionAngle()       const { return _pa;      }

    double  getMjd()                 const { return _mjd;     }
    double  getMagnitude()           const { return _mag;     }

    void setId                 (int64_t const id  ) { _orbitId = id; }
    void setRa                 (double  const ra  ) { _ra   = ra;    }
    void setDec                (double  const dec ) { _dec  = dec;   }
    void setSemiMinorAxisLength(double  const smia) { _smia = smia;  }
    void setSemiMajorAxisLength(double  const smaa) { _smaa = smaa;  }
    void setPositionAngle      (double  const pa  ) { _pa   = pa;    }
    void setMjd                (double  const mjd ) { _mjd  = mjd;   }
    void setMagnitude          (double  const mag ) { _mag  = mag;   }

    bool operator==(MovingObjectPrediction const & d) const;

private :

    int64_t _orbitId; //!< ID of the orbit this is a prediction for
    double  _ra;      //!< right ascension (deg)
    double  _dec;     //!< declination (deg)
    double  _smaa;    //!< error ellipse semi major axis (deg)
    double  _smia;    //!< error ellipse semi minor axis (deg)
    double  _pa;      //!< error ellipse position angle (deg)
    double  _mjd;     //!< input ephemerides date time (UTC MJD)
    double  _mag;     //!< apparent magnitude (mag)

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        ar & _orbitId;
        ar & _ra;
        ar & _dec;
        ar & _smaa;
        ar & _smia;
        ar & _pa;
        ar & _mjd;
        ar & _mag;
    }

    friend class boost::serialization::access;
    friend class formatters::MovingObjectPredictionVectorFormatter;
};

inline bool operator!=(MovingObjectPrediction const & d1, MovingObjectPrediction const & d2) {
    return !(d1 == d2);
}


// Classes that require special handling in the SWIG interface file
#ifndef SWIG

/*!
    A persistable container of MovingObjectPrediction instances, implemented using std::vector.
 */
class MovingObjectPredictionVector :
    public  lsst::mwi::persistence::Persistable,
    private lsst::mwi::data::Citizen
{
public :

    typedef boost::shared_ptr<MovingObjectPredictionVector> Ptr;
    typedef std::vector<MovingObjectPrediction>             Vector;

    typedef Vector::allocator_type         allocator_type;
    typedef Vector::iterator               iterator;
    typedef Vector::const_iterator         const_iterator;
    typedef Vector::reverse_iterator       reverse_iterator;
    typedef Vector::const_reverse_iterator const_reverse_iterator;
    typedef Vector::size_type              size_type;
    typedef Vector::difference_type        difference_type;
    typedef Vector::reference              reference;
    typedef Vector::const_reference        const_reference;
    typedef Vector::value_type             value_type;

    MovingObjectPredictionVector();
    explicit MovingObjectPredictionVector(size_type sz);
    MovingObjectPredictionVector(size_type sz, value_type const & val);

    template <typename InputIterator>
    MovingObjectPredictionVector(InputIterator beg, InputIterator end) :
        lsst::mwi::data::Citizen(typeid(*this)),
        _vec(beg, end)
    {}

    virtual ~MovingObjectPredictionVector();

    MovingObjectPredictionVector(MovingObjectPredictionVector const & vec);
    explicit MovingObjectPredictionVector(Vector const & vec);
    MovingObjectPredictionVector & operator=(MovingObjectPredictionVector const & vec);
    MovingObjectPredictionVector & operator=(Vector const & vec);

    void swap(MovingObjectPredictionVector & v) { using std::swap; swap(_vec, v._vec); }
    void swap(Vector & v)          { using std::swap; swap(_vec, v);      }

    size_type size()     const { return _vec.size();     }
    size_type max_size() const { return _vec.max_size(); }
    bool      empty()    const { return _vec.empty();    }
    size_type capacity() const { return _vec.capacity(); }

    void reserve(size_type const n) { _vec.reserve(n); }

    template <typename InputIterator>
    void assign(InputIterator beg, InputIterator end)      { _vec.assign(beg, end); }
    void assign(size_type const n, value_type const & val) { _vec.assign(n, val);   }

    reference       at        (size_type const i)       { return _vec.at(i); }
    const_reference at        (size_type const i) const { return _vec.at(i); }
    reference       operator[](size_type const i)       { return _vec[i];    }
    const_reference operator[](size_type const i) const { return _vec[i];    }

    reference       front()       { return _vec.front(); }
    const_reference front() const { return _vec.front(); }
    reference       back ()       { return _vec.back();  }
    const_reference back () const { return _vec.back();  }

    iterator               begin ()       { return _vec.begin();  }
    const_iterator         begin () const { return _vec.begin();  }
    reverse_iterator       rbegin()       { return _vec.rbegin(); }
    const_reverse_iterator rbegin() const { return _vec.rbegin(); }
    iterator               end   ()       { return _vec.end();    }
    const_iterator         end   () const { return _vec.end();    }
    reverse_iterator       rend  ()       { return _vec.rend();   }
    const_reverse_iterator rend  () const { return _vec.rend();   }

    void push_back (value_type const & value) { _vec.push_back(value);  }

    void pop_back () { _vec.pop_back();  }
    void clear()     { _vec.clear();     }

    template <typename InputIterator>
    void     insert(iterator pos, InputIterator beg, InputIterator end) { _vec.insert(pos, beg, end);      }
    iterator insert(iterator pos, value_type const & val)               { return _vec.insert(pos, val);    }
    void     insert(iterator pos, size_type n, value_type const & val)  { return _vec.insert(pos, n, val); }

    iterator erase(iterator pos)               { return _vec.erase(pos);      }
    iterator erase(iterator beg, iterator end) { return _vec.erase(beg, end); }

    void resize(size_type n)                   { _vec.resize(n);        }
    void resize(size_type n, value_type value) { _vec.resize(n, value); }

    bool operator==(MovingObjectPredictionVector const & v) { return _vec == v._vec; }
    bool operator!=(MovingObjectPredictionVector const & v) { return _vec != v._vec; }

private :

    LSST_PERSIST_FORMATTER(formatters::MovingObjectPredictionVectorFormatter);

    Vector _vec;
};

#endif // SWIG


}}  // end of namespace lsst::fw

#endif // LSST_FW_MOVING_OBJECT_PREDICTION_H


