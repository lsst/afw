// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   DiaSource.h
//! \brief  The C++ representation of a Difference-Image-Analysis Source.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_DETECTION_SOURCE_H
#define LSST_AFW_DETECTION_SOURCE_H

#include <bitset>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <lsst/daf/data/Citizen.h>
#include <lsst/pex/persistence/Persistable.h>


namespace boost {
namespace serialization {
    class access;
}}


namespace lsst {
namespace fw {

#ifndef SWIG
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;
#endif

// forward declarations for formatters
namespace formatters {
    class DiaSourceVectorFormatter;
}


/*!
    Contains attributes for Difference-Image-Analysis Source records. This class is useful
    when an unadorned data structure is required (e.g. for placement into shared memory) or
    is all that is necessary.

    \p
    The C++ fields are derived from the LSST DC2 MySQL schema, which is reproduced below:

    \code
    CREATE TABLE DIASource
    (
        diaSourceId      BIGINT        NOT NULL,
        ccdExposureId    BIGINT        NOT NULL,
        filterId         TINYINT       NOT NULL,
        objectId         BIGINT        NULL,
        movingObjectId   BIGINT        NULL,
        scId             INTEGER       NOT NULL,
        colc             DOUBLE        NOT NULL,
        colcErr          DECIMAL(4,2)  NOT NULL,
        rowc             DOUBLE        NOT NULL,
        rowcErr          DECIMAL(4,2)  NOT NULL,
        dcol             DOUBLE        NOT NULL,
        drow             DOUBLE        NOT NULL,
        ra               DOUBLE        NOT NULL,
        decl             DOUBLE        NOT NULL,
        raErr4detection  DECIMAL(7,5)  NOT NULL,
        decErr4detection DECIMAL(7,5)  NOT NULL,
        raErr4wcs        DECIMAL(7,5)  NULL,
        decErr4wcs       DECIMAL(7,5)  NULL,
        cx               DOUBLE        NOT NULL,
        cy               DOUBLE        NOT NULL,
        cz               DOUBLE        NOT NULL,
        taiMidPoint      DECIMAL(12,7) NOT NULL,
        taiRange         DECIMAL(12,7) NOT NULL,
        fwhmA            DECIMAL(4,2)  NOT NULL,
        fwhmB            DECIMAL(4,2)  NOT NULL,
        fwhmTheta        DECIMAL(4,1)  NOT NULL,
        flux             DECIMAL(12,2) NOT NULL,
        fluxErr          DECIMAL(10,2) NOT NULL,
        psfMag           DECIMAL(7,3)  NOT NULL,
        psfMagErr        DECIMAL(6,3)  NOT NULL,
        apMag            DECIMAL(7,3)  NOT NULL,
        apMagErr         DECIMAL(6,3)  NOT NULL,
        modelMag         DECIMAL(6,3)  NOT NULL,
        modelMagErr      DECIMAL(6,3)  NOT NULL,
        apDia            FLOAT(0)      NULL,
        Ixx              FLOAT(0)      NULL,
        IxxErr           FLOAT(0)      NULL,
        Iyy              FLOAT(0)      NULL,
        IyyErr           FLOAT(0)      NULL,
        Ixy              FLOAT(0)      NULL,
        IxyErr           FLOAT(0)      NULL,
        snr              FLOAT(0)      NOT NULL,
        chi2             FLOAT(0)      NOT NULL,
        flag4association SMALLINT      NULL,
        flag4detection   SMALLINT      NULL,
        flag4wcs         SMALLINT      NULL,
        _dataSource      TINYINT       NOT NULL,
        PRIMARY KEY (diaSourceId),
        KEY (ccdExposureId),
        KEY (filterId),
        KEY (movingObjectId),
        KEY (objectId),
        KEY (scId)
    ) ENGINE=MyISAM;
    \endcode

    \p
    Note that the C++ fields are listed in a different order than their corresponding database
    columns: fields are sorted by type size to minimize the number of padding bytes that the
    compiler must insert to meet field alignment requirements.
 */
class DiaSource {

public :

    typedef boost::shared_ptr<DiaSource> Ptr;

    /*! An integer id for each nullable field. */
    enum NullableField {

        OBJECT_ID = 0,
        MOVING_OBJECT_ID,
        RA_ERR_4_WCS,
        DEC_ERR_4_WCS,
        AP_DIA,
        IXX,
        IXX_ERR,
        IYY,
        IYY_ERR,
        IXY,
        IXY_ERR,
        FLAG_4_ASSOCIATION,
        FLAG_4_DETECTION,
        FLAG_4_WCS,
        NUM_NULLABLE_FIELDS
    };

    DiaSource();

    DiaSource(
        int64_t id,
        double  colc,
        double  rowc,
        double  dcol,
        double  drow
    );

    // getters
    int64_t getId()               const { return _diaSourceId;      }
    int64_t getCcdExposureId()    const { return _ccdExposureId;    }
    int64_t getObjectId()         const { return _objectId;         }
    int64_t getMovingObjectId()   const { return _movingObjectId;   }
    double  getColc()             const { return _colc;             }
    double  getRowc()             const { return _rowc;             }
    double  getDcol()             const { return _dcol;             }
    double  getDrow()             const { return _drow;             }
    double  getRa()               const { return _ra;               }
    double  getDec()              const { return _decl;             }
    double  getRaErr4detection()  const { return _raErr4detection;  }
    double  getDecErr4detection() const { return _decErr4detection; }
    double  getRaErr4wcs()        const { return _raErr4wcs;        }
    double  getDecErr4wcs()       const { return _decErr4wcs;       }
    double  getCx()               const { return _cx;               }
    double  getCy()               const { return _cy;               }
    double  getCz()               const { return _cz;               }
    double  getTaiMidPoint()      const { return _taiMidPoint;      }
    double  getTaiRange()         const { return _taiRange;         }
    double  getFlux()             const { return _flux;             }
    double  getFluxErr()          const { return _fluxErr;          }
    double  getPsfMag()           const { return _psfMag;           }
    double  getPsfMagErr()        const { return _psfMagErr;        }
    double  getApMag()            const { return _apMag;            }
    double  getApMagErr()         const { return _apMagErr;         }
    double  getModelMag()         const { return _modelMag;         }
    double  getModelMagErr()      const { return _modelMagErr;      }
    float   getColcErr()          const { return _colcErr;          }
    float   getRowcErr()          const { return _rowcErr;          }
    float   getFwhmA()            const { return _fwhmA;            }
    float   getFwhmB()            const { return _fwhmB;            }
    float   getFwhmTheta()        const { return _fwhmTheta;        }
    float   getApDia()            const { return _apDia;            }
    float   getIxx()              const { return _ixx;              }
    float   getIxxErr()           const { return _ixxErr;           }
    float   getIyy()              const { return _iyy;              }
    float   getIyyErr()           const { return _iyyErr;           }
    float   getIxy()              const { return _ixy;              }
    float   getIxyErr()           const { return _ixyErr;           }
    float   getSnr()              const { return _snr;              }
    float   getChi2()             const { return _chi2;             }
    int32_t getScId()             const { return _scId;             }
    int16_t getFlag4association() const { return _flag4association; }
    int16_t getFlag4detection()   const { return _flag4detection;   }
    int16_t getFlag4wcs()         const { return _flag4wcs;         }
    int8_t  getFilterId()         const { return _filterId;         }
    int8_t  getDataSource()       const { return _dataSource;       }

    // setters
    void setId              (int64_t const diaSourceId     ) { _diaSourceId      = diaSourceId;      }
    void setCcdExposureId   (int64_t const ccdExposureId   ) { _ccdExposureId    = ccdExposureId;    }
    void setColc            (double  const colc            ) { _colc             = colc;             }
    void setRowc            (double  const rowc            ) { _rowc             = rowc;             }
    void setDcol            (double  const dcol            ) { _dcol             = dcol;             }
    void setDrow            (double  const drow            ) { _drow             = drow;             }
    void setRa              (double  const ra              ) { _ra               = ra;               }
    void setDec             (double  const decl            ) { _decl             = decl;             }
    void setRaErr4detection (double  const raErr4detection ) { _raErr4detection  = raErr4detection;  }
    void setDecErr4detection(double  const decErr4detection) { _decErr4detection = decErr4detection; }
    void setCx              (double  const cx              ) { _cx               = cx;               }
    void setCy              (double  const cy              ) { _cy               = cy;               }
    void setCz              (double  const cz              ) { _cz               = cz;               }
    void setTaiMidPoint     (double  const taiMidPoint     ) { _taiMidPoint      = taiMidPoint;      }
    void setTaiRange        (double  const taiRange        ) { _taiRange         = taiRange;         }
    void setFlux            (double  const flux            ) { _flux             = flux;             }
    void setFluxErr         (double  const fluxErr         ) { _fluxErr          = fluxErr;          }
    void setPsfMag          (double  const psfMag          ) { _psfMag           = psfMag;           }
    void setPsfMagErr       (double  const psfMagErr       ) { _psfMagErr        = psfMagErr;        }
    void setApMag           (double  const apMag           ) { _apMag            = apMag;            }
    void setApMagErr        (double  const apMagErr        ) { _apMagErr         = apMagErr;         }
    void setModelMag        (double  const modelMag        ) { _modelMag         = modelMag;         }
    void setModelMagErr     (double  const modelMagErr     ) { _modelMagErr      = modelMagErr;      }
    void setColcErr         (float   const colcErr         ) { _colcErr          = colcErr;          }
    void setRowcErr         (float   const rowcErr         ) { _rowcErr          = rowcErr;          }
    void setFwhmA           (float   const fwhmA           ) { _fwhmA            = fwhmA;            }
    void setFwhmB           (float   const fwhmB           ) { _fwhmB            = fwhmB;            }
    void setFwhmTheta       (float   const fwhmTheta       ) { _fwhmTheta        = fwhmTheta;        }
    void setSnr             (float   const snr             ) { _snr              = snr;              }
    void setChi2            (float   const chi2            ) { _chi2             = chi2;             }
    void setScId            (int32_t const scId            ) { _scId             = scId;             }
    void setFilterId        (int8_t  const filterId        ) { _filterId         = filterId;         }
    void setDataSource      (int8_t  const dataSource      ) { _dataSource       = dataSource;       }

    void setObjectId(int64_t const objectId) {
        _objectId = objectId;
        setNotNull(OBJECT_ID);
    }
    void setMovingObjectId(int64_t const movingObjectId) {
        _movingObjectId = movingObjectId;
        setNotNull(MOVING_OBJECT_ID);
    }
    void setRaErr4wcs(double const raErr4wcs) {
        _raErr4wcs = raErr4wcs;
        setNotNull(RA_ERR_4_WCS);
    }
    void setDecErr4wcs(double const decErr4wcs) {
        _decErr4wcs = decErr4wcs;
        setNotNull(DEC_ERR_4_WCS);
    }
    void setApDia(float const apDia) {
        _apDia = apDia;
        setNotNull(AP_DIA);
    }
    void setIxx   (float const ixx   ) { _ixx    = ixx;    setNotNull(IXX);     }
    void setIxxErr(float const ixxErr) { _ixxErr = ixxErr; setNotNull(IXX_ERR); }         
    void setIyy   (float const iyy   ) { _iyy    = iyy;    setNotNull(IYY);     }     
    void setIyyErr(float const iyyErr) { _iyyErr = iyyErr; setNotNull(IYY_ERR); }         
    void setIxy   (float const ixy   ) { _ixy    = ixy;    setNotNull(IXY);     }      
    void setIxyErr(float const ixyErr) { _ixyErr = ixyErr; setNotNull(IXY_ERR); }         
    void setFlag4association(int16_t const flag4association) {
        _flag4association = flag4association;
        setNotNull(FLAG_4_ASSOCIATION);
    }
    void setFlag4detection(int16_t const flag4detection) {
        _flag4detection = flag4detection;
        setNotNull(FLAG_4_DETECTION);
    }
    void setFlag4wcs(int16_t const flag4wcs) {
        _flag4wcs = flag4wcs;
        setNotNull(FLAG_4_WCS);
    }


    // Get/set whether or not fields are null
    bool isNull    (NullableField const f) const            { return _nulls.test(f); }
    void setNull   (NullableField const f)                  { _nulls.set(f);         }
    void setNotNull(NullableField const f)                  { _nulls.reset(f);       }
    void setNull   (NullableField const f, bool const null) { _nulls.set(f, null);   }
    void setNull   ()                                       { _nulls.set();          }
    void setNotNull()                                       { _nulls.reset();        }

    bool operator==(DiaSource const & d) const;

private :

    int64_t _diaSourceId;      // BIGINT        NOT NULL
    int64_t _ccdExposureId;    // BIGINT        NOT NULL
    int64_t _objectId;         // BIGINT        NULL
    int64_t _movingObjectId;   // BIGINT        NULL
    double  _colc;             // DOUBLE        NOT NULL
    double  _rowc;             // DOUBLE        NOT NULL
    double  _dcol;             // DOUBLE        NOT NULL
    double  _drow;             // DOUBLE        NOT NULL
    double  _ra;               // DOUBLE        NOT NULL
    double  _decl;             // DOUBLE        NOT NULL
    double  _raErr4detection;  // DECIMAL(7,5)  NOT NULL
    double  _decErr4detection; // DECIMAL(7,5)  NOT NULL
    double  _raErr4wcs;        // DECIMAL(7,5)  NULL
    double  _decErr4wcs;       // DECIMAL(7,5)  NULL
    double  _cx;               // DOUBLE        NOT NULL
    double  _cy;               // DOUBLE        NOT NULL
    double  _cz;               // DOUBLE        NOT NULL
    double  _taiMidPoint;      // DECIMAL(12,7) NOT NULL
    double  _taiRange;         // DECIMAL(12,7) NOT NULL
    double  _flux;             // DECIMAL(12,2) NOT NULL
    double  _fluxErr;          // DECIMAL(10,2) NOT NULL
    double  _psfMag;           // DECIMAL(7,3)  NOT NULL
    double  _psfMagErr;        // DECIMAL(6,3)  NOT NULL
    double  _apMag;            // DECIMAL(7,3)  NOT NULL
    double  _apMagErr;         // DECIMAL(6,3)  NOT NULL
    double  _modelMag;         // DECIMAL(6,3)  NOT NULL
    double  _modelMagErr;      // DECIMAL(6,3)  NOT NULL
    float   _colcErr;          // DECIMAL(4,2)  NOT NULL
    float   _rowcErr;          // DECIMAL(4,2)  NOT NULL
    float   _fwhmA;            // DECIMAL(4,2)  NOT NULL
    float   _fwhmB;            // DECIMAL(4,2)  NOT NULL
    float   _fwhmTheta;        // DECIMAL(4,1)  NOT NULL
    float   _apDia;            // FLOAT(0)      NULL
    float   _ixx;              // FLOAT(0)      NULL
    float   _ixxErr;           // FLOAT(0)      NULL
    float   _iyy;              // FLOAT(0)      NULL
    float   _iyyErr;           // FLOAT(0)      NULL
    float   _ixy;              // FLOAT(0)      NULL
    float   _ixyErr;           // FLOAT(0)      NULL
    float   _snr;              // FLOAT(0)      NOT NULL
    float   _chi2;             // FLOAT(0)      NOT NULL
    int32_t _scId;             // INTEGER       NOT NULL
    int16_t _flag4association; // SMALLINT      NULL
    int16_t _flag4detection;   // SMALLINT      NULL
    int16_t _flag4wcs;         // SMALLINT      NULL
    int8_t  _filterId;         // TINYINT       NOT NULL
    int8_t  _dataSource;       // TINYINT       NOT NULL

    std::bitset<NUM_NULLABLE_FIELDS> _nulls;

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        ar & _diaSourceId;
        ar & _ccdExposureId;
        ar & _objectId;
        ar & _movingObjectId;
        ar & _colc;
        ar & _rowc;
        ar & _dcol;
        ar & _drow;
        ar & _ra;
        ar & _decl;
        ar & _raErr4detection;
        ar & _decErr4detection;
        ar & _raErr4wcs;
        ar & _decErr4wcs;
        ar & _cx;
        ar & _cy;
        ar & _cz;
        ar & _taiMidPoint;
        ar & _taiRange;
        ar & _flux;
        ar & _fluxErr;
        ar & _psfMag;
        ar & _psfMagErr;
        ar & _apMag;
        ar & _apMagErr;
        ar & _modelMag;
        ar & _modelMagErr;
        ar & _colcErr;
        ar & _rowcErr;
        ar & _fwhmA;
        ar & _fwhmB;
        ar & _fwhmTheta;
        ar & _apDia;
        ar & _ixx;
        ar & _ixxErr;
        ar & _iyy;
        ar & _iyyErr;
        ar & _ixy;
        ar & _ixyErr;
        ar & _snr;
        ar & _chi2;
        ar & _scId;
        ar & _flag4association;
        ar & _flag4detection;
        ar & _flag4wcs;
        ar & _filterId;
        ar & _dataSource;

        bool b;
        if (Archive::is_loading::value) {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                ar & b;
                _nulls.set(i, b);
            }
        } else {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                b = _nulls.test(i);
                ar & b;
            }
        }
    }

    friend class boost::serialization::access;
    friend class formatters::DiaSourceVectorFormatter;
};

inline bool operator!=(DiaSource const & d1, DiaSource const & d2) {
    return !(d1 == d2);
}


// Classes that require special handling in the SWIG interface file
#ifndef SWIG

/*!
    A persistable container of DiaSource instances (implemented using std::vector).
 */
class DiaSourceVector :
    public lsst::pex::persistence::Persistable,
    public lsst::daf::data::Citizen
{
public :
    typedef boost::shared_ptr<DiaSourceVector> Ptr;
    typedef std::vector<DiaSource>             Vector;

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

    DiaSourceVector();
    explicit DiaSourceVector(size_type sz);
    DiaSourceVector(size_type sz, value_type const & val);

    template <typename InputIterator>
    DiaSourceVector(InputIterator beg, InputIterator end) :
        lsst::daf::data::Citizen(typeid(*this)),
        _vec(beg, end)
    {}

    virtual ~DiaSourceVector();

    DiaSourceVector(DiaSourceVector const & vec);
    explicit DiaSourceVector(Vector const & vec);
    DiaSourceVector & operator=(DiaSourceVector const & vec);
    DiaSourceVector & operator=(Vector const & vec);

    void swap(DiaSourceVector & v) { using std::swap; swap(_vec, v._vec); }
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

    void push_back(value_type const & value) { _vec.push_back(value);  }

    void pop_back() { _vec.pop_back();  }
    void clear()    { _vec.clear();     }

    iterator insert(iterator pos, value_type const & val)              { return _vec.insert(pos, val);    }
    void     insert(iterator pos, size_type n, value_type const & val) { return _vec.insert(pos, n, val); }

    template <typename InputIterator>
    void insert(iterator pos, InputIterator beg, InputIterator end) { _vec.insert(pos, beg, end); }

    iterator erase(iterator pos)               { return _vec.erase(pos);      }
    iterator erase(iterator beg, iterator end) { return _vec.erase(beg, end); }

    void resize(size_type n)                 { _vec.resize(n);      }
    void resize(size_type n, value_type val) { _vec.resize(n, val); }

    bool operator==(DiaSourceVector const & v) { return _vec == v._vec; }
    bool operator!=(DiaSourceVector const & v) { return _vec != v._vec; }

private :

    LSST_PERSIST_FORMATTER(formatters::DiaSourceVectorFormatter);

    Vector _vec;
};

#endif // SWIG


}}  // end of namespace lsst::afw

#endif // LSST_AFW_DETECTION_SOURCE_H


