// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  base container for source/dia source data
//
//##====----------------                                ----------------====##/


/*
        id      BIGINT         NOT NULL,
        filterId         TINYINT        NOT NULL,
        objectId         BIGINT         NULL,
        movingObjectId   BIGINT         NULL,
        procHistoryId    INTEGER        NOT NULL,
        ra               DOUBLE(12,9)   NOT NULL,
        decl             DOUBLE(11,9)   NOT NULL,
        xFlux            DOUBLE(10)     NULL,
        xFluxErr         DOUBLE(10)     NULL,
        yFlux            DOUBLE(10)     NULL,
        yFluxErr         DOUBLE(10)     NULL,
        raFlux           DOUBLE(10)     NULL,
        raFluxErr        DOUBLE(10)     NULL,
        declFlux         DOUBLE(10)     NULL,
        declFluxErr      DOUBLE(10)     NULL,
        xPeak            DOUBLE(10)     NULL,
        yPeak            DOUBLE(10)     NULL,
        raPeak           DOUBLE(10)     NULL,
        declPeak         DOUBLE(10)     NULL,
        xAstrom          DOUBLE(10)     NULL,
        xAstromErr       DOUBLE(10)     NULL,
        yAstrom          DOUBLE(10)     NULL,
        yAstromErr       DOUBLE(10)     NULL,
        raAstrom         DOUBLE(10)     NULL,
        raAstromErr      DOUBLE(10)     NULL,
        declAstrom       DOUBLE(10)     NULL,
        declAstromErr    DOUBLE(10)     NULL,        
        taiMidPoint      DOUBLE(12,7)   NOT NULL,
        fwhmA            FLOAT(0)       NOT NULL,
        fwhmB            FLOAT(0)       NOT NULL,
        fwhmTheta        FLOAT(0)       NOT NULL,
        psfMag           DOUBLE(7,3)    NOT NULL,
        psfMagErr        FLOAT(0)       NOT NULL,
        apMag            DOUBLE(7,3)    NOT NULL,
        apMagErr         FLOAT(0)       NOT NULL,
        modelMag         DOUBLE(6,3)    NOT NULL,

        apDia            FLOAT(0)       NULL,

        snr              FLOAT(0)       NOT NULL,
        chi2             FLOAT(0)       NOT NULL,

        flag4association SMALLINT      NULL,
        flag4detection   SMALLINT      NULL,
        flag4wcs         SMALLINT      NULL,
*/

#include <vector>

#include <boost/any.hpp>


namespace boost {
namespace serialization {
    class access;
}}

namespace lsst {
namespace afw {
    namespace formatters {
        class SourceVectorFormatter;
    }
namespace detection{

#ifndef SWIG
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;
#endif

class SourceBase {

public:
    typedef boost::shared_ptr<SourceBase> Ptr;

    enum FieldId {
        ID = 0,
        AMP_EXPOSURE_ID
        FILTER_ID,
        IBJECT_ID,
        MOVING_OBJECT_ID,
        PROC_HISTORY_ID,
        RA,
        DECL,
        RA_ERR_4_WCS,
        DEC_ERR_4_WCS,
        RA_ERR_4_DETECTION,
        DEC_ERR_4_DETECTION,
        X_FLUX,
        X_FLUX_ERR,
        Y_FLUX,
        Y_FLUX_ERR,
        RA_FLUX,
        RA_FLUX_ERR,
        DEC_FLUX,
        DEC_FLUX_ERR,
        X_PEAK,
        Y_PEAK,
        RA_PEAK,
        DEC_PEAK,
        X_ASTROM,
        X_ASTROM_ERR,
        Y_ASTROM,
        Y_ASTROM_ERR,
        RA_ASTROM,
        RA_ASTROM_ERR,
        DEC_ASTROM,
        DEC_ASTROM_ERR,
        TAI_MID_POINT,
        TAI_RANGE,
        FWHM_A,
        FWHM_B,
        FWHM_THETA,
        PSF_MAG,
        PSF_MAG_ERR,
        AP_MAG,
        AP_MAG_ERR,
        MODEL_MAG,
        MODEL_MAG_ERR,
        AP_DIA,
        SNR,
        CHI2,
        FLAG_4_ASSOCIATION,
        FLAG_4_DETECTION,
        FLAG_4_WCS,    
        NUM_FIELDS
    };

    SourceBase() : _fieldList(NUM_FIELDS), _fieldNameList(NUM_FIELDS) {
        setNull();
    }

    virtual ~SourceBase() {
        setNull();                   
    } 

    template<typename T>
    T get(int const id) const {
        if(isNull(id))
            //TODO use LSST exception handling...
            return static_cast<T>(0)
        else return boost::any_cast<T>(_fieldList[id]);
    }
    
    template<typename T>
    T & get(int const field) {
        if(isNull(field)) { 
            //TODO use LSST exception handling...
            throw NullPointerException;
        }
        else return boost::any_cast<T>(_fieldList[field]);
    }  
           
        
    template<typename T>
    bool get(int const id, T & out) const {
        if(isNull(id))
            return false;
        else {
            out = boost::any_cast<T>(_fieldList[id]);
            return true;
        }
    }
    
    template<typename T>    
    void set(int const id, T in) {    
        _fieldList[id] = in;
    }
    
    bool isNull    (int const field) const { 
        return _fieldList[field].empty(); 
    }
    
    void setNull   (int const field) {
        _fieldList[field] = boost::any();
    }
    
    void setNull   () {
        for(int f= 0; f < _fieldList.size(); f++)
            setNull(f);            
    }
    
    
    // named setters to facilitate use and keep old API
    // templated set method prefered
    void setId(int64_t const & id) { return set(ID, id);}
    void setAmpExposureId(int64_t const & amdExposureId) {return set(AMP_EXPOSURE_ID, ampExposureId)};    
    void setFilterId(int8_t const & filterId) { return set(FILTER_ID, filterId); }
    void setObjectId(int64_t const & objectId) { return set(OBJECT_ID, objectId); }
    void setMovingObjectId(int64_t const & movingObjectId) { return set(MOVING_OBJECT_ID, movingObjectId); }
    void setProcHistoryId(int32_t const & procHistoryId) { return set(PROC_HISTORY_ID, procHistoryId); }
    void setRa(double const & ra) {return set(RA, ra); }
    void setDec(double const & dec) { return set(DECL, dec); }
    void setRaErr4detection(float const & raErr) { return set(RA_ERR_4_DETECTION, raErr); } 
    void setDecErr4detection(float const & decErr) {return set(DEC_ERR_4_DETECTION, decErr); }
    void setRaErr4wcs(float const & raErr) { return set(RA_ERR_4_WCS, raErr); }
    void setDecErr4wcs(float const & decErr) {return set(DEC_ERR_4_WCS, decErr); }
    void setXFlux(double const & xFlux) {return set(X_FLUX, xFlux); }
    void setXFluxErr(double const & xFluxErr) { return set(X_FLUX_ERR, xFluxErr); }
    void setYFlux(double const & yFlux) { return set(Y_FLUX, yFlux); }
    void setYFluxErr(double const & yFluxErr) { return set(Y_FLUX_ERR, yFluxErr); }
    void setRaFlux(double const & raFlux) { return set(RA_FLUX, raFlux); }
    void setRaFluxErr(double const & raFluxErr) { return set(RA_FLUX_ERR, raFluxErr); }
    void setDecFlux(double const & decFlux) { return set(DEC_FLUX, decFlux); }
    void setDecFluxErr(double const & decFluxErr) { return set(DEC_FLUX_ERR, decFluxErr); }
    void setXPeak(double const & xPeak) { return set(X_PEAK, xPeak); }
    void setYPeak(double const & yPeak) { return set(Y_PEAK, yPeak); }
    void setRaPeak(double const & raPeak) { return set(RA_PEAK, raPeak); }
    void setDecPeak(double const & decPeak) { return set(DEC_PEAK, decPeak); }
    void setXAstrom(double const & xAstrom) { return set(X_ASTROM, xAstrom); }
    void setXAstromErr(double const & xAstromErr) { return set(X_ASTROM_ERR, xAstromErr); }
    void setYAstrom(double const & yAstrom) { return set(Y_ASTROM, yAstrom); }
    void setYAstromErr(double const & yAstromErr) { return set(Y_ASTROM_ERR, yAstromErr); }
    void setRaAstrom(double const & raAstrom) { return set(RA_ASTROM, raAstrom); }
    void setRaAstromErr(double const & raAstromErr) { return set(RA_ASTROM_ERR, raAstromErr); }
    void setDecAstrom(double const & decAstrom) { return set(DEC_ASTROM, decAstrom); }
    void setDecAstromErr(double const & decAstromErr) { return set(DEC_ASTROM_ERR, decAstromErr); }
    void setTaiMidPoint(double const & taiMidPoint) { return set(TAI_MID_POINT, taiMidPoint); }
    void setTaiRange(float const & taiRange) { return set(TAI_RANGE, taiRange); }
    void setFwhmA(float const & fwhmA) { return set(FWHM_A, fwhmA); }
    void setFwhmB(float const & fwhmB) { return set(FWHM_B, fwhmB); }
    void setFwhmTheta(float const & fwhmTheta) { return set(FWHM_THETA, fwhmTheta); }
    void setPsfMag(double const & psfMag) { return set(PSF_MAG, psfMag); }
    void setPsfMagErr(float const & psfMagErr) { return set(PSF_MAG_ERR, psfMagErr); }
    void setApMag(double const & apMag) { return set(AP_MAG, apMag); }
    void setApMagErr(float const & apMagErr) { return set(AP_MAG_ERR, apMagErr); }
    void setModelMag(double const & modelMag) { return set(MODEL_MAG, modelMag); }
    void setModelMagErr(float const & modelMagErr) { return set(MODEL_MAG_ERR, modelMagErr); }
    void setApDia(float const & apDia) { return set(AP_DIA, apDia); }
    void setSnr(float const & snr) { return set(SNR, snr); }
    void setChi2(float const & chi2) { return set(CHI2, chi2); }
    void setFlag4association(int16_t const & flag) { return set(FLAG_4_ASSOCIATION. flag); }
    void setFlag4detection(int16_t const & flag) { return set(FLAG_4_DETECTION, flag); }
    void setFlag4wcs(int16_t const & flag) { return set(FLAG_4_WCS, flag); }

    // named getters to facilitate use and keep old API
    // use of templated get() is prefered   
    int64_t getId() const { return get(ID);}
    int64_t getAmpExposureId() const {return get(AMP_EXPOSURE_ID)};    
    int8_t  getFilterId() const { return get(FILTER_ID); }
    int64_t getObjectId() const { return get(OBJECT_ID); }
    int64_t getMovingObjectId() const { return get(MOVING_OBJECT_ID); }
    int32_t getProcHistoryId() const { return get(PROC_HISTORY_ID); }
    double  getRa() const {return get(RA); }
    double  getDec() const { return get(DECL); }
    float   getRaErr4detection() const { return get(RA_ERR_4_DETECTION); } 
    float   getDecErr4detection() const {return get(DEC_ERR_4_DETECTION); }
    float   getRaErr4wcs() const { return get(RA_ERR_4_WCS); }
    float   getDecErr4wcs() const {return get(DEC_ERR_4_WCS); }
    double  getXFlux() const {return get(X_FLUX); }
    double  getXFluxErr() const { return get(X_FLUX_ERR); }
    double  getYFlux() const { return get(Y_FLUX); }
    double  getYFluxErr() const { return get(Y_FLUX_ERR); }
    double  getRaFlux() const { return get(RA_FLUX); }
    double  getRaFluxErr() const { return get(RA_FLUX_ERR); }
    double  getDecFlux() const { return get(DEC_FLUX); }
    double  getDecFluxErr() const { return get(DEC_FLUX_ERR); }
    double  getXPeak() const { return get(X_PEAK); }
    double  getYPeak() const { return get(Y_PEAK); }
    double  getRaPeak() const { return get(RA_PEAK); }
    double  getDecPeak() const { return get(DEC_PEAK); }
    double  getXAstrom() const { return get(X_ASTROM); }
    double  getXAstromErr() const { return get(X_ASTROM_ERR); }
    double  getYAstrom() const { return get(Y_ASTROM); }
    double  getYAstromErr() const { return get(Y_ASTROM_ERR); }
    double  getRaAstrom() const { return get(RA_ASTROM); }
    double  getRaAstromErr() const { return get(RA_ASTROM_ERR); }
    double  getDecAstrom() const { return get(DEC_ASTROM); }
    double  getDecAstromErr() const { return get(DEC_ASTROM_ERR); }
    double  getTaiMidPoint() const { return get(TAI_MID_POINT); }
    float   getTaiRange() const { return get(TAI_RANGE); }
    float   getFwhmA() const { return get(FWHM_A); }
    float   getFwhmB() const { return get(FWHM_B); }
    float   getFwhmTheta() const { return get(FWHM_THETA); }
    double  getPsfMag() const { return get(PSF_MAG); }
    float   getPsfMagErr() const { return get(PSF_MAG_ERR); }
    double  getApMag() const { return get(AP_MAG); }
    float   getApMagErr() const { return get(AP_MAG_ERR); }
    double  getModelMag() const { return get(MODEL_MAG); }
    float   getModelMagErr() const { return get(MODEL_MAG_ERR); }
    float   getApDia() const { return get(AP_DIA); }
    float   getSnr() const { return get(SNR); }
    float   getChi2() const { return get(CHI2); }
    int16_t getFlag4association() const { return get(FLAG_4_ASSOCIATION); }
    int16_t getFlag4detection() const { return get(FLAG_4_DETECTION); }
    int16_t getFlag4wcs() const { return get(FLAG_4_WCS); }
    
    //named reference getters
    //use with caution, if field is null, will throw bad_any_cast exception
    int64_t & getId() { return get(ID);}
    int64_t & getAmpExposureId() {return get(AMP_EXPOSURE_ID)};    
    int8_t  & getFilterId() { return get(FILTER_ID); }
    int64_t & getObjectId() { return get(OBJECT_ID); }
    int64_t & getMovingObjectId() { return get(MOVING_OBJECT_ID); }
    int32_t & getProcHistoryId() { return get(PROC_HISTORY_ID); }
    double  & getRa() {return get(RA); }
    double  & getDec() { return get(DECL); }
    float   & getRaErr4detection() { return get(RA_ERR_4_DETECTION); } 
    float   & getDecErr4detection() {return get(DEC_ERR_4_DETECTION); }
    float   & getRaErr4wcs() { return get(RA_ERR_4_WCS); }
    float   & getDecErr4wcs() {return get(DEC_ERR_4_WCS); }
    double  & getXFlux() {return get(X_FLUX); }
    double  & getXFluxErr() { return get(X_FLUX_ERR); }
    double  & getYFlux() { return get(Y_FLUX); }
    double  & getYFluxErr() { return get(Y_FLUX_ERR); }
    double  & getRaFlux() { return get(RA_FLUX); }
    double  & getRaFluxErr() { return get(RA_FLUX_ERR); }
    double  & getDecFlux() { return get(DEC_FLUX); }
    double  & getDecFluxErr() { return get(DEC_FLUX_ERR); }
    double  & getXPeak() { return get(X_PEAK); }
    double  & getYPeak() { return get(Y_PEAK); }
    double  & getRaPeak() { return get(RA_PEAK); }
    double  & getDecPeak() { return get(DEC_PEAK); }
    double  & getXAstrom() { return get(X_ASTROM); }
    double  & getXAstromErr() { return get(X_ASTROM_ERR); }
    double  & getYAstrom() { return get(Y_ASTROM); }
    double  & getYAstromErr() { return get(Y_ASTROM_ERR); }
    double  & getRaAstrom() { return get(RA_ASTROM); }
    double  & getRaAstromErr() { return get(RA_ASTROM_ERR); }
    double  & getDecAstrom() { return get(DEC_ASTROM); }
    double  & getDecAstromErr() { return get(DEC_ASTROM_ERR); }
    double  & getTaiMidPoint() { return get(TAI_MID_POINT); }
    float   & getTaiRange() { return get(TAI_RANGE); }
    float   & getFwhmA() { return get(FWHM_A); }
    float   & getFwhmB() { return get(FWHM_B); }
    float   & getFwhmTheta() { return get(FWHM_THETA); }
    double  & getPsfMag() { return get(PSF_MAG); }
    float   & getPsfMagErr() { return get(PSF_MAG_ERR); }
    double  & getApMag() { return get(AP_MAG); }
    float   & getApMagErr() { return get(AP_MAG_ERR); }
    double  & getModelMag() { return get(MODEL_MAG); }
    float   & getModelMagErr() { return get(MODEL_MAG_ERR); }
    float   & getApDia() { return get(AP_DIA); }
    float   & getSnr() { return get(SNR); }
    float   & getChi2() { return get(CHI2); }
    int16_t & getFlag4association() { return get(FLAG_4_ASSOCIATION); }
    int16_t & getFlag4detection() { return get(FLAG_4_DETECTION); }
    int16_t & getFlag4wcs() { return get(FLAG_4_WCS); }

    
    
    virtual bool operator==(SourceBase const & d) const;
    
    
protected:
    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {  
        for(int f = 0; f< _fieldList.size(); f++) {
            get(f).serialize(ar, version);
        }
    
        bool b;
        if (Archive::is_loading::value) {
            for (int i = 0; i < _fieldList.size(); ++i) {
                ar & b;
                if(b)
                    setNull(i);
            }
        } else {
            for (int i = 0; i < _fieldList.size(); ++i) {
                b = isNull(i);
                ar & b;
            }
        }        
    }

    void addField(std::string const & name) {
        _fieldList.resize(_fieldList.size() + 1, Field());
        _fieldNameList.resize(_fieldNameList.size() + 1, name);
    }
    
    void addField(std::vector<std::string> nameList)
    {
        _fieldList.resize(_fieldList.size() + nameList.size(), Field());
        _fieldNameList.reserve(_fieldNameList.size() + nameList.size());
        for(int i = 0; i < nameList.size(); i++)
            _fieldNameList.push_back(nameList[i];                      
    }
     
    bool renameField(int const & id, std::string name)
    {
        if(id >=0 ** id < _fieldNameList.size()){
            _fieldNameList[id] = name;        
            return true;
        }
        else return false;
    }
private:
    class Field : public boost::any{
    public:
        Field(){}

        template<typename T>
        Field(T value)
            : boost::any(value) {}                           
        
        template <typename Archive> 
        void serialize(Archive & ar, unsigned int const version);          
    }

    std::vector<Field> _fieldList;    
    std::vector<std::string> _fieldNameList;
    
    friend class boost::serialization::access;
};

inline bool operator!=(SourceBase const & d1, SourceBase const & d2) {
    return !(d1 == d2);
}


typedef std::vector<SourceBase::Ptr> SourceVector;
typedef boost::shared_ptr<SourceVector> SourceVectorPtr;
typedef boost::shared_ptr<SourceVector const> SourceVectorConstPtr;

class PersistableSourceVector : public daf::base::Persistable {
    typedef SourceVector::iterator iterator;
public:
    
    PersistableSourceVector(SourceVector sources)
        : _sources(sources) {}
    PersistableSourceVector(SourceVectorPtr sources)
        : _sources(sources) {}
    
    SourceVectorPtr getSources() {return _sources;} 
    SourceVectorConstPtr const getSources() const {return _sources;}
    void SetSources(SourceVectorPtr sources) { _source.swap(sources);}
    
private:
    SourceVectorPtr _sources;
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::SourceVectorFormatter);
}
    
}}} //namespace lsst::afw::detection
