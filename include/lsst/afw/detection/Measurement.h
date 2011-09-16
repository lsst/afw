#if !defined(LSST_AFW_DETECTION_MEASUREMENT_H)
#define LSST_AFW_DETECTION_MEASUREMENT_H 1

#include <map>
#include "boost/format.hpp"
#include "boost/make_shared.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/xml_oarchive.hpp"
#include "boost/archive/xml_iarchive.hpp"
#include "boost/serialization/export.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/vector.hpp"

#include "lsst/base.h"
#include "lsst/utils/Demangle.h"
#include "lsst/utils/ieee.h"
#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/afw/detection/Schema.h"

namespace lsst { namespace afw { namespace detection {

#ifndef SWIG
using boost::serialization::make_nvp;
#endif

class Source;

/************************************************************************************************************/
/*
 * This is a base class for measurements of a set of quantities.  For example, we'll inherit from this
 * class to support Photometric measurements of various kinds.
 *
 * Measurement is a Go4 "Composite", able to hold individual measurements of type T (e.g. a PSF magnitude and
 * its error) or a set of measurements (PSF, Aperture, and model magnitudes, each with their errors and maybe
 * ancillary numbers such as a Sersic index or axis ratio).
 */
template<typename T>
class Measurement {
public:
    typedef boost::shared_ptr<T> TPtr;  // make TPtr public for swig's sake
    typedef boost::shared_ptr<Measurement> Ptr;
    typedef typename std::vector<boost::shared_ptr<T> >::iterator iterator;
    typedef typename std::vector<boost::shared_ptr<T> >::const_iterator const_iterator;

    Measurement() : _measuredValues(), _mySchema(new Schema) { }
    virtual ~Measurement() {}

    /// Are there any known algorithms?
    bool empty() const {
        return _measuredValues.empty();
    }
    /// Return an iterator to the start of the set
    iterator begin() {
        return _measuredValues.begin();
    }
    /// Return a const iterator to the start of the set
    const_iterator begin() const {
        return _measuredValues.begin();
    }
    /// Return an iterator to the end of the set
    iterator end() {
        return _measuredValues.end();
    }
    /// Return an const iterator to the end of the set
    const_iterator end() const {
        return _measuredValues.end();
    }
#if 1
    /// Return an iterator to the named algorithm
    const_iterator find_iter(std::string const&name // The name of the desired algorithm
                 ) const {
        for (typename Measurement::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            if ((*ptr)->getSchema()->getComponent() == name) {
                return ptr;
            }
        }

        return end();
    }
#endif
    /// Return a T::Ptr
    /// \throws lsst::pex::exceptions::NotFoundException
    TPtr find(std::string const&name=std::string("") // The name of the desired algorithm
                    ) const {
        typename Measurement::const_iterator ptr = begin();

        if (name == "") {               // no name specified, ...
            if (ptr + 1 == end()) {     // ... but only one registered algorithm
                return *ptr;
            }
        }

        for (typename Measurement::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            if ((*ptr)->getSchema()->getComponent() == name) {
                return *ptr;
            }
        }

        if (name == "") {
            throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                              "You may only omit the algorithm's name if exactly one is registered");
        } else {
            throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Unknown algorithm " + name);
        }
    }

    /// Add a (shared_pointer to) an individual measurement of type T
    void add(TPtr val) {
        _measuredValues.push_back(val);
    }

    /// Print all the values to os;  note that this is a virtual function called by operator<<
    virtual std::ostream &output(std::ostream &os) const {
        for (typename Measurement::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            if (ptr != begin()) {
                os << " ";
            }
            os << "[" << **ptr << "]";
        }

        return os;
    }

    /// Return our Measurement's schema
    virtual Schema::Ptr getSchema() const {
        return _mySchema;
    }
    /// Resize the list of individual measurements of type T
    void resize(int n) { _data.resize(n); }

    /// Allocate enough space in _data to hold all values declared in the schema
    void init() {
        defineSchema(_mySchema);
        resize(getSchema()->size());    // getSchema() is virtual, but this is called from most-derived ctor
    }
    /**
     * Return the name of the algorithm used to measure this component
     */
    std::string const& getAlgorithm() const {
        return getSchema()->getComponent();
    }
    /**
     * Return some Measurement's value as a double given its Schema
     *
     * \sa getAsLong() to return as a long
     */
    double get(Schema const& se        ///< The schema entry for the value you want
              ) const {
        return getAsType<double>(se);
    }
    /**
     * Return some Measurement's value as a double given its Schema
     *
     * \sa getAsLong() to return as a long
     */
    double get(unsigned int i,                 ///< Index to set
               Schema const& se        ///< The schema entry for the value you want
              ) const {
        return getAsType<double>(i, se);
    }
    /**
     * Return some Measurement's value as a long given its Schema
     *
     * \sa get() to return as a double
     */
    long getAsLong(Schema const& se   ///< The schema entry for the value you want
                  ) const {
        return getAsType<long>(se);
    }
    /**
     * Return some Measurement's value as a long given its Schema
     *
     * \sa get() to return as a double
     */
    long getAsLong(unsigned int i,      ///< Index to set
                   Schema const& se     ///< The schema entry for the value you want
                  ) const {
        return getAsType<long>(i, se);
    }
    /**
     * Return a T by name and component
     *
     * Note that all values are returned as double; use the (protected) templated form of get if you need more
     * control
     *
     * \sa getAsLong() to return as a long
     */
    double get(std::string const& name,        ///< the name within T
               std::string const& component="" ///< the name within the set of measurements
              ) const {
        return get(getSchema()->find(name, component));
    }

    /**
     * Return an element T of an array by index name and component
     *
     * Note that all values are returned as double; use the (protected) templated form of get if you need more
     * control
     *
     * \sa getAsLong() to return as a long
     */
    double get(unsigned int i,                 ///< Index to set
               std::string const& name,        ///< the name within T
               std::string const& component="" ///< the name within the set of measurements
              ) const {
        return get(i, getSchema()->find(name, component));
    }             
protected:
    /// Fast compile-time-computed access to set the values of _data
    template<unsigned int INDEX, typename U>
    void set(U value                    ///< Desired value
            ) {
        assert(INDEX < _data.size());
        _data[INDEX] = value;
    }

    /// Fast compile-time-computed access to set the values of _data
    template<unsigned int INDEX, typename U>
    void set(unsigned int i,            ///< Index to set
             U value                    ///< Desired value
            ) {
        assert(INDEX + i < _data.size());
        _data[INDEX + i] = value;
    }

    /// Fast compile-time-computed access to retrieve the values of _data
    template<unsigned int INDEX, typename U>
    U get() const {
        assert(INDEX < _data.size());
        return boost::any_cast<U>(_data[INDEX]);
    }

    /// Fast compile-time-computed access to retrieve the values of _data as an array
    template<unsigned int INDEX, typename U>
    U get(unsigned int i                ///< Desired index
         ) const {
        assert(INDEX + i < _data.size());
        return boost::any_cast<U>(_data[INDEX + i]);
    }

private:
    virtual void defineSchema(Schema::Ptr ) {}

    /// Return a value as the specified type
    template<typename U>
    U getAsType(Schema const& se        ///< The schema entry for the value you want
               ) const {
        return getAsType<U>(0, se);
    }

    /// Return a value as the specified type
    template<typename U>
    U getAsType(unsigned int i,         ///< Index into array (if se is an array)
                Schema const& se        ///< The schema entry for the value you want
               ) const {
        unsigned int const index = se.getIndex() + i;
        if (index >= _data.size()) {
            std::ostringstream msg;
            if (index - i < _data.size()) { // the problem is that i takes us out of range
                msg << "Index " << i << " is out of range for " << se.getName() <<
                    "[0," << se.getDimen() - 1 << "]";
            } else {
                msg << "Index " << index << " out of range [0," << _data.size() << "] for " << se.getName();
            }
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, msg.str());
        }
        boost::any const& val = _data[index];

        switch (se.getType()) {
          case Schema::CHAR:
            return static_cast<U>(boost::any_cast<char>(val));
          case Schema::SHORT:
            return static_cast<U>(boost::any_cast<short>(val));
          case Schema::INT:
            return static_cast<U>(boost::any_cast<int>(val));
          case Schema::LONG:
#if defined(__ICC)
#pragma warning (push)
#pragma warning (disable: 2259)          // conversion from "long" to "double" may lose significant bits
#endif
            return static_cast<U>(boost::any_cast<long>(val));
#if defined(__ICC)
#pragma warning (pop)
#endif
          case Schema::FLOAT:
            return static_cast<U>(boost::any_cast<float>(val));
          case Schema::DOUBLE:
            return static_cast<U>(boost::any_cast<double>(val));
          default:
            break;
        }
        
        std::ostringstream msg;
        msg << "Unable to retrieve value of type " << se.getType() << " for " << se.getName();
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, msg.str());
    }


    /// boost::serialization methods
    friend class boost::serialization::access;

    template <class Archive> void serialize(Archive& ar,
              unsigned int const version) {
        size_t dataLen;
        if (Archive::is_saving::value) {
            dataLen = _data.size();
        }
        ar & make_nvp("dataLen", dataLen);
        if (Archive::is_loading::value) {
            _data.resize(dataLen);
        }
        for (size_t i = 0; i < dataLen; ++i) {
            if (Archive::is_saving::value) {
                if (_data[i].type() == typeid(void)) {
                    int which = -1;
                    ar & make_nvp("which", which);
                } else if (_data[i].type() == typeid(char)) {
                    int which = 1;
                    char value = boost::any_cast<char>(_data[i]);
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else if (_data[i].type() == typeid(short)) {
                    int which = 2;
                    short value = boost::any_cast<short>(_data[i]);
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else if (_data[i].type() == typeid(int)) {
                    int which = 3;
                    int value = boost::any_cast<int>(_data[i]);
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else if (_data[i].type() == typeid(long)) {
                    int which = 4;
                    long value = boost::any_cast<long>(_data[i]);
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else if (_data[i].type() == typeid(float)) {
                    int which = 5;
                    float value = boost::any_cast<float>(_data[i]);
                    int fpClass = 0;
                    if (lsst::utils::isnan(value)) {
                        fpClass = 1;
                    } else if (lsst::utils::isinf(value)) {
                        fpClass = value > 0.0 ? 2 : 3;
                    }
                    which += fpClass;
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else if (_data[i].type() == typeid(double)) {
                    int which = 9;
                    double value = boost::any_cast<double>(_data[i]);
                    int fpClass = 0;
                    if (lsst::utils::isnan(value)) {
                        fpClass = 1;
                        value = 0;
                    } else if (lsst::utils::isinf(value)) {
                        fpClass = value > 0.0 ? 2 : 3;
                        value = 0;
                    }
                    which += fpClass;
                    ar & make_nvp("which", which) & make_nvp("value", value);
                } else {
                    std::ostringstream msg;
                    msg << "Unable to convert measurement for persistence: type "
                        << _data[i].type().name() << " at position " << i;
                    throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, msg.str());
                }
            }
            if (Archive::is_loading::value) {
                int which;
                ar & make_nvp("which", which);
                switch (which) {
                case -1:
                    break;
                case 1:
                    {
                        char value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 2:
                    {
                        short value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 3:
                    {
                        int value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 4:
                    {
                        long value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 5:
                    {
                        float value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 6:
                    {
                        float value;
                        ar & make_nvp("value", value);
                        _data[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                case 7:
                    {
                        float value;
                        ar & make_nvp("value", value);
                        _data[i] = std::numeric_limits<float>::infinity();
                    }
                    break;
                case 8:
                    {
                        float value;
                        ar & make_nvp("value", value);
                        _data[i] = -std::numeric_limits<float>::infinity();
                    }
                    break;
                case 9:
                    {
                        double value;
                        ar & make_nvp("value", value);
                        _data[i] = value;
                    }
                    break;
                case 10:
                    {
                        double value;
                        ar & make_nvp("value", value);
                        _data[i] = std::numeric_limits<double>::quiet_NaN();
                    }
                    break;
                case 11:
                    {
                        double value;
                        ar & make_nvp("value", value);
                        _data[i] = std::numeric_limits<double>::infinity();
                    }
                    break;
                case 12:
                    {
                        double value;
                        ar & make_nvp("value", value);
                        _data[i] = -std::numeric_limits<double>::infinity();
                    }
                    break;
                default:
                    std::ostringstream msg;
                    msg << "Unable to recognize type for retrieval: type "
                        << which << " at position " << i;
                    throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, msg.str());
                    break;
                }
            }
        }

        ar & make_nvp("values", _measuredValues);
        ar & make_nvp("schema", _mySchema);
    }


    typedef std::vector<boost::any> DataStore;
    // The elements of T (if a leaf)
    DataStore _data;

    // The set of Ts (if a composite)
    std::vector<TPtr> _measuredValues;
public:
    std::vector<TPtr> const& getValues() const { return  _measuredValues; }
private:

    // T's schema
    Schema::Ptr _mySchema;
};

#define LSST_SERIALIZE_PARENT(c) \
    friend class boost::serialization::access; \
    template <class Archive> \
    void serialize(Archive& ar, unsigned int const version) { \
        ar & boost::serialization::make_nvp("base", boost::serialization::base_object< c >(*this)); \
    }

#ifdef SWIG
#define LSST_REGISTER_SERIALIZER(c) /**/
#else
#define LSST_REGISTER_SERIALIZER(c) \
    BOOST_CLASS_EXPORT_GUID(c, #c)
#endif

/// Print v to os, using dynamic dispatch
template<typename T>
std::ostream &operator<<(std::ostream &os, Measurement<T> const& v) {
    return v.output(os);
}


}}}
#endif
