#if !defined(LSST_AFW_DETECTION_MEASUREMENT_H)
#define LSST_AFW_DETECTION_MEASUREMENT_H 1

#include <map>
#include "boost/make_shared.hpp"

#include "lsst/afw/detection/Schema.h"

namespace lsst { namespace afw { namespace detection {

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
    TPtr find(std::string const&name // The name of the desired algorithm
                    ) const {
        for (typename Measurement::const_iterator ptr = begin(); ptr != end(); ++ptr) {
            if ((*ptr)->getSchema()->getComponent() == name) {
                return *ptr;
            }
        }

        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Unknown algorithm " + name);
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
            throw std::runtime_error(msg.str());
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
        throw std::runtime_error(msg.str());
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

/// Print v to os, using dynamic dispatch
template<typename T>
std::ostream &operator<<(std::ostream &os, Measurement<T> const& v) {
    return v.output(os);
}

/************************************************************************************************************/
/*
 * Measure a quantity using a set of algorithms.  Each algorithm will fill one item in the returned
 * Values (a Measurement)
 */
template<typename T, typename ImageT, typename PeakT>
class MeasureQuantity {
public:
    typedef Measurement<T> Values;
    typedef boost::shared_ptr<T> (*makeMeasureQuantityFunc)(typename ImageT::ConstPtr, PeakT const&);
private:
    typedef std::map<std::string, makeMeasureQuantityFunc> AlgorithmList;
public:

    MeasureQuantity(typename ImageT::ConstPtr im) : _im(im), _algorithms() {}
    virtual ~MeasureQuantity() {}

    /// Include the algorithm called name in the list of measurement algorithms to use
    ///
    /// This name is looked up in the registry (\sa declare), and used as the name of the
    /// measurement if you wish to retrieve it using the schema
    ///
    void addAlgorithm(std::string const& name ///< The name of the algorithm
                     ) {
        _algorithms[name] = _lookupAlgorithm(name);
    }
    /// Actually measure im using all requested algorithms, returning the result
    Values measure(PeakT const& peak     ///< approximate position of object's centre
                  ) {
        Values values;

        for (typename AlgorithmList::iterator ptr = _algorithms.begin(); ptr != _algorithms.end(); ++ptr) {
            boost::shared_ptr<T> val = ptr->second(_im, peak);
            val->getSchema()->setComponent(ptr->first); // name this type of measurement (e.g. psf)
            values.add(val);
        }

        return values;
    }

    static bool declare(std::string const& name, makeMeasureQuantityFunc func);
private:
    //
    // The data that we wish to measure
    //
    typename ImageT::ConstPtr _im;
    //
    // The list of algorithms that we wish to use
    //
    AlgorithmList _algorithms;
    //
    // A mapping from names to algorithms
    //
    // _registryWorker must be inline as it contains a critical static variable, _registry
    //    
    typedef std::map<std::string, makeMeasureQuantityFunc> AlgorithmRegistry;

    static inline makeMeasureQuantityFunc _registryWorker(std::string const& name,
                                                          makeMeasureQuantityFunc func);
    static makeMeasureQuantityFunc _lookupAlgorithm(std::string const& name);
    /// The unknown algorithm; used to allow _lookupAlgorithm use _registryWorker
    static boost::shared_ptr<T> _iefbr14(typename ImageT::ConstPtr, PeakT const &) {
        return boost::shared_ptr<T>();
    }
    //
    // Do the real work of measuring things
    //
    // Can't be pure virtual as we create a do-nothing MeasureQuantity which we then add to
    //
    virtual boost::shared_ptr<T> doMeasure(typename ImageT::ConstPtr, PeakT const&) {
        return boost::shared_ptr<T>();
    }
};

/**
 * Support the algorithm registry
 */
template<typename T, typename ImageT, typename PeakT>
typename MeasureQuantity<T, ImageT, PeakT>::makeMeasureQuantityFunc
MeasureQuantity<T, ImageT, PeakT>::_registryWorker(
        std::string const& name,
        typename MeasureQuantity<T, ImageT, PeakT>::makeMeasureQuantityFunc func
                                           )
{
    // N.b. This is a pointer rather than an object as this helps the intel compiler generate a
    // single copy of the _registry across multiple dynamically loaded libraries.  The intel
    // bug ID for RHL's report is 580524
    static typename MeasureQuantity<T, ImageT, PeakT>::AlgorithmRegistry *_registry = NULL;

    if (!_registry) {
        _registry = new MeasureQuantity<T, ImageT, PeakT>::AlgorithmRegistry;
    }

    if (func == _iefbr14) {     // lookup func
        typename MeasureQuantity<T, ImageT, PeakT>::AlgorithmRegistry::const_iterator ptr =
            _registry->find(name);
        
        if (ptr == _registry->end()) {
            throw std::runtime_error("Unknown algorithm " + name);
        }
        
        func = ptr->second;
    } else {                            // register func
        (*_registry)[name] = func;
    }

    return func;
}

/**
 * Register the factory function for a named algorithm
 */
template<typename T, typename ImageT, typename PeakT>
bool MeasureQuantity<T, ImageT, PeakT>::declare(
        std::string const& name,
        typename MeasureQuantity<T, ImageT, PeakT>::makeMeasureQuantityFunc func
                                        )
{
    _registryWorker(name, func);

    return true;
}

/**
 * Return the factory function for a named algorithm
 */
template<typename T, typename ImageT, typename PeakT>
typename MeasureQuantity<T, ImageT, PeakT>::makeMeasureQuantityFunc
MeasureQuantity<T, ImageT, PeakT>::_lookupAlgorithm(std::string const& name)
{
    return _registryWorker(name, _iefbr14);
}

}}}
#endif
