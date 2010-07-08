// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Class encapsulating an identifier for an LSST filter.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_IMAGE_FILTER_H
#define LSST_AFW_IMAGE_FILTER_H

#include <string>
#include "boost/shared_ptr.hpp"
#include "lsst/tr1/unordered_map.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * Describe the properties of a Filter (e.g. effective wavelength)
 */
class FilterProperty {
public:
    boost::shared_ptr<FilterProperty> Ptr;
    boost::shared_ptr<FilterProperty const> ConstPtr;
    
    FilterProperty(std::string const& name, lsst::pex::policy::Policy const& pol=lsst::pex::policy::Policy(),
                   bool force=false);
    /**
     * Return a filter's name
     */
    std::string const& getName() const { return _name; }
    /**
     * Return the filter's effective wavelength (nm)
     */
    double getLambdaEff() const { return _lambdaEff; }
    /**
     * Clear all definitions
     */
    static void reset() { _initRegistry(); }

    static FilterProperty const& lookup(std::string const& name);
private:
    typedef std::tr1::unordered_map<std::string const, FilterProperty> PropertyMap;

    static void _initRegistry();

    std::string _name;                  // name of filter
    double _lambdaEff;                  // effective wavelength (nm)

    static PropertyMap *_propertyMap;   // mapping from name -> FilterProperty
};

/************************************************************************************************************/
/*!
 * \brief  Holds an integer identifier for an LSST filter.
 */
class Filter
{
public :
    enum { AUTO=-1, UNKNOWN=-1 };
    /*!
     * Creates a Filter with the given name
     */
    explicit Filter(std::string const& name, ///< Name of filter
                    bool const force=false   ///< Allow us to construct an unknown Filter
                   ) : _id(_lookup(name, force)), _name(name) {}
    /**
     * Creates a Filter with the given identifier
     */
    explicit Filter(int id=UNKNOWN      ///< Id number of desired filter
                   ) : _id(id), _name(_lookup(id)) {}
    /**
     * Return a Filter's integral id
     */
    int getId() const { return _id; }
    /**
     * Return a Filter's name
     */
    std::string const& getName() const { return _name; }
    
    FilterProperty const& getFilterProperty() const;
    /**
     * Clear all definitions
     */
    static void reset() { _initRegistry(); }
    /*
     * Define a filter
     */
    static int define(FilterProperty const& filterProperty, int id=AUTO, bool force=false);
    /*
     * Define an alias for a filter
     */
    static int defineAlias(std::string const& oldName, std::string const& newName, bool force=false);

    static std::vector<std::string> getNames();
private :
    typedef std::tr1::unordered_map<std::string const, std::string const> AliasMap;
    typedef std::tr1::unordered_map<std::string const, unsigned int const> NameMap;
    typedef std::tr1::unordered_map<unsigned int const, std::string const> IdMap;

    static void _initRegistry();
    static int _lookup(std::string const& name, bool const force=false);
    static std::string const& _lookup(int id);

    int _id;
    std::string _name;

    static int _id0;                    // next Id to use
    static AliasMap *_aliasMap;         // mapping from alias -> name
    static IdMap *_idMap;               // mapping from id -> name
    static NameMap *_nameMap;           // mapping from name -> id
};

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_FILTER_H
