// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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
#include <unordered_map>
#include <vector>
#include "boost/shared_ptr.hpp"
#include "lsst/base.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace daf {
    namespace base {
        class PropertySet;
    }
}

namespace afw {
namespace image {

/**
 * Describe the properties of a Filter (e.g. effective wavelength)
 */
class FilterProperty {
public:
    boost::shared_ptr<FilterProperty> Ptr;
    boost::shared_ptr<FilterProperty const> ConstPtr;

    explicit FilterProperty(std::string const& name, double lambdaEff, bool force=false) :
        _name(name), _lambdaEff(lambdaEff) { _insert(force); }
    explicit FilterProperty(
        std::string const& name,
        lsst::daf::base::PropertySet const& prop=lsst::daf::base::PropertySet(),
        bool force=false
        );
    explicit FilterProperty(std::string const& name, lsst::pex::policy::Policy const& pol, bool force=false);
    /**
     * Return a filter's name
     */
    std::string const& getName() const { return _name; }
    /**
     * Return the filter's effective wavelength (nm)
     */
    double getLambdaEff() const { return _lambdaEff; }
    /*
     * Compare two FilterProperties
     */
    bool operator==(FilterProperty const& rhs) const;
    /**
     * Return true iff rhs != this
     */
    bool operator!=(FilterProperty const& rhs ///< Object to compare with this
                   ) const { return !(*this == rhs); }
    /**
     * Clear all definitions
     */
    static void reset() { _initRegistry(); }

    static FilterProperty const& lookup(std::string const& name);
private:
    typedef std::unordered_map<std::string, FilterProperty> PropertyMap;

    static void _initRegistry();
    void _insert(bool force=false);

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
     * Create a Filter from a PropertySet (e.g. a FITS header)
     */
    explicit Filter(CONST_PTR(lsst::daf::base::PropertySet), bool const force=false);

    /*
     * Compare two Filters
     */
    bool operator==(Filter const& rhs) const;
    bool operator!=(Filter const& rhs) const { return !(*this == rhs); }

    /**
     * Return a Filter's integral id
     */
    int getId() const { return _id; }
    /**
     * Return a Filter's name
     */
    std::string const& getName() const { return _name; }
    /**
     * Return a filter's canonical name
     *
     * I.e. if this filter's an alias, return the name of the aliased Filter
     */
    std::string const& getCanonicalName() const { return _lookup(_id); }
    /**
     * Return all aliases by which this filter is known
     *
     * The list excludes the canonical name
     */
    std::vector<std::string> getAliases() const;

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
    typedef std::unordered_map<std::string, std::string const> AliasMap;
    typedef std::unordered_map<std::string, unsigned int const> NameMap;
    typedef std::unordered_map<unsigned int, std::string const> IdMap;

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

namespace detail {
    int stripFilterKeywords(PTR(lsst::daf::base::PropertySet) metadata);
}

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_FILTER_H
