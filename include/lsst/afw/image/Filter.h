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
//         Class encapsulating an identifier for an LSST filter.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_IMAGE_FILTER_H
#define LSST_AFW_IMAGE_FILTER_H

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "lsst/base.h"
#include "lsst/daf/base/PropertySet.h"


namespace lsst {
namespace afw {
namespace image {

/**
 * Describe the properties of a Filter (e.g. effective wavelength)
 */
class FilterProperty final {
public:
    explicit FilterProperty(std::string const& name, double lambdaEff, double lambdaMin = NAN,
                            double lambdaMax = NAN, bool force = false)
            : _name(name), _lambdaEff(lambdaEff), _lambdaMin(lambdaMin), _lambdaMax(lambdaMax) {
        _insert(force);
    }
    /**
     * @param name name of filter
     * @param prop values describing the Filter
     * @param force Allow this name to replace a previous one
     */
    explicit FilterProperty(std::string const& name,
                            lsst::daf::base::PropertySet const& prop = lsst::daf::base::PropertySet(),
                            bool force = false);

    FilterProperty(FilterProperty const&) = default;
    FilterProperty(FilterProperty&&) noexcept = default;
    FilterProperty& operator=(FilterProperty const&) = default;
    FilterProperty& operator=(FilterProperty&&) noexcept = default;
    ~FilterProperty() noexcept = default;

    /**
     * Return a filter's name
     */
    std::string const& getName() const noexcept { return _name; }
    /**
     * Return the filter's effective wavelength (nm)
     */
    double getLambdaEff() const noexcept { return _lambdaEff; }
    /**
     * Return the filter's minimum wavelength (nm) where the transmission is above 1% of the maximum.
     */
    double getLambdaMin() const noexcept { return _lambdaMin; }
    /**
     * Return the filter's maximum wavelength (nm) where the transmission is above 1% of the maximum.
     */
    double getLambdaMax() const noexcept { return _lambdaMax; }
    /**
     * Return true iff two FilterProperties are identical
     *
     * @param rhs Object to compare with this
     */
    bool operator==(FilterProperty const& rhs) const noexcept;
    /**
     * Return true iff rhs != this
     */
    bool operator!=(FilterProperty const& rhs  ///< Object to compare with this
                    ) const noexcept {
        return !(*this == rhs);
    }
    /**
     * Clear all definitions
     */
    static void reset() { _initRegistry(); }

    /**
     * Lookup the properties of a filter "name"
     *
     * @param name name of desired filter
     */
    static FilterProperty const& lookup(std::string const& name);

private:
    typedef std::unordered_map<std::string, FilterProperty> PropertyMap;

    /**
     * Initialise the Filter registry
     */
    static void _initRegistry();
    /**
     * Insert FilterProperty into registry
     *
     * @param force Allow this name to replace a previous one?
     */
    void _insert(bool force = false);

    std::string _name;  // name of filter
    double _lambdaEff;  // effective wavelength (nm)
    double _lambdaMin;  // minimum wavelength (nm)
    double _lambdaMax;  // maximum wavelength (nm)

    static PropertyMap* _propertyMap;  // mapping from name -> FilterProperty
};

/**
 * Holds an integer identifier for an LSST filter.
 */
class Filter final {
public:
    static int const AUTO;
    static int const UNKNOWN;

    /**
     * Creates a Filter with the given name
     */
    explicit Filter(std::string const& name,  ///< Name of filter
                    bool const force = false  ///< Allow us to construct an unknown Filter
                    )
            : _id(_lookup(name, force)), _name(name) {}
    /**
     * Creates a Filter with the given identifier
     */
    explicit Filter(int id = UNKNOWN  ///< Id number of desired filter
                    )
            : _id(id), _name(_lookup(id)) {}
    /**
     * Create a Filter from a PropertySet (e.g. a FITS header)
     *
     * @param metadata Metadata to process (e.g. a IFITS header)
     * @param force Allow us to construct an unknown Filter
     */
    explicit Filter(std::shared_ptr<lsst::daf::base::PropertySet const> metadata, bool const force = false);

    Filter(Filter const&) = default;
    Filter(Filter&&) noexcept = default;
    Filter& operator=(Filter const&) = default;
    Filter& operator=(Filter&&) noexcept = default;
    ~Filter() noexcept = default;

    /**
     * Are two filters identical?
     */
    bool operator==(Filter const& rhs) const noexcept;
    bool operator!=(Filter const& rhs) const noexcept { return !(*this == rhs); }

    /**
     * Return a Filter's integral id
     */
    int getId() const noexcept { return _id; }
    /**
     * Return a Filter's name
     */
    std::string const& getName() const noexcept { return _name; }
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

    /**
     * Return a Filter's FilterProperty
     */
    FilterProperty const& getFilterProperty() const;
    /**
     * Clear all definitions
     */
    static void reset() { _initRegistry(); }
    /**
     * Define a filter name to have the specified id
     *
     * If id == Filter::AUTO a value will be chosen for you.
     *
     * It is an error to attempt to change a name's id (unless you specify force)
     */
    static int define(FilterProperty const& filterProperty, int id = AUTO, bool force = false);
    /**
     * Define an alias for a filter
     *
     * @param oldName old name for Filter
     * @param newName new name for Filter
     * @param force force an alias even if newName is already in use
     */
    static int defineAlias(std::string const& oldName, std::string const& newName, bool force = false);

    /**
     * Return a list of known filters
     */
    static std::vector<std::string> getNames();

private:
    typedef std::unordered_map<std::string, std::string const> AliasMap;
    typedef std::unordered_map<std::string, unsigned int const> NameMap;
    typedef std::unordered_map<unsigned int, std::string const> IdMap;

    /**
     * Initialise the Filter registry
     */
    static void _initRegistry();
    /**
     * Lookup the ID associated with a name
     *
     * @param name Name of filter
     * @param force return an invalid ID, but don't throw, if name is unknown
     */
    static int _lookup(std::string const& name, bool const force = false);
    /**
     * Lookup the name associated with an ID
     */
    static std::string const& _lookup(int id);

    int _id;
    std::string _name;

    static int _id0;             // next Id to use
    static AliasMap* _aliasMap;  // mapping from alias -> name
    static IdMap* _idMap;        // mapping from id -> name
    static NameMap* _nameMap;    // mapping from name -> id
};

namespace detail {
/**
 * Remove Filter-related keywords from the metadata
 *
 * @param[in, out] metadata Metadata to be stripped
 * @return Number of keywords stripped
 */
int stripFilterKeywords(std::shared_ptr<lsst::daf::base::PropertySet> metadata);
}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_FILTER_H
