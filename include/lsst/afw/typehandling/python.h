/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_TYPEHANDLING_PYTHON_H
#define LSST_AFW_TYPEHANDLING_PYTHON_H

#include "pybind11/pybind11.h"

#include <string>

#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * "Trampoline" for Storable to let it be used as a base class in Python.
 *
 * Subclasses of Storable that are wrapped in %pybind11 should have a similar
 * helper that subclasses `StorableHelper<subclass>`. This helper can be
 * skipped if the subclass neither adds any virtual methods nor implements
 * any abstract methods.
 *
 * @tparam Base the exact (most specific) class being wrapped
 *
 * @see [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
 */
template <class Base = Storable>
class StorableHelper : public Base, public pybind11::trampoline_self_life_support {
public:
    using Base::Base;

    /**
     * Delegating constructor for wrapped class.
     *
     * While we would like to simply inherit base class constructors, when doing so, we cannot
     * change their access specifiers.  One consequence is that it's not possible to use inheritance
     * to expose a protected constructor to python.  The alternative, used here, is to create a new
     * public constructor that delegates to the base class public or protected constructor with the
     * same signature.
     *
     * @tparam Args  Variadic type specification
     * @param ...args  Arguments to forward to the Base class constructor.
     */
    template<typename... Args>
    explicit StorableHelper<Base>(Args... args) : Base(args...) {}

    std::shared_ptr<Storable> cloneStorable() const override {
        /* __deepcopy__ takes an optional dict, but PYBIND11_OVERLOAD_* won't
         * compile unless you give it arguments that work for the C++ method
         */
        PYBIND11_OVERLOAD_NAME(std::shared_ptr<Storable>, Base, "__deepcopy__", cloneStorable, );
    }

    std::string toString() const override {
        PYBIND11_OVERLOAD_NAME(std::string, Base, "__repr__", toString, );
    }

    std::size_t hash_value() const override {
        PYBIND11_OVERLOAD_NAME(std::size_t, Base, "__hash__", hash_value, );
    }

    bool equals(Storable const& other) const noexcept override {
        PYBIND11_OVERLOAD_NAME(bool, Base, "__eq__", equals, other);
    }

    bool isPersistable() const noexcept override {
        PYBIND11_OVERLOAD(
            bool, Base, isPersistable
        );
    }

    std::string getPersistenceName() const override {
        PYBIND11_OVERLOAD_NAME(
            std::string, Base, "_getPersistenceName", getPersistenceName
        );
    }

    std::string getPythonModule() const override {
        PYBIND11_OVERLOAD_NAME(
            std::string, Base, "_getPythonModule", getPythonModule
        );
    }

    void write(table::io::OutputArchiveHandle& handle) const override;
};

std::string declareGenericMapRestrictions(std::string const& className, std::string const& keyName);

/**
 * StorableHelper persistence.
 *
 * We cannot directly override `StorableHelper::write` or `StorableHelperFactory::read` in python using
 * PYBIND11_OVERLOAD* macros as the required argument c++ types (OutputArchiveHandle, InputArchive,
 * CatalogVector) are not bound to python types.  Instead, we allow python subclasses of Storable to define
 * `_write` and `_read` methods that return/accept a string serialization of the object.  The `_write` method
 * should take no arguments (besides self) and return a byte string.  The `_read` method should be a static
 * method that takes a string and returns an instance of the python class.  The python pickle module may be
 * useful for handling the string serialization / deserialization.
 *
 * The `StorableHelper::write` and `StorableHelperFactory::read` c++ methods will directly call the python
 * methods described above when required.
 */

namespace {

class StorableHelperPersistenceHelper {
public:
    table::Schema schema;
    table::Key<table::Array<std::uint8_t>> bytes;

    static StorableHelperPersistenceHelper const &get() {
        static StorableHelperPersistenceHelper instance;
        return instance;
    }

    // No copying
    StorableHelperPersistenceHelper(StorableHelperPersistenceHelper const &) = delete;
    StorableHelperPersistenceHelper &operator=(StorableHelperPersistenceHelper const &) = delete;

    // No moving
    StorableHelperPersistenceHelper(StorableHelperPersistenceHelper &&) = delete;
    StorableHelperPersistenceHelper &operator=(StorableHelperPersistenceHelper &&) = delete;

private:
    StorableHelperPersistenceHelper() :
        schema(),
        bytes(schema.addField<table::Array<std::uint8_t>>(
            "bytes", "an opaque bytestring representation of a Storable", ""
        ))
    {}
};


class StorableHelperFactory : public table::io::PersistableFactory {
public:
    StorableHelperFactory(std::string const &module, std::string const &name) :
        table::io::PersistableFactory(name),
        _module(module),
        _name(name)
    {}

    std::shared_ptr<table::io::Persistable> read(
        InputArchive const &archive,
        CatalogVector const &catalogs
    ) const override {
        pybind11::gil_scoped_acquire gil;
        auto const &keys = StorableHelperPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        auto const &record = catalogs.front().front();
        std::string stringRep = formatters::bytesToString(record.get(keys.bytes));
        auto cls = pybind11::module::import(_module.c_str()).attr(_name.c_str());
        auto pyobj = cls.attr("_read")(pybind11::bytes(stringRep));
        return pyobj.cast<std::shared_ptr<Storable>>();
    }

private:
    std::string _module;
    std::string _name;
};

}  // namespace


template <typename Base>
void StorableHelper<Base>::write(table::io::OutputArchiveHandle& handle) const {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload = pybind11::get_overload(static_cast<const Base *>(this), "_write");
    if (!overload)
        throw std::runtime_error("Cannot find StorableHelper _write overload");
    auto o = overload().cast<std::string>();
    auto const &keys = StorableHelperPersistenceHelper::get();
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = cat.addNew();
    record->set(keys.bytes, formatters::stringToBytes(o));
    handle.saveCatalog(cat);
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
