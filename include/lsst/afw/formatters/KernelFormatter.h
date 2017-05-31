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

#ifndef LSST_AFW_MATH_KERNELFORMATTER_H
#define LSST_AFW_MATH_KERNELFORMATTER_H

/*
 * Interface for KernelFormatter class
 */

#include <lsst/afw/math/Kernel.h>
#include <lsst/daf/base/Persistable.h>
#include <lsst/daf/persistence/Formatter.h>
#include <lsst/daf/persistence/FormatterStorage.h>
#include <lsst/pex/policy/Policy.h>

namespace lsst {
namespace afw {
namespace formatters {

/**
 * Formatter for persistence of Kernel instances.
 */
class KernelFormatter : public lsst::daf::persistence::Formatter {
public:
    /** Minimal destructor.
     */
    virtual ~KernelFormatter(void);

    virtual void write(lsst::daf::base::Persistable const* persistable,
                       std::shared_ptr<lsst::daf::persistence::FormatterStorage> storage,
                       std::shared_ptr<lsst::daf::base::PropertySet> additionalData);

    virtual lsst::daf::base::Persistable* read(std::shared_ptr<lsst::daf::persistence::FormatterStorage> storage,
                                               std::shared_ptr<lsst::daf::base::PropertySet> additionalData);

    virtual void update(lsst::daf::base::Persistable* persistable,
                        std::shared_ptr<lsst::daf::persistence::FormatterStorage> storage,
                        std::shared_ptr<lsst::daf::base::PropertySet> additionalData);

    /** Serialize a Kernel to a Boost archive.  Handles text or XML
     * archives, input or output.
     * @param[in,out] ar Boost archive
     * @param[in] version version of the KernelFormatter
     * @param[in,out] persistable Pointer to the Kernel as a Persistable
     */
    template <class Archive>
    static void delegateSerialize(Archive& ar, unsigned int const version,
                                  lsst::daf::base::Persistable* persistable);

private:
    /** Constructor.
     * @param[in] policy Policy for configuring this Formatter
     */
    explicit KernelFormatter(std::shared_ptr<lsst::pex::policy::Policy> policy);

    std::shared_ptr<lsst::pex::policy::Policy> _policy;

    /** Factory method for KernelFormatter.
     * @param[in] policy Policy for configuring the KernelFormatter
     * @returns Shared pointer to a new instance
     */
    static std::shared_ptr<lsst::daf::persistence::Formatter> createInstance(
            std::shared_ptr<lsst::pex::policy::Policy> policy);

    /** Register this Formatter subclass through a static instance of
     * FormatterRegistration.
     */
    static lsst::daf::persistence::FormatterRegistration kernelRegistration;
    static lsst::daf::persistence::FormatterRegistration fixedKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration analyticKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration deltaFunctionKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration linearCombinationKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration separableKernelRegistration;
};
}
}
}  // namespace lsst::daf::persistence

#endif
