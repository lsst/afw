// -*- lsst-c++ -*-

#include <cstdio>
#include <complex>

#include "fitsio.h"
#include "boost/cstdint.hpp"

#include "lsst/afw/table/TableBase.h"
#include "lsst/daf/base/PropertySet.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct MakeRequiredHeaderStrings {

    static char const * getFormat(boost::int16_t *) { return "%dI"; }
    static char const * getFormat(boost::uint16_t *) { return "%dU"; }
    static char const * getFormat(boost::int32_t *) { return "%dJ"; }
    static char const * getFormat(boost::uint32_t *) { return "%dV"; }
    static char const * getFormat(boost::int64_t *) { return "%dK"; }
    static char const * getFormat(float *) { return "%dE"; }
    static char const * getFormat(double *) { return "%dK"; }
    static char const * getFormat(std::complex<float> *) { return "%dC"; }
    static char const * getFormat(std::complex<double> *) { return "%dM"; }

    template <typename T>
    void operator()(LayoutItem<T> const & item) const {
        ttype->push_back(item.field.getName());
        tform->push_back(
            (boost::format(getFormat((typename Field<T>::Element*)0)) % item.field.getElementCount()).str()
        );
    }

    std::vector<std::string> * ttype;
    std::vector<std::string> * tform;
};

struct UpdateHeaderStrings {

    template <typename T>
    static char const * getUnitsComment(LayoutItem<T> const & item) { return 0; }

    template <typename T>
    static char const * getUnitsComment(LayoutItem< Point<T> > const & item) { return "{x, y}"; }

    template <typename T>
    static char const * getUnitsComment(LayoutItem< Shape<T> > const & item) { return "{xx, yy, xy}"; }

    template <typename T>
    static char const * getUnitsComment(LayoutItem< Covariance<T> > const & item) {
        return "{[0,0], [0,1], [1,1], ...}";
    }

    template <typename T>
    static char const * getUnitsComment(LayoutItem< Covariance< Point<T> > > const & item) {
        return "{[x,x], [x,y], [y,y]}";
    }

    template <typename T>
    static char const * getUnitsComment(LayoutItem< Covariance< Shape<T> > > const & item) {
        return "{[xx,xx], [xx,yy], [yy,yy], [xx,xy], [yy,xy], [xy,xy]}";
    }

    template <typename T>
    void operator()(LayoutItem<T> const & item) const {
        char ttype[9];
        std::sprintf(ttype, "ttype%d", n);
        fits_update_key_str(
            fptr, ttype, 
            const_cast<char*>(item.field.getName().c_str()),
            const_cast<char*>(item.field.getDoc().c_str()),
            status
        );
        if (!item.field.getUnits().empty()) {
            char tunit[9];
            std::sprintf(tunit, "tunit%d", n);
            fits_update_key_str(
                fptr, tunit, 
                const_cast<char*>(item.field.getUnits().c_str()),
                const_cast<char*>(getUnitsComment(item)),
                status
            );
        }
    }

    mutable int n;
    int * status;
    fitsfile * fptr;
};

void initializeFitsTableHeader(fitsfile * fptr, Layout const & layout, int nRecords) {
    int status = 0;
    std::vector<std::string> ttypeVector;
    std::vector<std::string> tformVector;
    MakeRequiredHeaderStrings func1 = { &ttypeVector, &tformVector };
    layout.forEach(func1);
    int nFields = ttypeVector.size();
    if (nFields >= 1000) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Tables with more than 999 columns cannot be written in FITS format."
        );
    }
    boost::scoped_array<char*> ttypeArray(new char*[nFields]);
    boost::scoped_array<char*> tformArray(new char*[nFields]);
    for (int i = 0; i < nFields; ++i) {
        ttypeArray[i] = const_cast<char*>(ttypeVector[i].c_str());
        tformArray[i] = const_cast<char*>(tformVector[i].c_str());
    }
    fits_create_tbl(fptr, BINARY_TBL, nRecords, nFields, ttypeArray.get(), tformArray.get(), 0, 0, &status);
    // TODO error checking
    UpdateHeaderStrings func2 = { 0, &status, fptr };
    layout.forEach(func2);
    // TODO error checking
}

} // anonymous

void TableBase::_writeFits(
    std::string const & name,
    CONST_PTR(daf::base::PropertySet) const & metadata,
    std::string const & mode
) const {
    
}

void TableBase::_writeFits(
    std::string const & name,
    LayoutMapper const & mapper,
    CONST_PTR(daf::base::PropertySet) const & metadata,
    std::string const & mode
) const {
    
}

}}} // namespace lsst::afw::table
