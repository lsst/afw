// -*- lsst-c++ -*-

#include <cstring>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/SchemaMapper.h"

namespace lsst { namespace afw { namespace table {

namespace {

//----- Copy functor for copying with mapper ----------------------------------------------------------------

struct CopyValue {

    template <typename U>
    void operator()(Key<U> const & inputKey, Key<U> const & outputKey) const {
        typename Field<U>::Element const * inputElem = _inputRecord->getElement(inputKey);
        std::copy(inputElem, inputElem + inputKey.getElementCount(), _outputRecord->getElement(outputKey));
    }

    void operator()(Key<Flag> const & inputKey, Key<Flag> const & outputKey) const {
        _outputRecord->set(outputKey, _inputRecord->get(inputKey));
    }

    CopyValue(RecordBase const * inputRecord, RecordBase * outputRecord) :
        _inputRecord(inputRecord), _outputRecord(outputRecord)
    {}

private:
    RecordBase const * _inputRecord;
    RecordBase * _outputRecord;
};

} // anonymous

void RecordBase::assign(RecordBase const & other) {
    if (this->getSchema() != other.getSchema()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Unequal schemas in record assignment."
        );
    }
    std::memcpy(_data, other._data, this->getSchema().getRecordSize());
    this->_assign(other);
}

void RecordBase::assign(RecordBase const & other, SchemaMapper const & mapper) {
    if (other.getSchema() != mapper.getInputSchema()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Unequal schemas between input record and mapper."
        );
    }
    if (this->getSchema() != mapper.getOutputSchema()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Unequal schemas between output record and mapper."
        );
    }
    mapper.forEach(CopyValue(&other, this));
    this->_assign(other);
}


}}} // namespace lsst::afw::table
