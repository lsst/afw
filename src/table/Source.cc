// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace table {

namespace {

class SourceTableImpl;

class SourceRecordImpl : public SourceRecord {
public:

    explicit SourceRecordImpl(PTR(SourceTable) const & table) : SourceRecord(table) {}

};

class SourceTableImpl : public SourceTable {
public:

    explicit SourceTableImpl(
        Schema const & schema,
        PTR(daf::base::PropertyList) const & metadata,
        PTR(IdFactory) const & idFactory
    ) : SourceTable(schema, metadata, idFactory) {}

    SourceTableImpl(SourceTableImpl const & other) : SourceTable(other) {}

private:

    virtual PTR(TableBase) _clone() const {
        return boost::make_shared<SourceTableImpl>(*this);
    }

    virtual PTR(RecordBase) _makeRecord() {
        PTR(SourceRecord) record = boost::make_shared<SourceRecordImpl>(getSelf<SourceTableImpl>());
        record->setId(getIdFactory()());
        return record;
    }

};

class SourceFitsWriter : public io::FitsWriter {
public:

    explicit SourceFitsWriter(Fits * fits) : io::FitsWriter(fits) {}

protected:
    
    virtual void _writeTable(CONST_PTR(TableBase) const & table);

    virtual void _writeRecord(RecordBase const & record);

private:
    int _spanCol;
    int _peakCol;
};

class SourceFitsReader : public io::FitsReader {
public:

    explicit SourceFitsReader(Fits * fits) : io::FitsReader(fits), _spanCol(-1), _peakCol(-1) {}

protected:

    virtual Schema _readSchema(int nCols=-1);

    virtual PTR(TableBase) _readTable(Schema const & schema);

    virtual PTR(RecordBase) _readRecord(PTR(TableBase) const & table);

private:
    int _spanCol;
    int _peakCol;
};

void SourceFitsWriter::_writeTable(CONST_PTR(TableBase) const & t) {
    CONST_PTR(SourceTable) table = boost::dynamic_pointer_cast<SourceTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Cannot use a SourceFitsWriter on a non-Source table."
        );
    }
    io::FitsWriter::_writeTable(table);
    _spanCol = _fits->addColumn<int>("spans", 0, "footprint spans (y, x0, x1)");
    _peakCol = _fits->addColumn<float>("peaks", 0, "footprint peaks (fx, fy, peakValue)");
    _fits->writeKey("SPANCOL", _spanCol + 1, "Column with footprint spans.");
    _fits->writeKey("PEAKCOL", _peakCol + 1, "Column with footprint peaks (float values).");
    _fits->writeKey("AFW_TYPE", "SOURCE", "Tells lsst::afw to load this as a Source table.");
    // TODO: stuff metadata, slots in header
}

void SourceFitsWriter::_writeRecord(RecordBase const & r) {
    SourceRecord const & record = static_cast<SourceRecord const &>(r);
    io::FitsWriter::_writeRecord(record);
    if (record.getFootprint()) {
        Footprint::SpanList const & spans = record.getFootprint()->getSpans();
        Footprint::PeakList const & peaks = record.getFootprint()->getPeaks();
        if (!spans.empty()) {
            std::vector<int> vec;
            vec.reserve(3 * spans.size());
            for (Footprint::SpanList::const_iterator j = spans.begin(); j != spans.end(); ++j) {
                vec.push_back((**j).getY());
                vec.push_back((**j).getX0());
                vec.push_back((**j).getX1());
            }
            _fits->writeTableArray(_row, _spanCol, vec.size(), &vec.front());
        }
        if (!peaks.empty()) {
            std::vector<float> vec;
            vec.reserve(3 * peaks.size());
            for (Footprint::PeakList::const_iterator j = peaks.begin(); j != peaks.end(); ++j) {
                vec.push_back((**j).getFx());
                vec.push_back((**j).getFy());
                vec.push_back((**j).getPeakValue());}
            _fits->writeTableArray(_row, _peakCol, vec.size(), &vec.front());
        }
    }
}

Schema SourceFitsReader::_readSchema(int nCols) {
    _fits->readKey("SPANCOL", _spanCol);
    if (_fits->status == 0) {
        --_spanCol;
    } else {
        _fits->status = 0;
        _spanCol = -1;
    }
    _fits->readKey("PEAKCOL", _peakCol);
    if (_fits->status == 0) {
        --_peakCol;
    } else {
        _fits->status = 0;
        _peakCol = -1;
    }
    int maxCol = std::min(_spanCol, _peakCol);
    return io::FitsReader::_readSchema(maxCol);
}

PTR(TableBase) SourceFitsReader::_readTable(Schema const & schema) {
    return SourceTable::make(schema);
}

PTR(RecordBase) SourceFitsReader::_readRecord(PTR(TableBase) const & table) {
    PTR(SourceRecord) record = boost::static_pointer_cast<SourceRecord>(io::FitsReader::_readRecord(table));
    if (!record) return record;
    boost::static_pointer_cast<SourceTable>(table)->getIdFactory().notify(record->getId());
    int spanElementCount = (_spanCol >= 0) ? _fits->getTableArraySize(_row, _spanCol) : 0;
    int peakElementCount = (_peakCol >= 0) ? _fits->getTableArraySize(_row, _peakCol) : 0;
    if (spanElementCount || peakElementCount) {
        PTR(Footprint) fp = boost::make_shared<Footprint>();
        if (spanElementCount) {
            if (spanElementCount % 3) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        _fits->fptr, _fits->status,
                        boost::format("Number of span elements (%d) must divisible by 3 (row %d)")
                        % spanElementCount % _row
                    )
                );
            }
            std::vector<int> spanElements(spanElementCount);
            _fits->readTableArray(_row, _spanCol, spanElementCount, &spanElements.front());
            std::vector<int>::iterator j = spanElements.begin();
            while (j != spanElements.end()) {
                int y = *j++;
                int x0 = *j++;
                int x1 = *j++;
                fp->addSpan(y, x0, x1);
            }
        }
        if (peakElementCount) {
            if (peakElementCount % 3) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    afw::fits::makeErrorMessage(
                        _fits->fptr, _fits->status,
                        boost::format("Number of peak elements (%d) must divisible by 3 (row %d)")
                        % peakElementCount % _row
                    )
                );
            }
            std::vector<float> peakElements(peakElementCount);
            _fits->readTableArray(_row, _peakCol, peakElementCount, &peakElements.front());
            std::vector<float>::iterator j = peakElements.begin();
            while (j != peakElements.end()) {
                float x = *j++;
                float y = *j++;
                float value = *j++;
                fp->getPeaks().push_back(boost::make_shared<detection::Peak>(x, y, value));
            }
        }
        record->setFootprint(fp);
    }
    return record;
}

static io::FitsReader::FactoryT<SourceFitsReader> sourceFitsReaderFactory("SOURCE");

} // anonymous

SourceRecord::SourceRecord(PTR(SourceTable) const & table) : RecordBase(table) {}

void SourceRecord::_assign(RecordBase const & other) {
    try {
        SourceRecord const & s = dynamic_cast<SourceRecord const &>(other);
        _footprint = s._footprint;
    } catch (std::bad_cast&) {}
}

PTR(SourceTable) SourceTable::make(
    Schema const & schema,
    PTR(daf::base::PropertyList) const & metadata,
    PTR(IdFactory) const & idFactory
) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Schema for Source must contain at least the keys defined by getMinimalSchema()."
        );
    }
    return boost::make_shared<SourceTableImpl>(schema, metadata, idFactory);
}

SourceTable::SourceTable(
    Schema const & schema,
    PTR(daf::base::PropertyList) const & metadata,
    PTR(IdFactory) const & idFactory
) : TableBase(schema), _metadata(metadata), _idFactory(idFactory)
{
    if (!_idFactory) _idFactory = IdFactory::makeSimple();
    if (!_metadata) _metadata = boost::make_shared<daf::base::PropertyList>();
}

SourceTable::SourceTable(SourceTable const & other) :
    TableBase(other),
    _metadata(boost::static_pointer_cast<daf::base::PropertyList>(other._metadata->deepCopy())),
    _idFactory(other._idFactory->clone()),
    _slotFlux(other._slotFlux), _slotCentroid(other._slotCentroid), _slotShape(other._slotShape)
{}

SourceTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID for source");
    parent = schema.addField<RecordId>("parent", "unique ID of parent source");
    sky = schema.addField<float>("sky", "sky background at location of source", "DN/pix");
    skyErr = schema.addField<float>("sky.err", "sky background uncertainty at location of source",
                                    "DN/pix");
    coord = schema.addField<Coord>("coord", "position of source in ra/dec", "radians");
}

SourceTable::MinimalSchema & SourceTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter) SourceTable::makeFitsWriter(io::FitsWriter::Fits * fits) const {
    return boost::make_shared<SourceFitsWriter>(fits);
}

template class SourceSet<SourceRecord>;
template class SourceSet<SourceRecord const>;













#if 0

SourceTable SourceTable::readFits(std::string const & filename) {
    fits::Fits file = fits::Fits::openFile(filename.c_str(), true);
    int spanCol = -1, peakCol = -1;
    file.readKey("SPANCOL", spanCol);
    if (file.status == 0) {
        --spanCol;
    } else {
        file.status = 0;
        spanCol = -1;
    }
    file.readKey("PEAKCOL", peakCol);
    if (file.status == 0) {
        --peakCol;
    } else {
        file.status = 0;
        peakCol = -1;
    }
    int maxCol = std::min(spanCol, peakCol);
    Schema schema = fits::readFitsHeader(file, true, maxCol);
    int nRecords = 0;
    file.readKey("NAXIS2", nRecords);
    SourceTable table(schema, nRecords);
    fits::readFitsRecords(file, table);
    if (spanCol >= 0 || peakCol >= 0) {
        int row = 0;
        for (SourceTable::Iterator i = table.begin(); i != table.end(); ++i, ++row) {
        }
    }
    file.closeFile();
    file.checkStatus();
    return table;
}

#endif

}}} // namespace lsst::afw::table
