// -*- lsst-c++ -*-
#include "lsst/afw/table/fits.h"
#include "lsst/afw/table/Source.h"

namespace lsst { namespace afw { namespace table {

void SourceTable::writeFits(std::string const & filename) {
    fits::Fits file = fits::Fits::createFile(filename.c_str());
    file.checkStatus();
    fits::writeFitsHeader(file, getSchema(), true);
    int spanCol = file.addColumn<int>("spans", 0, "footprint spans (y, x0, x1)");
    int peakCol = file.addColumn<float>("peaks", 0, "footprint peaks (fx, fy, peakValue)");
    file.writeKey<int>("SPANCOL", spanCol + 1, "Column with footprint spans.");
    file.writeKey<int>("PEAKCOL", peakCol + 1, "Column with footprint peaks (float values).");
    file.checkStatus();
    fits::writeFitsRecords(file, *this);
    int row = 0;
    for (Iterator i = begin(); i != end(); ++i, ++row) {
        if (i->hasFootprint()) {
            Footprint::SpanList const & spans = i->getFootprint().getSpans();
            Footprint::PeakList const & peaks = i->getFootprint().getPeaks();
            if (!spans.empty()) {
                std::vector<int> vec;
                vec.reserve(3 * spans.size());
                for (Footprint::SpanList::const_iterator j = spans.begin(); j != spans.end(); ++j) {
                    vec.push_back((**j).getY());
                    vec.push_back((**j).getX0());
                    vec.push_back((**j).getX1());
                }
                file.writeTableArray(row, spanCol, vec.size(), &vec.front());
            }
            if (!peaks.empty()) {
                std::vector<float> vec;
                vec.reserve(3 * peaks.size());
                for (Footprint::PeakList::const_iterator j = peaks.begin(); j != peaks.end(); ++j) {
                    vec.push_back((**j).getFx());
                    vec.push_back((**j).getFy());
                    vec.push_back((**j).getPeakValue());}
                file.writeTableArray(row, peakCol, vec.size(), &vec.front());
            }
        }
    }
    file.checkStatus();
    file.closeFile();
    file.checkStatus();
}

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
            int spanElementCount = (spanCol >= 0) ? file.getTableArraySize(row, spanCol) : 0;
            int peakElementCount = (peakCol >= 0) ? file.getTableArraySize(row, peakCol) : 0;
            if (spanElementCount || peakElementCount) {
                Footprint fp;
                if (spanElementCount) {
                    if (spanElementCount % 3) {
                        throw LSST_EXCEPT(
                            afw::fits::FitsError,
                            afw::fits::makeErrorMessage(
                                file.fptr, file.status,
                                boost::format("Number of span elements (%d) must divisible by 3 (row %d)")
                                % spanElementCount % row
                            )
                        );
                    }
                    std::vector<int> spanElements(spanElementCount);
                    file.readTableArray(row, spanCol, spanElementCount, &spanElements.front());
                    std::vector<int>::iterator j = spanElements.begin();
                    while (j != spanElements.end()) {
                        int y = *j++;
                        int x0 = *j++;
                        int x1 = *j++;
                        fp.addSpan(y, x0, x1);
                    }
                }
                if (peakElementCount) {
                    if (peakElementCount % 3) {
                        throw LSST_EXCEPT(
                            afw::fits::FitsError,
                            afw::fits::makeErrorMessage(
                                file.fptr, file.status,
                                boost::format("Number of peak elements (%d) must divisible by 3 (row %d)")
                                % peakElementCount % row
                            )
                        );
                    }
                    std::vector<float> peakElements(peakElementCount);
                    file.readTableArray(row, peakCol, peakElementCount, &peakElements.front());
                    std::vector<float>::iterator j = peakElements.begin();
                    while (j != peakElements.end()) {
                        float x = *j++;
                        float y = *j++;
                        float value = *j++;
                        fp.getPeaks().push_back(boost::make_shared<detection::Peak>(x, y, value));
                    }
                }
                i->setFootprint(fp);
            }
        }
    }
    file.closeFile();
    file.checkStatus();
    return table;
}

}}} // namespace lsst::afw::table
