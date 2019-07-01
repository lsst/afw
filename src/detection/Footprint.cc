
/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/geom/transformFactory.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace lsst {
namespace afw {

template std::shared_ptr<detection::Footprint> table::io::PersistableFacade<
        detection::Footprint>::dynamicCast(std::shared_ptr<table::io::Persistable> const&);

namespace detection {

Footprint::Footprint(std::shared_ptr<geom::SpanSet> inputSpans, lsst::geom::Box2I const& region)
        : _spans(inputSpans), _peaks(PeakTable::makeMinimalSchema()), _region(region) {}

Footprint::Footprint(std::shared_ptr<geom::SpanSet> inputSpans, afw::table::Schema const& peakSchema,
                     lsst::geom::Box2I const& region)
        : _spans(inputSpans), _peaks(peakSchema), _region(region) {}

void Footprint::setSpans(std::shared_ptr<geom::SpanSet> otherSpanSet) { _spans = otherSpanSet; }

std::shared_ptr<PeakRecord> Footprint::addPeak(float fx, float fy, float height) {
    std::shared_ptr<PeakRecord> p = getPeaks().addNew();
    p->setIx(fx);
    p->setIy(fy);
    p->setFx(fx);
    p->setFy(fy);
    p->setPeakValue(height);
    return p;
}

void Footprint::sortPeaks(afw::table::Key<float> const& key) {
    auto validatedKey = key.isValid() ? key : PeakTable::getPeakValueKey();
    getPeaks().sort([&validatedKey](detection::PeakRecord const& a, detection::PeakRecord const& b) {
        return a.get(validatedKey) > b.get(validatedKey);
    });
}

void Footprint::shift(int dx, int dy) {
    setSpans(getSpans()->shiftedBy(dx, dy));
    for (auto& peak : getPeaks()) {
        peak.setIx(peak.getIx() + dx);
        peak.setIy(peak.getIy() + dy);
        peak.setFx(peak.getFx() + dx);
        peak.setFy(peak.getFy() + dy);
    }
}

void Footprint::clipTo(lsst::geom::Box2I const& box) {
    setSpans(getSpans()->clippedTo(box));
    removeOrphanPeaks();
}

bool Footprint::contains(lsst::geom::Point2I const& pix) const { return getSpans()->contains(pix); }

std::shared_ptr<Footprint> Footprint::transform(std::shared_ptr<geom::SkyWcs> source,
                                                std::shared_ptr<geom::SkyWcs> target,
                                                lsst::geom::Box2I const& region, bool doClip) const {
    auto srcToTarget = geom::makeWcsPairTransform(*source, *target);
    return transform(*srcToTarget, region, doClip);
}

std::shared_ptr<Footprint> Footprint::transform(lsst::geom::LinearTransform const& t,
                                                lsst::geom::Box2I const& region, bool doClip) const {
    return transform(lsst::geom::AffineTransform(t), region, doClip);
}

std::shared_ptr<Footprint> Footprint::transform(lsst::geom::AffineTransform const& t,
                                                lsst::geom::Box2I const& region, bool doClip) const {
    return transform(*geom::makeTransform(t), region, doClip);
}

std::shared_ptr<Footprint> Footprint::transform(geom::TransformPoint2ToPoint2 const& t,
                                                lsst::geom::Box2I const& region, bool doClip) const {
    // Transfrom the SpanSet first
    auto transformedSpan = getSpans()->transformedBy(t);
    // Use this new SpanSet and the peakSchema to create a new Footprint
    auto newFootprint = std::make_shared<Footprint>(transformedSpan, getPeaks().getSchema(), region);
    // now populate the new Footprint with transformed Peaks
    std::vector<lsst::geom::Point2D> peakPosList;
    peakPosList.reserve(_peaks.size());
    for (auto const& peak : getPeaks()) {
        peakPosList.emplace_back(peak.getF());
    }
    auto newPeakPosList = t.applyForward(peakPosList);
    auto newPeakPos = newPeakPosList.cbegin();
    for (auto peak = getPeaks().cbegin(), endPeak = getPeaks().cend(); peak != endPeak;
         ++peak, ++newPeakPos) {
        newFootprint->addPeak(newPeakPos->getX(), newPeakPos->getY(), peak->getPeakValue());
    }
    if (doClip) {
        newFootprint->clipTo(region);
    }
    return newFootprint;
}

void Footprint::dilate(int r, geom::Stencil s) { setSpans(getSpans()->dilated(r, s)); }

void Footprint::dilate(geom::SpanSet const& other) { setSpans(getSpans()->dilated(other)); }

void Footprint::erode(int r, geom::Stencil s) {
    setSpans(getSpans()->eroded(r, s));
    removeOrphanPeaks();
}

void Footprint::erode(geom::SpanSet const& other) {
    setSpans(getSpans()->eroded(other));
    removeOrphanPeaks();
}

void Footprint::removeOrphanPeaks() {
    for (auto iter = getPeaks().begin(); iter != getPeaks().end(); ++iter) {
        if (!getSpans()->contains(lsst::geom::Point2I(iter->getIx(), iter->getIy()))) {
            iter = getPeaks().erase(iter);
            --iter;
        }
    }
}

std::vector<std::shared_ptr<Footprint>> Footprint::split() const {
    auto splitSpanSets = getSpans()->split();
    std::vector<std::shared_ptr<Footprint>> footprintList;
    footprintList.reserve(splitSpanSets.size());
    for (auto& spanPtr : splitSpanSets) {
        auto tmpFootprintPointer = std::make_shared<Footprint>(spanPtr, getPeaks().getSchema(), getRegion());
        tmpFootprintPointer->_peaks = getPeaks();
        // No need to remove any peaks, as there is only one Footprint, so it will
        // simply be a copy of the original
        if (splitSpanSets.size() > 1) {
            tmpFootprintPointer->removeOrphanPeaks();
        }
        footprintList.push_back(std::move(tmpFootprintPointer));
    }
    return footprintList;
}

bool Footprint::operator==(Footprint const& other) const {
    /* If the peakCatalogs are not the same length the Footprints can't be equal */
    if (getPeaks().size() != other.getPeaks().size()) {
        return false;
    }
    /* Check that for every peak in the PeakCatalog there is a corresponding peak
     * in the other, and if not return false
     */
    for (auto const& selfPeak : getPeaks()) {
        bool match = false;
        for (auto const& otherPeak : other.getPeaks()) {
            if (selfPeak.getI() == otherPeak.getI() && selfPeak.getF() == otherPeak.getF() &&
                selfPeak.getPeakValue() == otherPeak.getPeakValue()) {
                match = true;
                break;
            }
        }
        if (!match) {
            return false;
        }
    }
    /* At this point the PeakCatalogs have evaluated true, compare the SpanSets
     */
    return *(getSpans()) == *(other.getSpans());
}

namespace {
std::string getFootprintPersistenceName() { return "Footprint"; }

class LegacyFootprintPersistenceHelper {
public:
    table::Schema spanSchema;
    table::Key<int> spanY;
    table::Key<int> spanX0;
    table::Key<int> spanX1;

    static LegacyFootprintPersistenceHelper const& get() {
        static LegacyFootprintPersistenceHelper instance;
        return instance;
    }

    // No copying
    LegacyFootprintPersistenceHelper(const LegacyFootprintPersistenceHelper&) = delete;
    LegacyFootprintPersistenceHelper& operator=(const LegacyFootprintPersistenceHelper&) = delete;

    // No moving
    LegacyFootprintPersistenceHelper(LegacyFootprintPersistenceHelper&&) = delete;
    LegacyFootprintPersistenceHelper& operator=(LegacyFootprintPersistenceHelper&&) = delete;

private:
    LegacyFootprintPersistenceHelper()
            : spanSchema(),
              spanY(spanSchema.addField<int>("y", "The row of the span", "pixel")),
              spanX0(spanSchema.addField<int>("x0", "First column of span (inclusive)", "pixel")),
              spanX1(spanSchema.addField<int>("x1", "Second column of span (inclusive)", "pixel")) {}
};

std::pair<afw::table::Schema&, table::Key<int>&> spanSetPersistenceHelper() {
    static afw::table::Schema spanSetIdSchema;
    static int initialize = true;
    static table::Key<int> idKey;
    if (initialize) {
        idKey = spanSetIdSchema.addField<int>("id", "id of the SpanSet catalog");
        initialize = false;
    }
    std::pair<afw::table::Schema&, table::Key<int>&> returnPair(spanSetIdSchema, idKey);
    return returnPair;
}
}  // end anonymous namespace

class FootprintFactory : public table::io::PersistableFactory {
public:
    std::shared_ptr<afw::table::io::Persistable> read(
            afw::table::io::InputArchive const& archive,
            afw::table::io::CatalogVector const& catalogs) const override {
        // Verify there are two catalogs
        LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
        std::shared_ptr<Footprint> loadedFootprint =
                detection::Footprint::readSpanSet(catalogs.front(), archive);
        // Now read in the PeakCatalog records
        detection::Footprint::readPeaks(catalogs.back(), *loadedFootprint);
        return loadedFootprint;
    }

    explicit FootprintFactory(std::string const& name) : afw::table::io::PersistableFactory(name) {}
};

namespace {
// Insert the factory into the registry (instantiating an instance is sufficient, because the
// the code that does the work is in the base class ctor)
FootprintFactory registration(getFootprintPersistenceName());
}  // end anonymous namespace

std::string Footprint::getPersistenceName() const { return getFootprintPersistenceName(); }

void Footprint::write(afw::table::io::OutputArchiveHandle& handle) const {
    // get the span schema and key
    auto const keys = spanSetPersistenceHelper();
    // create the output catalog
    afw::table::BaseCatalog spanSetCat = handle.makeCatalog(keys.first);
    // create a record that will hold the ID of the recursively saved SpanSet
    auto record = spanSetCat.addNew();
    record->set(keys.second, handle.put(getSpans()));
    handle.saveCatalog(spanSetCat);
    // save the peaks into a catalog
    afw::table::BaseCatalog peakCat = handle.makeCatalog(getPeaks().getSchema());
    peakCat.insert(peakCat.end(), getPeaks().begin(), getPeaks().end(), true);
    handle.saveCatalog(peakCat);
}

std::unique_ptr<Footprint> Footprint::readSpanSet(afw::table::BaseCatalog const& catalog,
                                                  afw::table::io::InputArchive const& archive) {
    int fieldCount = catalog.getSchema().getFieldCount();
    LSST_ARCHIVE_ASSERT(fieldCount == 1 || fieldCount == 3);
    std::shared_ptr<geom::SpanSet> loadedSpanSet;
    if (fieldCount == 1) {
        // This is a new style footprint with a SpanSet as a member, treat accordingly
        auto const schemaAndKey = spanSetPersistenceHelper();
        int persistedSpanSetId = catalog.front().get(schemaAndKey.second);
        loadedSpanSet = std::dynamic_pointer_cast<geom::SpanSet>(archive.get(persistedSpanSetId));
    } else {
        // This block is for an old style footprint load.
        auto const& keys = LegacyFootprintPersistenceHelper::get();
        std::vector<geom::Span> tempVec;
        tempVec.reserve(catalog.size());
        for (auto const& val : catalog) {
            tempVec.push_back(geom::Span(val.get(keys.spanY), val.get(keys.spanX0), val.get(keys.spanX1)));
        }
        loadedSpanSet = std::make_shared<geom::SpanSet>(std::move(tempVec));
    }
    auto loadedFootprint = std::unique_ptr<Footprint>(new Footprint(loadedSpanSet));
    return loadedFootprint;
}

void Footprint::readPeaks(afw::table::BaseCatalog const& peakCat, Footprint& loadedFootprint) {
    using namespace std::string_literals;
    if (!peakCat.getSchema().contains(PeakTable::makeMinimalSchema())) {
        // need to handle an older form of Peak persistence for backwards compatibility
        afw::table::SchemaMapper mapper(peakCat.getSchema());
        mapper.addMinimalSchema(PeakTable::makeMinimalSchema());
        afw::table::Key<float> oldX = peakCat.getSchema()["x"];
        afw::table::Key<float> oldY = peakCat.getSchema()["y"];
        afw::table::Key<float> oldPeakValue = peakCat.getSchema()["value"];
        mapper.addMapping(oldX, "f.x"s);
        mapper.addMapping(oldY, "f.y"s);
        mapper.addMapping(oldPeakValue, "peakValue"s);
        loadedFootprint.setPeakSchema(mapper.getOutputSchema());
        auto peaks = loadedFootprint.getPeaks();
        peaks.reserve(peakCat.size());
        for (auto const& peak : peakCat) {
            auto newPeak = peaks.addNew();
            newPeak->assign(peak, mapper);
            newPeak->setIx(static_cast<int>(newPeak->getFx()));
            newPeak->setIy(static_cast<int>(newPeak->getFy()));
        }
        return;
    }
    loadedFootprint.setPeakSchema(peakCat.getSchema());
    auto& peaks = loadedFootprint.getPeaks();
    peaks.reserve(peakCat.size());
    for (auto const& peak : peakCat) {
        peaks.addNew()->assign(peak);
    }
}

std::shared_ptr<Footprint> mergeFootprints(Footprint const& footprint1, Footprint const& footprint2) {
    // Bail out early if the schemas are not the same
    if (footprint1.getPeaks().getSchema() != footprint2.getPeaks().getSchema()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "Cannot merge Footprints with different Schemas");
    }

    // Merge the SpanSets
    auto unionedSpanSet = footprint1.getSpans()->union_(*(footprint2.getSpans()));

    // Construct merged Footprint
    auto mergedFootprint = std::make_shared<Footprint>(unionedSpanSet, footprint1.getPeaks().getSchema());
    // Copy over the peaks from both footprints
    mergedFootprint->setPeakCatalog(PeakCatalog(footprint1.getPeaks().getTable()));
    PeakCatalog& peaks = mergedFootprint->getPeaks();
    peaks.reserve(footprint1.getPeaks().size() + footprint2.getPeaks().size());
    peaks.insert(peaks.end(), footprint1.getPeaks().begin(), footprint1.getPeaks().end(), true);
    peaks.insert(peaks.end(), footprint2.getPeaks().begin(), footprint2.getPeaks().end(), true);

    // Sort the PeaksCatalog according to value
    mergedFootprint->sortPeaks();

    return mergedFootprint;
}

std::vector<lsst::geom::Box2I> footprintToBBoxList(Footprint const& footprint) {
    typedef std::uint16_t PixelT;
    lsst::geom::Box2I fpBBox = footprint.getBBox();
    std::shared_ptr<image::Image<PixelT>> idImage(new image::Image<PixelT>(fpBBox));
    *idImage = 0;
    int const height = fpBBox.getHeight();
    lsst::geom::Extent2I shift(fpBBox.getMinX(), fpBBox.getMinY());
    footprint.getSpans()->setImage(*idImage, static_cast<PixelT>(1), fpBBox, true);

    std::vector<lsst::geom::Box2I> bboxes;
    /*
     * Our strategy is to find a row of pixels in the Footprint and interpret it as the first
     * row of a rectangular set of pixels.  We then extend this rectangle upwards as far as it
     * will go, and define that as a BBox.  We clear all those pixels, and repeat until there
     * are none left.  I.e. a Footprint will get cut up like this:
     *
     *       .555...
     *       22.3314
     *       22.331.
     *       .000.1.
     * (as shown in Footprint_1.py)
     */

    int y0 = 0;  // the first row with non-zero pixels in it
    while (y0 < height) {
        lsst::geom::Box2I bbox;  // our next BBox
        for (int y = y0; y != height; ++y) {
            // Look for a set pixel in this row
            image::Image<PixelT>::x_iterator begin = idImage->row_begin(y), end = idImage->row_end(y);
            image::Image<PixelT>::x_iterator first = std::find(begin, end, 1);

            if (first != end) {  // A pixel is set in this row
                image::Image<PixelT>::x_iterator last = std::find(first, end, 0) - 1;
                int const x0 = first - begin;
                int const x1 = last - begin;

                std::fill(first, last + 1, 0);  // clear pixels; we don't want to see them again

                bbox.include(lsst::geom::Point2I(x0, y));  // the LLC
                bbox.include(lsst::geom::Point2I(x1, y));  // the LRC; initial guess for URC

                // we found at least one pixel so extend the BBox upwards
                for (++y; y != height; ++y) {
                    if (std::find(idImage->at(x0, y), idImage->at(x1 + 1, y), 0) != idImage->at(x1 + 1, y)) {
                        break;  // some pixels weren't set, so the BBox stops here, (actually in previous row)
                    }
                    std::fill(idImage->at(x0, y), idImage->at(x1 + 1, y), 0);

                    bbox.include(lsst::geom::Point2I(x1, y));  // the new URC
                }

                bbox.shift(shift);
                bboxes.push_back(bbox);
            } else {
                y0 = y + 1;
            }
            break;
        }
    }

    return bboxes;
}

void Footprint::setPeakSchema(afw::table::Schema const& peakSchema) {
    setPeakCatalog(PeakCatalog(peakSchema));
}

void Footprint::setPeakCatalog(PeakCatalog const& otherPeaks) {
    if (!getPeaks().empty()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Cannot change the PeakCatalog unless it is empty");
    }
    // this syntax doesn't work in Python, which is why this method has to exist
    getPeaks() = otherPeaks;
}
}  // namespace detection
}  // namespace afw
}  // namespace lsst
