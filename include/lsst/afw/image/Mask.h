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

/*
 * Image pixel bitmasks
 */

#ifndef LSST_AFW_IMAGE_MASK_H
#define LSST_AFW_IMAGE_MASK_H

#include <list>
#include <map>
#include <string>

#include <memory>

#include "lsst/base.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/ImageBase.h"
#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/image/detail/MaskDict.h"
#include "lsst/afw/fitsDefaults.h"

namespace lsst {
namespace afw {
namespace image {

// all masks will initially be instantiated with the same pixel type
namespace detail {
/// tag for a Mask
struct Mask_tag : public detail::basic_tag {};
}  // namespace detail

/**
 * Represent a 2-dimensional array of bitmask pixels
 *
 * Some mask planes are always defined (although you can add more with Mask::addMaskPlane). See
 * `MaskDict._addInitialMaskPlanes` for their defintions.
 */
template <typename MaskPixelT = lsst::afw::image::MaskPixel>
class Mask : public ImageBase<MaskPixelT> {
public:
    using MaskPlaneDict = detail::MaskPlaneDict;
    using MaskPlaneDocDict = detail::MaskPlaneDocDict;

    using image_category = detail::Mask_tag;

    /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
    template <typename MaskPT = MaskPixelT>
    struct ImageTypeFactory {
        /// Return the desired type
        using type = Mask<MaskPT>;
    };

    // Constructors
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param width number of columns
     * @param height number of rows
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]
    explicit Mask(unsigned int width, unsigned int height, MaskPlaneDict const& planeDefs);
    explicit Mask(unsigned int width, unsigned int height,
                  std::shared_ptr<detail::MaskDict> maskDict = nullptr);
    /**
     * Construct a Mask initialized to a specified value
     *
     * @param width number of columns
     * @param height number of rows
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]
    explicit Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                  MaskPlaneDict const& planeDefs);
    explicit Mask(unsigned int width, unsigned int height, MaskPixelT initialValue,
                  std::shared_ptr<detail::MaskDict> maskDict = nullptr);
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param dimensions Number of columns, rows
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]
    explicit Mask(lsst::geom::Extent2I const& dimensions, MaskPlaneDict const& planeDefs);
    explicit Mask(lsst::geom::Extent2I const& dimensions = lsst::geom::Extent2I(),
                  std::shared_ptr<detail::MaskDict> maskDict = nullptr);
    /**
     * Construct a Mask initialized to a specified value
     *
     * @param dimensions Number of columns, rows
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]
    explicit Mask(lsst::geom::Extent2I const& dimensions, MaskPixelT initialValue,
                  MaskPlaneDict const& planeDefs);
    explicit Mask(lsst::geom::Extent2I const& dimensions, MaskPixelT initialValue,
                  std::shared_ptr<detail::MaskDict> maskDict = nullptr);
    /**
     * Construct a Mask initialized to 0x0
     *
     * @param bbox Desired number of columns/rows and origin
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]]
    explicit Mask(lsst::geom::Box2I const& bbox, MaskPlaneDict const& planeDefs);

    // Nullptr constructor makes a default MaskDict
    explicit Mask(lsst::geom::Box2I const& bbox, std::shared_ptr<detail::MaskDict> maskDict = nullptr);

    /**
     * Construct a Mask initialized to a specified value
     *
     * @param bbox Desired number of columns/rows and origin
     * @param initialValue Initial value
     * @param planeDefs desired mask planes
     */
    // [[deprecated("Replaced by a shared_ptr interface to MaskDict. Will be removed after v27.")]
    explicit Mask(lsst::geom::Box2I const& bbox, MaskPixelT initialValue, MaskPlaneDict const& planeDefs);
    explicit Mask(lsst::geom::Box2I const& bbox, MaskPixelT initialValue,
                  std::shared_ptr<detail::MaskDict> maskDict = nullptr);
    /**
     *  Construct a Mask by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in]      hdu           HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                                of afw::fits::DEFAULT_HDU reads the Primary HDU unless it is empty,
     *                                in which case it reads the first extension HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      allowUnsafe   Permit reading into the requested pixel type even
     *                                when on-disk values may overflow or truncate.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(std::string const& fileName, int hdu = fits::DEFAULT_HDU,
                  std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                          std::shared_ptr<lsst::daf::base::PropertySet>(),
                  lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
                  bool conformMasks = false, bool allowUnsafe = false);

    /**
     *  Construct a Mask by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in]      hdu           HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                                of afw::fits::DEFAULT_HDU reads the Primary HDU unless it is empty,
     *                                in which case it reads the first extension HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      allowUnsafe   Permit reading into the requested pixel type even
     *                                when on-disk values may overflow or truncate.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(fits::MemFileManager& manager, int hdu = fits::DEFAULT_HDU,
                  std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                          std::shared_ptr<lsst::daf::base::PropertySet>(),
                  lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
                  bool conformMasks = false, bool allowUnsafe = false);

    /**
     *  Construct a Mask from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from, already at the desired HDU.
     *  @param[in,out]  metadata      Metadata read from the header (may be null).
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *  @param[in]      allowUnsafe   Permit reading into the requested pixel type even
     *                                when on-disk values may overflow or truncate.
     *
     *  The meaning of the bitplanes is given in the header.  If conformMasks is false (default),
     *  the bitvalues will be changed to match those in Mask's plane dictionary.  If it's true, the
     *  bitvalues will be left alone, but Mask's dictionary will be modified to match the
     *  on-disk version.
     */
    explicit Mask(fits::Fits& fitsfile,
                  std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                          std::shared_ptr<lsst::daf::base::PropertySet>(),
                  lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
                  bool conformMasks = false, bool allowUnsafe = false);

    // generalised copy constructor
    template <typename OtherPixelT>
    Mask(Mask<OtherPixelT> const& rhs, const bool deep)
            : image::ImageBase<MaskPixelT>(rhs, deep), _maskDict(rhs._maskDict) {}

    /**
     * Construct a Mask from another Mask
     *
     * @param src mask to copy
     * @param deep deep copy? (construct a view with shared pixels if false)
     */
    Mask(const Mask& src, const bool deep = false);
    Mask(Mask&& src);
    ~Mask() override;
    /**
     * Construct a Mask from a subregion of another Mask
     *
     * @param src mask to copy
     * @param bbox subregion to copy
     * @param origin coordinate system of the bbox
     * @param deep deep copy? (construct a view with shared pixels if false)
     */
    Mask(const Mask& src, const lsst::geom::Box2I& bbox, ImageOrigin const origin = PARENT,
         const bool deep = false);

    explicit Mask(ndarray::Array<MaskPixelT, 2, 1> const& array, bool deep = false,
                  lsst::geom::Point2I const& xy0 = lsst::geom::Point2I());

    void swap(Mask& rhs);
    // Operators

    Mask& operator=(MaskPixelT const rhs);
    Mask& operator=(const Mask& rhs);
    Mask& operator=(Mask&& rhs);

    /// OR a Mask into a Mask
    Mask& operator|=(Mask const& rhs);
    /// OR a bitmask into a Mask
    Mask& operator|=(MaskPixelT const rhs);

    /// AND a Mask into a Mask
    Mask& operator&=(Mask const& rhs);
    /// AND a bitmask into a Mask
    Mask& operator&=(MaskPixelT const rhs);

    /**
     * Return a subimage corresponding to the given box.
     *
     * @param  bbox   Bounding box of the subimage returned.
     * @param  origin Origin bbox is rleative to; PARENT accounts for xy0, LOCAL does not.
     * @return        A subimage view into this.
     *
     * This method is wrapped as __getitem__ in Python.
     *
     * @note This method permits mutable views to be obtained from const
     *       references to images (just as the copy constructor does).
     *       This is an intrinsic flaw in Image's design.
     */
    Mask subset(lsst::geom::Box2I const& bbox, ImageOrigin origin = PARENT) const {
        return Mask(*this, bbox, origin, false);
    }

    /// Return a subimage corresponding to the given box (interpreted as PARENT coordinates).
    Mask operator[](lsst::geom::Box2I const& bbox) const { return subset(bbox); }

    using ImageBase<MaskPixelT>::operator[];

    /// XOR a Mask into a Mask
    Mask& operator^=(Mask const& rhs);
    /// XOR a bitmask into a Mask
    Mask& operator^=(MaskPixelT const rhs);

    /**
     * get a reference to the specified pixel
     *
     * @param x x index
     * @param y y index
     */
    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y);
    /**
     * get the specified pixel (const version)
     *
     * @param x x index
     * @param y y index
     */
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y) const;
    /**
     * is the specified mask plane set in the specified pixel?
     *
     * @param x x index
     * @param y y index
     * @param plane plane ID
     */
    bool operator()(int x, int y, int plane) const;
    /**
     * get a reference to the specified pixel checking array bounds
     *
     * @param x x index
     * @param y y index
     * @param check Check array bounds?
     */
    typename ImageBase<MaskPixelT>::PixelReference operator()(int x, int y, CheckIndices const& check);
    /**
     * get the specified pixel with array checking (const version)
     *
     * @param x x index
     * @param y y index
     * @param check Check array bounds?
     */
    typename ImageBase<MaskPixelT>::PixelConstReference operator()(int x, int y,
                                                                   CheckIndices const& check) const;
    /**
     * is the specified mask plane set in the specified pixel, checking array bounds?
     *
     * @param x x index
     * @param y y index
     * @param plane plane ID
     * @param check Check array bounds?
     */
    bool operator()(int x, int y, int plane, CheckIndices const& check) const;

    //@{
    /**
     *  Write a mask to a regular FITS file.
     *
     *  @param[in] fileName      Name of the file to write.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(std::string const& fileName, daf::base::PropertySet const* metadata = nullptr,
                   std::string const& mode = "w") const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            std::string const& fileName, std::shared_ptr<daf::base::PropertySet const> metadata,
            std::string const& mode = "w") const;
    //@}

    //@{
    /**
     *  Write a mask to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(fits::MemFileManager& manager, daf::base::PropertySet const* metadata = nullptr,
                   std::string const& mode = "w") const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            fits::MemFileManager& manager, std::shared_ptr<daf::base::PropertySet const> metadata,
            std::string const& mode = "w") const;
    //@}

    //@{
    /**
     *  Write a mask to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     */
    void writeFits(fits::Fits& fitsfile, daf::base::PropertySet const* metadata = nullptr) const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet const> metadata) const;
    //@}

    //@{
    /**
     *  Write a mask to a regular FITS file.
     *
     *  @param[in] filename      Name of the file to write.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     *  @param[in] header        Additional values to write to the header (may be null).
     */
    void writeFits(std::string const& filename, fits::ImageWriteOptions const& options,
                   std::string const& mode = "w", daf::base::PropertySet const* header = nullptr) const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            std::string const& filename, fits::ImageWriteOptions const& options, std::string const& mode,
            std::shared_ptr<daf::base::PropertySet const> header) const;
    //@}

    //@{
    /**
     *  Write a mask to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     *  @param[in] header        Additional values to write to the header (may be null).
     */
    void writeFits(fits::MemFileManager& manager, fits::ImageWriteOptions const& options,
                   std::string const& mode = "w", daf::base::PropertySet const* header = nullptr) const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            fits::MemFileManager& manager, fits::ImageWriteOptions const& options, std::string const& mode,
            std::shared_ptr<daf::base::PropertySet const> header) const;
    //@}

    //@{
    /**
     *  Write a mask to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] header        Additional values to write to the header (may be null).
     */
    void writeFits(fits::Fits& fitsfile, fits::ImageWriteOptions const& options,
                   daf::base::PropertySet const* header = nullptr) const;
    [[deprecated("Replaced by a non-shared_ptr overload.  Will be removed after v27.")]] void writeFits(
            fits::Fits& fitsfile, fits::ImageWriteOptions const& options,
            std::shared_ptr<daf::base::PropertySet const> header) const;
    //@}

    /**
     *  Read a Mask from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of afw::fits::DEFAULT_HDU is interpreted as
     *                         "the first HDU with NAXIS != 0".
     */
    static Mask readFits(std::string const& filename, int hdu = fits::DEFAULT_HDU) {
        return Mask<MaskPixelT>(filename, hdu);
    }

    /**
     *  Read a Mask from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of afw::fits::DEFAULT_HDU is interpreted as
     *                          "the first HDU with NAXIS != 0".
     */
    static Mask readFits(fits::MemFileManager& manager, int hdu = fits::DEFAULT_HDU) {
        return Mask<MaskPixelT>(manager, hdu);
    }

    /// Interpret a mask value as a comma-separated list of mask plane names
    std::string interpret(MaskPixelT value);

    std::string getAsString(int x, int y) { return interpret((*this)(x, y)); }

    // Mask Plane ops

    /// Clear all the pixels
    void clearAllMaskPlanes();
    /// Clear the specified bit in all pixels
    void clearMaskPlane(int plane);

    /**
     * Clears default MaskDict planes and docs.
     *
     * TODO: this feels dangerous; once done, it applies everywhere, we can't remake it unless it's been
     * saved.
     * TODO: do we actually want clearCanonical to be an option, or just always do it to keep things
     * consistent?
     * @param clearCanonical Also remove all canonical planes, so that new planes start at 0.
     */
    static void clearDefaultMaskDict(bool clearCanonical = false);

    /// Reset the default MaskDict to the normal initial list, and set the canonical planes to match.
    static void restoreDefaultMaskDict();

    /// Set the default MaskDict to a new MaskDict, and override the canonical planes to match.
    static void setDefaultMaskDict(std::shared_ptr<detail::MaskDict> maskDict);

    /**
     * Set the bit specified by "planeId" for pixels (x0, y) ... (x1, y)
     */
    void setMaskPlaneValues(const int plane, const int x0, const int x1, const int y);

    /**
     * Given a PropertySet that contains the MaskPlane assignments, return a MaskDict containing those planes.
     *
     * @param metadata Metadata from a Mask.
     * @returns MaskDict containing the plane names, ids, and docs read from the metadata.
     */
    static std::shared_ptr<detail::MaskDict> parseMaskPlaneMetadata(
            std::shared_ptr<lsst::daf::base::PropertySet const> metadata);

    // Operations on the mask plane dictionary

    [[deprecated("Doc field will become non-optional. Will be removed after v27.")]] static int addMaskPlane(
            const std::string& name);
    // TODO: can we deprecate these two? Hope so!
    static int addMaskPlane(const std::string& name, const std::string& doc);
    static void removeMaskPlane(const std::string& name);

    /// Add a new named mask plane and doc to this Mask's plane map.
    int addPlane(const std::string& name, const std::string& doc);

    /**
     * @brief Clear all pixels of the specified mask and remove the plane from the mask plane dictionary;
     * optionally remove the plane from the default dictionary too.
     *
     * @param name of maskplane
     * @param removeFromDefault remove from default mask plane dictionary too
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    void removeAndClearMaskPlane(const std::string& name, bool const removeFromDefault = false);

    /**
     * Return the mask plane number corresponding to a plane name.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    [[deprecated("Replaced with non-static `getPlaneId()`. Will be removed after v27.")]] static int
    getMaskPlane(const std::string& name);
    int getPlaneId(std::string name) const;

    /**
     * Return the bitmask corresponding to a vector of plane names OR'd together
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    [[deprecated("Replaced with non-static `getBitMask()`.  Will be removed after v27.")]] static MaskPixelT
    getPlaneBitMask(const std::vector<std::string>& names);
    MaskPixelT getBitMask(const std::vector<std::string>& names) const;

    /**
     * Return the bitmask corresponding to a plane name.
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if plane is invalid
     */
    static MaskPixelT getPlaneBitMask(const std::string& name);
    MaskPixelT getBitMask(std::string name) const;

    // Return the bit mask corresponding to the given plane id.
    static MaskPixelT getBitMaskFromPlaneId(int id) {
        return (id >= 0 && id < getNumPlanesMax()) ? (1 << static_cast<MaskPixelT>(id)) : 0;
    }

    static int getNumPlanesMax() { return 8 * sizeof(MaskPixelT); }
    int getNumPlanesUsed();

    // TODO: I think we can drop these.
    /// Return the Mask's bit plane map.
    // MaskPlaneDict const& getMaskPlaneDict() const { return _maskDict->getMaskPlaneDict(); }
    /// Return the Mask's bit plane map docstrings.
    // MaskPlaneDocDict const& getMaskPlaneDocDict() const { return _maskDict->getMaskPlaneDocDict(); }

    // TODO: not sure we want to keep this?
    std::shared_ptr<detail::MaskDict const> const getMaskDict() const { return _maskDict; }

    /// Print a formatted string showing the mask plane bits, names, and docs.
    std::string printMaskPlanes() const;

    /**
     * Given a PropertySet, replace any existing MaskPlane assignments with the current ones.
     */
    void addMaskPlanesToMetadata(std::shared_ptr<lsst::daf::base::PropertySet>) const;

    // This one isn't static, it fixes up a given Mask's planes.
    /**
     * Adjust this mask to conform to the standard Mask class's mask plane dictionary,
     * adding any new mask planes to the standard.
     *
     * Ensures that this mask (presumably from some external source) has the same plane assignments
     * as the Mask class. If a change in plane assignments is needed, the bits within each pixel
     * are permuted as required.  The provided `masterPlaneDict` describes the true state of the bits
     * in this Mask's pixels and overrides its current MaskDict
     *
     * Any new mask planes found in this mask are added to unused slots in the Mask class's mask plane
     * dictionary.
     *
     * @param currentMaskDict mask plane dictionary currently in use for this mask.
     */
    void conformMaskPlanes(std::shared_ptr<detail::MaskDict> const& currentMaskDict);

private:
    friend class MaskFitsReader;

    std::shared_ptr<detail::MaskDict> _maskDict;  // our bitplane dictionary

    static int _setMaskPlaneDict(MaskPlaneDict const& mpd);
    static const std::string maskPlanePrefix;
    static const std::string maskPlaneDocPrefix;

    /**
     * Initialise mask planes; called by constructors
     */
    // deprecated on DM-32438
    void _initializePlanes(MaskPlaneDict const& planeDefs);  // called by ctors

    /// Initialise mask planes; called by constructors.
    void _initializeMaskDict(std::shared_ptr<detail::MaskDict> maskDict);

    // Make names in templatized base class visible (Meyers, Effective C++, Item 43)
    using ImageBase<MaskPixelT>::_getRawView;
    using ImageBase<MaskPixelT>::swap;

    /**
     * Check that masks have the same dictionary version.
     *
     * @throws lsst::pex::exceptions::RuntimeError
     */
    void checkMaskDictionaries(Mask const& other);
};

template <typename PixelT>
void swap(Mask<PixelT>& a, Mask<PixelT>& b);
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_MASK_H
