// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class Wcs
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_AFW_IMAGE_WCS_H
#define LSST_AFW_IMAGE_WCS_H

#include <vw/Math.h>

#include <lsst/daf/base.h>
#include <lsst/daf/data/LsstBase.h>

struct wcsprm;                          // defined in wcs.h

namespace lsst {
namespace afw {
    namespace formatters {
        class WcsFormatter;
    }
namespace image {

    typedef vw::math::Vector<double, 2> Coord2D;

    /// \brief Wcs supports coordinate system transformations between pixel and world coordinates
    ///
    /// All Wcs (in the FITS sense) coordinate conventions are supported via
    /// Mark Calabretta's wcslib package (http://www.atnf.csiro.au/people/mcalabre)
    ///
    class Wcs : public lsst::daf::base::Persistable,
                public lsst::daf::data::LsstBase {
    public:
        
        Wcs();
        Wcs(lsst::daf::base::DataProperty::PtrType fitsMetaData);
        Wcs(Wcs const &);
        Wcs & operator = (const Wcs &);

        ~Wcs();

        /// Return the input fits header
        lsst::daf::base::DataProperty::PtrType getFitsMetaData() const { 
            return _fitsMetaData; 
        }

        /// Return true iff Wcs is valid
        operator bool() const { return _wcsInfo != NULL; }

        void raDecToColRow(Coord2D sky, Coord2D& pix) const;
        Coord2D raDecToColRow(Coord2D sky) const;
        Coord2D raDecToColRow(double const ra, double const dec) const;

        void colRowToRaDec(Coord2D pix, Coord2D& sky) const;
        Coord2D colRowToRaDec(Coord2D pix) const;
        Coord2D colRowToRaDec(double const col, double const row) const;

        double pixArea(Coord2D pix) const;
    private:
        lsst::daf::base::LSST_PERSIST_FORMATTER(lsst::afw::formatters::WcsFormatter);

        lsst::daf::base::DataProperty::PtrType _fitsMetaData; ///< Input FITS header.  Caveat Emptor: may contain other keywords
        // including e.g. SIMPLE and BITPIX
        struct wcsprm* _wcsInfo;
        int _nWcsInfo;
        int _relax; ///< Degree of permissiveness for wcspih (0 for strict); see wcshdr.h for details.
        int _wcsfixCtrl; ///< Do potentially unsafe translations of non-standard unit strings? 0/1 = no/yes
        int _wcshdrCtrl; ///< Controls messages to stderr from wcshdr (0 for none); see wcshdr.h for details
        int _nReject;
    };
  
}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_WCS_H


