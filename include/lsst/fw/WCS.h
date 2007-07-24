// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class WCS
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_WCS_H
#define LSST_WCS_H

#include <vw/Math.h>
#include "lsst/fw/LsstBase.h"
#include "lsst/fw/DataProperty.h"

struct wcsprm;                          // defined in wcs.h

namespace lsst {

    namespace fw {
        
        typedef vw::math::Vector<double, 2> Coord2D;

        /// \brief WCS supports coordinate system transformations between pixel and world coordinates
        ///
        /// All WCS (in the FITS sense) coordinate conventions are supported via
        /// Mark Calabretta's wcslib package (http://www.atnf.csiro.au/people/mcalabre)
        ///
        class WCS : private LsstBase {
        public:
            
            WCS();
            WCS(DataPropertyPtrT fitsMetaData);
            ~WCS();

            /// Return the input fits header
            DataPropertyPtrT getFitsMetaData() const { return _fitsMetaData; }

            /// Return true iff WCS is valid
            operator bool() const { return _wcsInfo != NULL; }

            void raDecToColRow(Coord2D sky, Coord2D& pix) const;
            Coord2D raDecToColRow(Coord2D sky) const;
            Coord2D raDecToColRow(double const ra, double const dec) const;

            void colRowToRaDec(Coord2D pix, Coord2D& sky) const;
            Coord2D colRowToRaDec(Coord2D pix) const;
            Coord2D WCS::colRowToRaDec(double const col, double const row) const;
        private:
            DataPropertyPtrT _fitsMetaData; ///< Input FITS header.  Caveat Emptor: may contain other keywords
            // including e.g. SIMPLE and BITPIX
            struct wcsprm* _wcsInfo;
            int _nWcsInfo;
            int _status;
            int _relax;  // should be set by policy
            int _ctrl;
            int _nReject;
        };
  
    } // namespace fw

} // namespace lsst

#endif // LSST_WCS_H


