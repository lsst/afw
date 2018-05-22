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
 * Write a FITS image to a file descriptor; useful for talking to DS9
 *
 * This version knows about LSST data structures
 */
#ifndef DOXYGEN    // Doxygen doesn't like includes inside namespaces
namespace posix {  // here so no-one includes them first outside namespace posix {}
#include <unistd.h>
#include <fcntl.h>
}
#endif  // !DOXYGEN
using namespace posix;
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cctype>

#include "lsst/pex/exceptions.h"
#include "boost/any.hpp"
#include "lsst/afw/fits.h"

#include "simpleFits.h"

namespace image = lsst::afw::image;
using lsst::daf::base::PropertySet;

#define FITS_SIZE 2880

/// @cond
class Card {
public:
    Card(const std::string &name, bool val, const char *commnt = "")
            : keyword(name), value(val), comment(commnt) {}
    Card(const std::string &name, int val, const char *commnt = "")
            : keyword(name), value(val), comment(commnt) {}
    Card(const std::string &name, double val, const char *commnt = "")
            : keyword(name), value(val), comment(commnt) {}
    Card(const std::string &name, float val, const char *commnt = "")
            : keyword(name), value(val), comment(commnt) {}
    Card(const std::string &name, const std::string &val, const char *commnt = "")
            : keyword(name), value(val), comment(commnt) {}
    Card(const std::string &name, const char *val, const char *commnt = "")
            : keyword(name), value(std::string(val)), comment(commnt) {}

    ~Card() {}

    int write(int fd, int ncard, char *record) const;

    std::string keyword;
    boost::any value;
    std::string comment;
};

/*
 * Write a Card
 */
int Card::write(int fd, int ncard, char *record) const {
    char *card = &record[80 * ncard];

    if (value.type() == typeid(std::string)) {
        const char *str = boost::any_cast<std::string>(value).c_str();
        if (keyword == "" || keyword == "COMMENT" || keyword == "END" || keyword == "HISTORY") {
            sprintf(card, "%-8.8s%-72s", keyword.c_str(), str);
        } else {
            sprintf(card, "%-8.8s= '%s' %c%-*s", keyword.c_str(), str, (comment == "" ? ' ' : '/'),
                    (int)(80 - 14 - strlen(str)), comment.c_str());
        }
    } else {
        sprintf(card, "%-8.8s= ", keyword.c_str());
        card += 10;
        if (value.type() == typeid(bool)) {
            sprintf(card, "%20s", boost::any_cast<bool>(value) ? "T" : "F");
        } else if (value.type() == typeid(int)) {
            sprintf(card, "%20d", boost::any_cast<int>(value));
        } else if (value.type() == typeid(double)) {
            sprintf(card, "%20.10f", boost::any_cast<double>(value));
        } else if (value.type() == typeid(float)) {
            sprintf(card, "%20.7f", boost::any_cast<float>(value));
        }
        card += 20;
        sprintf(card, " %c%-48s", (comment == "" ? ' ' : '/'), comment.c_str());
    }
    /*
     * Write record if full
     */
    if (++ncard == 36) {
        if (posix::write(fd, record, FITS_SIZE) != FITS_SIZE) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Cannot write header record");
        }
        ncard = 0;
    }

    return ncard;
}
/// @endcond

/*
 * Utilities
 *
 * Flip high-order bit so as to write unsigned short to FITS.  Grrr.
 */
namespace {
void flip_high_bit(char *arr,      // array that needs bits swapped
                   const int n) {  // number of bytes in arr
    if (n % 2 != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Attempt to bit flip odd number of bytes: %d") % n).str());
    }

    unsigned short *uarr = reinterpret_cast<unsigned short *>(arr);
    for (unsigned short *end = uarr + n / 2; uarr < end; ++uarr) {
        *uarr ^= 0x8000;
    }
}
}

/*
 * Byte swap ABABAB -> BABABAB in place
 */
namespace {
void swap_2(char *arr,      // array to swap
            const int n) {  // number of bytes
    if (n % 2 != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Attempt to byte swap odd number of bytes: %d") % n).str());
    }

    for (char *end = arr + n; arr < end; arr += 2) {
        char t = arr[0];
        arr[0] = arr[1];
        arr[1] = t;
    }
}
/*
 * Byte swap ABCDABCD -> DCBADCBA in place (e.g. sun <--> vax)
 */
void swap_4(char *arr,      // array to swap
            const int n) {  // number of bytes
    if (n % 4 != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Attempt to byte swap non-multiple of 4 bytes: %d") % n).str());
    }

    for (char *end = arr + n; arr < end; arr += 4) {
        char t = arr[0];
        arr[0] = arr[3];
        arr[3] = t;
        t = arr[1];
        arr[1] = arr[2];
        arr[2] = t;
    }
}

/*
 * Byte swap ABCDEFGH -> HGFEDCBA in place (e.g. sun <--> vax)
 */
void swap_8(char *arr,      // array to swap
            const int n) {  // number of bytes
    if (n % 8 != 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Attempt to byte swap non-multiple of 8 bytes: %d") % n).str());
    }

    for (char *end = arr + n; arr < end; arr += 8) {
        char t = arr[0];
        arr[0] = arr[7];
        arr[7] = t;
        t = arr[1];
        arr[1] = arr[6];
        arr[6] = t;
        t = arr[2];
        arr[2] = arr[5];
        arr[5] = t;
        t = arr[3];
        arr[3] = arr[4];
        arr[4] = t;
    }
}

int write_fits_hdr(int fd, int bitpix, int naxis, int *naxes, std::list<Card> &cards, /* extra header cards */
                   int primary) /* is this the primary HDU? */
{
    int i;
    char record[FITS_SIZE + 1]; /* write buffer */

    int ncard = 0;
    if (primary) {
        Card card("SIMPLE", true);
        ncard = card.write(fd, ncard, record);
    } else {
        Card card("XTENSION", "IMAGE");
        ncard = card.write(fd, ncard, record);
    }

    {
        Card card("BITPIX", bitpix);
        ncard = card.write(fd, ncard, record);
    }
    {
        Card card("NAXIS", naxis);
        ncard = card.write(fd, ncard, record);
    }
    for (i = 0; i < naxis; i++) {
        char key[] = "NAXIS.";
        sprintf(key, "NAXIS%d", i + 1);
        Card card(key, naxes[i]);
        ncard = card.write(fd, ncard, record);
    }
    if (primary) {
        Card card("EXTEND", true, "There may be extensions");
        ncard = card.write(fd, ncard, record);
    }
    /*
     * Write extra header cards
     */
    for (std::list<Card>::const_iterator card = cards.begin(); card != cards.end(); card++) {
        ncard = card->write(fd, ncard, record);
    }

    {
        Card card("END", "");
        ncard = card.write(fd, ncard, record);
    }
    while (ncard != 0) {
        Card card("", "");
        ncard = card.write(fd, ncard, record);
    }

    return 0;
}

/*
 * Pad out to a FITS record boundary
 */
void pad_to_fits_record(int fd,      // output file descriptor
                        int npixel,  // number of pixels already written to HDU
                        int bitpix   // bitpix for this datatype
                        ) {
    const int bytes_per_pixel = (bitpix > 0 ? bitpix : -bitpix) / 8;
    int nbyte = npixel * bytes_per_pixel;

    if (nbyte % FITS_SIZE != 0) {
        char record[FITS_SIZE + 1]; /* write buffer */

        nbyte = FITS_SIZE - nbyte % FITS_SIZE;
        memset(record, ' ', nbyte);
        if (write(fd, record, nbyte) != nbyte) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              "error padding file to multiple of fits block size");
        }
    }
}

int write_fits_data(int fd, int bitpix, char *begin, char *end) {
    const int bytes_per_pixel = (bitpix > 0 ? bitpix : -bitpix) / 8;
    int swap_bytes = 0;          // the default
#if defined(LSST_LITTLE_ENDIAN)  // we'll need to byte swap FITS
    if (bytes_per_pixel > 1) {
        swap_bytes = 1;
    }
#endif

    char *buff = NULL;       // I/O buffer
    bool allocated = false;  // do I need to free it?
    if (swap_bytes || bitpix == 16) {
        buff = new char[FITS_SIZE * bytes_per_pixel];
        allocated = true;
    }

    int nbyte = end - begin;
    int nwrite = (nbyte > FITS_SIZE) ? FITS_SIZE : nbyte;
    for (char *ptr = begin; ptr != end; nbyte -= nwrite, ptr += nwrite) {
        if (end - ptr < nwrite) {
            nwrite = end - ptr;
        }

        if (swap_bytes) {
            memcpy(buff, ptr, nwrite);
            if (bitpix == 16) {  // flip high-order bit
                flip_high_bit(buff, nwrite);
            }

            if (bytes_per_pixel == 2) {
                swap_2(buff, nwrite);
            } else if (bytes_per_pixel == 4) {
                swap_4(buff, nwrite);
            } else if (bytes_per_pixel == 8) {
                swap_8(buff, nwrite);
            } else {
                fprintf(stderr, "You cannot get here\n");
                abort();
            }
        } else {
            if (bitpix == 16) {  // flip high-order bit
                memcpy(buff, ptr, nwrite);
                flip_high_bit(buff, nwrite);
            } else {
                buff = ptr;
            }
        }

        if (write(fd, buff, nwrite) != nwrite) {
            perror("Error writing image: ");
            break;
        }
    }

    if (allocated) {
        delete buff;
    }


    return (nbyte == 0 ? 0 : -1);
}

void
addWcs(std::string const & wcsName, std::list<Card> & cards, int x0=0.0, int y0=0.0)
{
    cards.push_back(Card(str(boost::format("CRVAL1%s") % wcsName),
                         x0, "(output) Column pixel of Reference Pixel"));
    cards.push_back(Card(str(boost::format("CRVAL2%s") % wcsName),
                         y0, "(output) Row pixel of Reference Pixel"));
    cards.push_back(Card(str(boost::format("CRPIX1%s") % wcsName), 1.0,
                         "Column Pixel Coordinate of Reference"));
    cards.push_back(Card(str(boost::format("CRPIX2%s") % wcsName), 1.0,
                         "Row Pixel Coordinate of Reference"));
    cards.push_back(Card(str(boost::format("CTYPE1%s") % wcsName), "LINEAR", "Type of projection"));
    cards.push_back(Card(str(boost::format("CTYPE1%s") % wcsName), "LINEAR", "Type of projection"));
    cards.push_back(Card(str(boost::format("CUNIT1%s") % wcsName), "PIXEL", "Column unit"));
    cards.push_back(Card(str(boost::format("CUNIT2%s") % wcsName), "PIXEL", "Row unit"));
}
}

namespace lsst {
namespace afw {
namespace display {

template <typename ImageT>
void writeBasicFits(int fd,                 // file descriptor to write to
                    ImageT const &data,     // The data to write
                    geom::SkyWcs const *Wcs,  // which Wcs to use for pixel
                    char const *title       // title to write to DS9
                    ) {
    /*
     * Allocate cards for FITS headers
     */
    std::list<Card> cards;
    /*
     * What sort if image is it?
     */
    int bitpix = lsst::afw::fits::getBitPix<typename ImageT::Pixel>();
    if (bitpix == 20) {  // cfitsio for "Unsigned short"
        cards.push_back(Card("BZERO", 32768.0, ""));
        cards.push_back(Card("BSCALE", 1.0, ""));
        bitpix = 16;
    } else if (bitpix == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unsupported image type");
    }
    /*
     * Generate WcsA, pixel coordinates, allowing for X0 and Y0
     */
    addWcs("A", cards, data.getX0(), data.getY0());
    /*
     * Now WcsB, so that pixel (0,0) is correctly labelled (but ignoring XY0)
     */
    addWcs("B", cards);

    if (title) {
        cards.push_back(Card("OBJECT", title, "Image being displayed"));
    }
    /*
     * Was there something else?
     */
    if (Wcs == NULL) {
        addWcs("", cards);              // works around a ds9 bug that WCSA/B is ignored if no Wcs is present
    } else {
        typedef std::vector<std::string> NameList;

        auto shift = lsst::geom::Extent2D(-data.getX0(), -data.getY0());
        auto newWcs = Wcs->copyAtShiftedPixelOrigin(shift);

        std::shared_ptr<lsst::daf::base::PropertySet> metadata = newWcs->getFitsMetadata();

        NameList paramNames = metadata->paramNames();

        for (NameList::const_iterator i = paramNames.begin(), end = paramNames.end(); i != end; ++i) {
            if (*i == "SIMPLE" || *i == "BITPIX" || *i == "NAXIS" || *i == "NAXIS1" || *i == "NAXIS2" ||
                *i == "XTENSION" || *i == "PCOUNT" || *i == "GCOUNT") {
                continue;
            }
            std::type_info const &type = metadata->typeOf(*i);
            if (type == typeid(bool)) {
                cards.push_back(Card(*i, metadata->get<bool>(*i)));
            } else if (type == typeid(int)) {
                cards.push_back(Card(*i, metadata->get<int>(*i)));
            } else if (type == typeid(float)) {
                cards.push_back(Card(*i, metadata->get<float>(*i)));
            } else if (type == typeid(double)) {
                cards.push_back(Card(*i, metadata->get<double>(*i)));
            } else {
                cards.push_back(Card(*i, metadata->get<std::string>(*i)));
            }
        }
    }
    /*
     * Basic FITS stuff
     */
    const int naxis = 2;  // == NAXIS
    int naxes[naxis];     /* values of NAXIS1 etc */
    naxes[0] = data.getWidth();
    naxes[1] = data.getHeight();

    write_fits_hdr(fd, bitpix, naxis, naxes, cards, 1);
    for (int y = 0; y != data.getHeight(); ++y) {
        if (write_fits_data(fd, bitpix, (char *)(data.row_begin(y)), (char *)(data.row_end(y))) < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              (boost::format("Error writing data for row %d") % y).str());
        }
    }

    pad_to_fits_record(fd, data.getWidth() * data.getHeight(), bitpix);
}

template <typename ImageT>
void writeBasicFits(std::string const &filename,  // file to write, or "| cmd"
                    ImageT const &data,           // The data to write
                    geom::SkyWcs const *Wcs,        // which Wcs to use for pixel
                    char const *title             // title to write to DS9
                    ) {
    int fd;
    if ((filename.c_str())[0] == '|') {  // a command
        const char *cmd = filename.c_str() + 1;
        while (isspace(*cmd)) {
            cmd++;
        }

        fd = fileno(popen(cmd, "w"));
    } else {
        fd = creat(filename.c_str(), 777);
    }

    if (fd < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Cannot open \"%s\"") % filename).str());
    }

    try {
        writeBasicFits(fd, data, Wcs, title);
    } catch (lsst::pex::exceptions::Exception &) {
        (void)close(fd);
        throw;
    }

    (void)close(fd);
}

/// @cond
#define INSTANTIATE(IMAGET)                                                              \
    template void writeBasicFits(int, IMAGET const &, geom::SkyWcs const *, char const *); \
    template void writeBasicFits(std::string const &, IMAGET const &, geom::SkyWcs const *, char const *)

#define INSTANTIATE_IMAGE(T) INSTANTIATE(lsst::afw::image::Image<T>)
#define INSTANTIATE_MASK(T) INSTANTIATE(lsst::afw::image::Mask<T>)

INSTANTIATE_IMAGE(std::uint16_t);
INSTANTIATE_IMAGE(int);
INSTANTIATE_IMAGE(float);
INSTANTIATE_IMAGE(double);
INSTANTIATE_IMAGE(std::uint64_t);

INSTANTIATE_MASK(std::uint16_t);
INSTANTIATE_MASK(image::MaskPixel);
/// @endcond
}
}
}
