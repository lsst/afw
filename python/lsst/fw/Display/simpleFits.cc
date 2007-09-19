/*
 * Write a fits image to a file descriptor; useful for talking to DS9
 *
 * This version knows about LSST data structures
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
namespace posix {
#   include <unistd.h>
#   include <fcntl.h>
}
using namespace posix;

#include "lsst/mwi/exceptions.h"
#include "lsst/mwi/utils/Utils.h"
#include <boost/any.hpp>

#include "simpleFits.h"

LSST_START_NAMESPACE(lsst);
LSST_START_NAMESPACE(fw);

using lsst::mwi::data::DataProperty;

#define FITS_SIZE 2880

class Card {
public:
    Card(const std::string &name, bool val, const char *commnt = ""
        ) : keyword(name), value(val), comment(commnt) { }
    Card(const std::string &name, int val, const char *commnt = ""
        ) : keyword(name), value(val), comment(commnt) { }
    Card(const std::string &name, const std::string &val, const char *commnt = ""
        ) : keyword(name), value(val), comment(commnt) { }
    Card(const std::string &name, const char *val, const char *commnt = ""
        ) : keyword(name), value(std::string(val)), comment(commnt) { }

    Card(const DataProperty& dp, const char *commnt = ""
        ) : keyword(dp.getName()), value(dp.getValue()), comment(commnt) { }
    

    ~Card() {}

    int write(int fd, int ncard, char *record) const;
    
    std::string keyword;
    boost::any value;
    std::string comment;
};

/*****************************************************************************/
/*
 * Write a Card
 */
int Card::write(int fd,
                int ncard,
                char *record
               ) const {
    char *card = &record[80*ncard];

    if (value.type() == typeid(std::string)) {
        const char *str = boost::any_cast<std::string>(value).c_str();
        if(keyword == "" ||
           keyword == "COMMENT" || keyword == "END" || keyword == "HISTORY") {
            sprintf(card, "%-8.8s%-72s", keyword.c_str(), str);
        } else {
            sprintf(card,"%-8.8s= '%s' %c%-*s",
                    keyword.c_str(), str,
                    (comment == "" ? ' ' : '/'),
                    (int)(80 - 14 - strlen(str)), comment.c_str());
        }
    } else {
        sprintf(card, "%-8.8s= ", keyword.c_str()); card += 10;
        if (value.type() == typeid(bool)) {
            sprintf(card, "%20s", boost::any_cast<bool>(value) ? "T" : "F");
        } else if (value.type() == typeid(int)) {
            sprintf(card, "%20d", boost::any_cast<int>(value));
        } else if (value.type() == typeid(double)) {
            sprintf(card, "%20g", boost::any_cast<double>(value));
        } else if (value.type() == typeid(float)) {
            sprintf(card, "%20g", boost::any_cast<float>(value));
        }
        card += 20;
        sprintf(card, " %c%-48s", (comment == "" ? ' ' : '/'), comment.c_str());
    }
/*
 * Write record if full
 */
    if(++ncard == 36) {
	if(posix::write(fd, record, FITS_SIZE) != FITS_SIZE) {
	    throw lsst::mwi::exceptions::Runtime("Cannot write header record");
	}
	ncard = 0;
    }
   
    return ncard;
}   

/*****************************************************************************/
/*
 * Byte swap ABABAB -> BABABAB in place
 */
namespace {
    void swap_2(char *arr,              // array to swap
                const int n) {          // number of bytes
        if(n%2 != 0) {
            throw
                lsst::mwi::exceptions::Runtime(boost::format("Attempt to byte swap odd number of bytes: %d") % n);
        }

        for(char *end = arr + n;arr < end;arr += 2) {
            char t = arr[0];
            arr[0] = arr[1];
            arr[1] = t;
        }
    }
    /*
     * Byte swap ABCDABCD -> DCBADCBA in place (e.g. sun <--> vax)
     */
    void swap_4(char *arr,              // array to swap
                const int n) {          // number of bytes
        if(n%4 != 0) {
            throw lsst::mwi::exceptions::Runtime(boost::format("Attempt to byte swap non-multiple of 4 bytes: %d") % n);
        }

        for(char *end = arr + n;arr < end;arr += 4) {
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
    void swap_8(char *arr,              // array to swap
                const int n) {          // number of bytes
        if(n%8 != 0) {
            throw lsst::mwi::exceptions::Runtime(boost::format("Attempt to byte swap non-multiple of 8 bytes: %d") % n);
        }

        for(char *end = arr + n;arr < end;arr += 8) {
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

/*****************************************************************************/

    int write_fits_hdr(int fd,
                       int bitpix,
                       int naxis,
                       int *naxes,
                       std::list<Card>& cards,  /* extra header cards */
                       int primary)		/* is this the primary HDU? */
    {
        int i;
        char record[FITS_SIZE + 1];		/* write buffer */
   
        int ncard = 0;
        if(primary) {
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
        for(i = 0; i < naxis; i++) {
            char key[] = "NAXIS.";
            sprintf(key, "NAXIS%d", i + 1);
            Card card(key, naxes[i]);
            ncard = card.write(fd, ncard, record);
        }
        if(primary) {
            Card card("EXTEND", true, "There may be extensions");
            ncard = card.write(fd,ncard,record);
        }
/*
 * Write extra header cards
 */
        for (std::list<Card>::const_iterator card = cards.begin(); card != cards.end(); card++) {
            ncard = card->write(fd,ncard,record);
        }

        {
            Card card("END", "");
            ncard = card.write(fd,ncard,record);
        }
        while(ncard != 0) {
            Card card("", "");
            ncard = card.write(fd,ncard,record);
        }

        return 0;
    }

/*
 * Pad out to a FITS record boundary
 */
    void pad_to_fits_record(int fd,		// output file descriptor
                            int npixel,	// number of pixels already written to HDU
                            int bitpix	// bitpix for this datatype
                           ) {
        const int bytes_per_pixel = (bitpix > 0 ? bitpix : -bitpix)/8;
        int nbyte = npixel*bytes_per_pixel;
    
        if(nbyte%FITS_SIZE != 0) {
            char record[FITS_SIZE + 1];	/* write buffer */
	
            nbyte = FITS_SIZE - nbyte%FITS_SIZE;
            memset(record, ' ', nbyte);
            if(write(fd, record, nbyte) != nbyte) {
                throw lsst::mwi::exceptions::Runtime("error padding file to multiple of fits block size");
            }
        }
    }

    int write_fits_data(int fd,
                        int bitpix,
                        int npix,
                        char *data) {
        char *buff = NULL;			/* I/O buffer */
        const int bytes_per_pixel = (bitpix > 0 ? bitpix : -bitpix)/8;
        int nbyte = npix*bytes_per_pixel;
        int swap_bytes = 0;			/* the default */
        static int warned = 0;		/* Did we warn about BZERO/BSCALE? */
   
#if defined(LSST_LITTLE_ENDIAN)		/* we'll need to byte swap FITS */
        if(bytes_per_pixel > 1) {
            swap_bytes = 1;
        }
#endif

        if(swap_bytes) {
            buff = new char[FITS_SIZE*bytes_per_pixel];
        }
    
        if(bytes_per_pixel == 2 && !warned) {
            warned = 1;
            fprintf(stderr,"Worry about BZERO/BSCALE\n");
        }
   
        while(nbyte > 0) {
            int nwrite = (nbyte > FITS_SIZE) ? FITS_SIZE : nbyte;
      
            if(swap_bytes) {
                memcpy(buff, data, nwrite);
                if(bytes_per_pixel == 2) {
                    swap_2((char *)buff, nwrite);
                } else if(bytes_per_pixel == 4) {
                    swap_4((char *)buff, nwrite);
                } else if(bytes_per_pixel == 8) {
                    swap_8((char *)buff, nwrite);
                } else {
                    fprintf(stderr,"You cannot get here\n");
                    abort();
                }
            } else {
                buff = data;
            }
      
            if(write(fd, buff, nwrite) != nwrite) {
                perror("Error writing image: ");
                break;
            }

            nbyte -= nwrite;
            data += nwrite;
        }
   
        if(swap_bytes) {
            delete buff;
        }

        return (nbyte == 0 ? 0 : -1);
    }
}

void writeVwFits(int fd,                // file descriptor to write to
                 const vw::ImageBuffer& buff, // The data to write
                 const WCS *WCS         // which WCS to use for pixel
                ) {
    /*
     * What sort if image is it?
     */
    int bitpix = 0;                     // BITPIX for fits file
    switch (buff.format.channel_type) {
      case vw::VW_CHANNEL_UINT8:
      case vw::VW_CHANNEL_INT8:
	bitpix = 8;
        break;
      case vw::VW_CHANNEL_UINT16:
	bitpix = -16;
	break;
      case vw::VW_CHANNEL_INT16:
	bitpix = 16;
        break;
      case vw::VW_CHANNEL_INT32:
	bitpix = 32;
	break;
      case vw::VW_CHANNEL_FLOAT32:
	bitpix = -32;
	break;
      default:
        throw lsst::mwi::exceptions::Runtime(boost::format("Unsupported channel type: %d") %
                        buff.format.channel_type);
    }
    char *data = static_cast<char *>(buff.data);
    
    /*
     * Allocate cards for FITS headers
     */
    std::list<Card> cards;
    /*
     * Generate cards for WCS, so that pixel (0,0) is correctly labelled
     */
    std::string WCSname = "A";
    cards.push_back(Card(str(boost::format("CRVAL1%s") % WCSname), 0, "(output) Column pixel of Reference Pixel"));
    cards.push_back(Card(str(boost::format("CRVAL2%s") %WCSname), 0, "(output) Row pixel of Reference Pixel"));
    cards.push_back(Card(str(boost::format("CRPIX1%s") %WCSname), 1, "Column Pixel Coordinate of Reference"));
    cards.push_back(Card(str(boost::format("CRPIX2%s") %WCSname), 1, "Row Pixel Coordinate of Reference"));
    cards.push_back(Card(str(boost::format("CTYPE1%s") %WCSname), "LINEAR", "Type of projection"));
    cards.push_back(Card(str(boost::format("CTYPE1%s") %WCSname), "LINEAR", "Type of projection"));
    cards.push_back(Card(str(boost::format("CUNIT1%s") %WCSname), "PIXEL", "Column unit"));
    cards.push_back(Card(str(boost::format("CUNIT2%s") %WCSname), "PIXEL", "Row unit"));
    /*
     * Was there something else?
     */
    if (WCS != NULL) {
        DataProperty::iteratorRangeType WCSCards = WCS->getFitsMetaData()->getChildren();
        
        for (DataProperty::ContainerIteratorType i = WCSCards.first; i != WCSCards.second; i++) {
            Card card(*(*i));
            
            if (card.keyword == "SIMPLE" ||
                card.keyword == "BITPIX" ||
                card.keyword == "NAXIS" ||
                card.keyword == "NAXIS1" ||
                card.keyword == "NAXIS2" ||
                card.keyword == "XTENSION" ||
                card.keyword == "PCOUNT" ||
                card.keyword == "GCOUNT"
               ) {
                continue;
            }

            cards.push_back(card);
        }
    }
    /*
     * Basic FITS stuff
     */
    const int naxis = 2;		// == NAXIS
    int naxes[naxis];			/* values of NAXIS1 etc */
    naxes[0] = buff.cols();
    naxes[1] = buff.rows();
    
    write_fits_hdr(fd, bitpix, naxis, naxes, cards, 1);
    for (unsigned int r = 0; r < buff.rows(); r++) {
	if(write_fits_data(fd, bitpix, buff.cols(), data + r*buff.rstride) < 0){
	    throw lsst::mwi::exceptions::Runtime(boost::format("Error writing data for row %d") % r);
	}
    }

    pad_to_fits_record(fd, buff.cols()*buff.rows(), bitpix);
}   

/******************************************************************************/

void writeVwFits(const std::string &filename, // file to write or "| cmd"
                 const vw::ImageBuffer &data, // The data to write
                 const WCS *WCS         // which WCS to use for pixel
                ) {
    int fd;
    if ((filename.c_str())[0] == '|') {		// a command
	const char *cmd = filename.c_str() + 1;
	while (isspace(*cmd)) {
	    cmd++;
	}

	fd = fileno(popen(cmd, "w"));
    } else {
	fd = creat(filename.c_str(), 777);
    }

    if (fd < 0) {
        throw lsst::mwi::exceptions::Runtime(boost::format("Cannot open \"%s\"") % filename);
    }

    try {
        writeVwFits(fd, data, WCS);
    } catch(Exception &e) {
        (void)close(fd);
        throw e;
    }

    (void)close(fd);
}

LSST_END_NAMESPACE(fw);
LSST_END_NAMESPACE(lsst);
