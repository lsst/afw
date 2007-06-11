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

#include "lsst/fw/Exception.h"
#include "lsst/fw/Utils.h"

#include "simpleFits.h"

LSST_START_NAMESPACE(lsst);
LSST_START_NAMESPACE(fw);

#define FITS_SIZE 2880

class Card {
public:
    enum {
	LOGICAL,
	STRING,
	INTEGER,
	DOUBLE,
	FLOAT,
    } type;

    Card(const std::string &str, bool val, const char *commnt = ""
        ) : keyword(str), comment(commnt) {
        type = LOGICAL;
        value.l = val;
    }
    Card(const std::string &str, int val, const char *commnt = ""
        ) : keyword(str), comment(commnt) {
        type = INTEGER;
        value.i = val;
    }
    Card(const std::string &str, const std::string &val, const char *commnt = ""
        ) : keyword(str), comment(commnt) {
        type = STRING;
        value.s = new std::string(val);
    }
    Card(const std::string &str, const char *val, const char *commnt = ""
        ) : keyword(str), comment(commnt) {
        type = STRING;
        value.s = new std::string(val);
    }

    ~Card() {
        if (type == STRING) {
            delete value.s;
        }
    }

    int write(int fd, int ncard, char *record) const;
    
    std::string keyword;
    union {
	bool l;
	std::string *s;
	int i;
	double d;
	float f;      
    } value;
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

    if (type == STRING) {
        if(keyword == "" ||
           keyword == "COMMENT" || keyword == "END" || keyword == "HISTORY") {
            sprintf(card, "%-8.8s%-72s", keyword.c_str(), value.s->c_str());
        } else {
            sprintf(card,"%-8.8s= '%s' %c%-*s",
                    keyword.c_str(), value.s->c_str(),
                    (comment == "" ? ' ' : '/'),
                    (int)(80 - 14 - value.s->size()), comment.c_str());
        }
    } else {
        sprintf(card, "%-8.8s= ", keyword.c_str()); card += 10;
        switch (type) {
          case Card::LOGICAL:
            sprintf(card, "%20s", value.l ? "T" : "F");
            break;
          case Card::STRING:
            break;
          case Card::INTEGER:
            sprintf(card, "%20d", value.i);
            break;
          case Card::DOUBLE:
            sprintf(card, "%20g", value.d);
            break;
          case Card::FLOAT:
            sprintf(card, "%20g", value.f);
        }
        card += 20;
        sprintf(card, " %c%-48s", (comment == "" ? ' ' : '/'), comment.c_str());
    }
/*
 * Write record if full
 */
    if(++ncard == 36) {
	if(posix::write(fd, record, FITS_SIZE) != FITS_SIZE) {
	    throw Exception("Cannot write header record");
	}
	ncard = 0;
    }
   
    return ncard;
}   

/*****************************************************************************/
/*
 * Byte swap ABABAB -> BABABAB in place
 */
static void swap_2(char *arr,		// array to swap
		   int n)		// number of bytes
{
    char *end,
	t;

    if(n%2 != 0) {
	throw
            Exception(boost::format("Attempt to byte swap odd number of bytes: %d") % n);
    }

    for(end = arr + n;arr < end;arr += 2) {
	t = arr[0];
	arr[0] = arr[1];
	arr[1] = t;
    }
}

/*
 * Byte swap ABCDABCD -> DCBADCBA in place (e.g. sun <--> vax)
 */
static void swap_4(char *arr,		// array to swap
		   int n)		// number of bytes
{
    char *end,
	t;

    if(n%4 != 0) {
	throw Exception(boost::format("Attempt to byte swap non-multiple of 4 bytes: %d") % n);
	n = 4*(int)(n/4);
    }

    for(end = arr + n;arr < end;arr += 4) {
	t = arr[0];
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
static void swap_8(char *arr,		// array to swap
		   int n)		// number of bytes
{
    char *end,
	t;

    if(n%8 != 0) {
	throw Exception(boost::format("Attempt to byte swap non-multiple of 8 bytes: %d") % n);
	n = 8*(int)(n/8);
    }

    for(end = arr + n;arr < end;arr += 8) {
	t = arr[0];
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

static int
write_fits_hdr(int fd,
	       int bitpix,
	       int naxis,
	       int *naxes,
	       Card **cards,		/* extra header cards */
	       int primary)		/* is this the primary HDU? */
{
    int i;
    int ncard;				/* number of cards written */
    char record[FITS_SIZE + 1];		/* write buffer */
   
    ncard = 0;
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
    if(cards != NULL) {
	while(*cards != NULL) {
	    ncard = (*cards)->write(fd,ncard,record);
	    cards++;
	}
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
static void pad_to_fits_record(int fd,		// output file descriptor
			       int npixel,	// number of pixels already written to HDU
			       int bitpix)	// bitpix for this datatype
{
    const int bytes_per_pixel = (bitpix > 0 ? bitpix : -bitpix)/8;
    int nbyte = npixel*bytes_per_pixel;
    
    if(nbyte%FITS_SIZE != 0) {
	char record[FITS_SIZE + 1];	/* write buffer */
	
	nbyte = FITS_SIZE - nbyte%FITS_SIZE;
	memset(record, ' ', nbyte);
	if(write(fd, record, nbyte) != nbyte) {
	    throw Exception("error padding file to multiple of fits block size");
	}
    }
}

static int write_fits_data(int fd,
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

void writeVwFits(int fd,                // file descriptor to write to
                 const vw::ImageBuffer& buff, // The data to write
                 const std::string& WCS // which WCS to use for pixel
                ) {
    Card **cards = NULL;		/* extra header info */
    const int naxis = 2;		// == NAXIS
    int naxes[2];			/* values of NAXIS1 etc */
    int ncard = 12;			/* number of extra header cards */
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
        throw Exception(boost::format("Unsupported channel type: %d") %
                        buff.format.channel_type);
    }
    char *data = static_cast<char *>(buff.data);
    
    /*
     * Allocate cards for FITS headers
     */
    if (WCS != "") {
       cards = new Card *[ncard + 1];
       cards[ncard] = NULL;
       /*
	* Generate cards for WCS, so that pixel (0,0) is correctly labelled
	*/
       int i = 0;
       cards[i++] = new Card(str(boost::format("CRVAL1%s") % WCS), 0,
                             "(output) Column pixel of Reference Pixel");
       cards[i++] = new Card(str(boost::format("CRVAL2%s") % WCS), 0,
                             "(output) Row pixel of Reference Pixel");
       cards[i++] = new Card(str(boost::format("CRPIX1%s") % WCS), 1,
                             "Column Pixel Coordinate of Reference");
       cards[i++] = new Card(str(boost::format("CRPIX2%s") % WCS), 1,
                             "Row Pixel Coordinate of Reference");
       cards[i++] = new Card(str(boost::format("CTYPE1%s") % WCS),
                             "LINEAR", "Type of projection");
       cards[i++] = new Card(str(boost::format("CTYPE1%s") % WCS),
                             "LINEAR", "Type of projection");
       cards[i++] = new Card(str(boost::format("CUNIT1%s") % WCS),
                             "PIXEL", "Column unit");
       cards[i++] = new Card(str(boost::format("CUNIT2%s") % WCS),
                             "PIXEL", "Row unit");
       cards[i] = NULL;
       assert(i <= ncard);
    }

    naxes[0] = buff.cols();
    naxes[1] = buff.rows();
    
    write_fits_hdr(fd, bitpix, naxis, naxes, cards, 1);
    for (unsigned int r = 0; r < buff.rows(); r++) {
	if(write_fits_data(fd, bitpix, buff.cols(), data + r*buff.rstride) < 0){
	    throw Exception(boost::format("Error writing data for row %d")
                                  % r);
	}
    }

    pad_to_fits_record(fd, buff.cols()*buff.rows(), bitpix);

    if (cards != NULL) {
       delete cards[0];
       delete cards;
    }
}   

/******************************************************************************/

void writeVwFits(const std::string &filename, // file to write or "| cmd"
                 const vw::ImageBuffer &data, // The data to write
                 const std::string &WCS // which WCS to use for pixel
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
        throw Exception(boost::format("Cannot open \"%s\"") % filename);
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
