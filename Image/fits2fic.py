#!/usr/bin/env python

import os
import sys

import pyfits

def usage():
    print 'Usage: %s <mef_fits_file>' % os.path.basename(sys.argv[0])
    sys.exit(2)

def main():
    if len(sys.argv) != 2:
        usage()
    fitsFile = sys.argv[1]
    hdus = pyfits.open(fitsFile)

    # If MEF file (which is normal), skip the primary HDU.
    if len(hdus) > 1:
        hdus.pop(0)

    # Make the containing FIC directory.
    if fitsFile[-5:] == '.fits':
        ficDirectory = fitsFile[:-5] + '.fic'
    else:
        ficDirectory = fitsFile + '.fic'
    try: os.mkdir(ficDirectory)
    except: pass

    # Write each image HDU to its own target FITS file.
    numImages = len(hdus)
    for i in range(numImages):
        hdu = hdus[i]
        phdu = pyfits.PrimaryHDU(hdu.data, hdu.header)
        targetFile = os.path.join(ficDirectory, 'target-%d.fits' % (i + 1))
        phdu.writeto(targetFile, clobber=True)
        print 'Wrote %s (%d of %d)' % (targetFile, i + 1, numImages)

    # Close the input file.
    hdus.close()

    # Create the properties for this FIC.
    propertiesFile = os.path.join(ficDirectory, 'properties')
    properties = open(propertiesFile, 'w')
    properties.write('target_count = %d\n' % numImages)
    properties.close()
    print 'Wrote %s' % propertiesFile

if __name__ == '__main__':
    main()
