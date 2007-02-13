// -*- lsst-c++ -*-

#ifndef LSST_PIXELREGION_H
#define LSST_PIXELREGION_H

// PixelRegion is defined to be consistent with VW style for specifying a rectangular
// region of pixels

struct PixelRegion {
    int upperLeftX;
    int upperLeftY;
    int width;
    int height;
};

struct PixelCoord {
    int x;
    int y;
};

#endif // LSST_PIXELREGION_H
