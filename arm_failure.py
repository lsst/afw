import numpy as np
import lsst.utils.tests
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.image


def circle(radius, num, x0, y0):
    theta = np.linspace(0, 2*np.pi, num=num, endpoint=False)
    x = radius*np.cos(theta) + x0
    y = radius*np.sin(theta) + y0
    return np.array([x, y]).transpose()

def polygon(num, radius, x0, y0):
    points = circle(radius, num, x0=x0, y0=y0)
    return afwGeom.Polygon([lsst.geom.Point2D(x, y) for x, y in reversed(points)])

n = 3
r = 23
poly = polygon(n, r, 75, 75)
box = lsst.geom.Box2I(lsst.geom.Point2I(15, 15),
                      lsst.geom.Extent2I(115, 115))
# box = lsst.geom.Box2I(lsst.geom.Point2I(15, 15),
#                       lsst.geom.Extent2I(115, 115))

image = poly.createImage(box)
diff = abs(1 - image.getArray().sum()/poly.calculateArea())
i = 0
for x in image.getArray():
    if i == 79:
        print("   ", i, x.sum())
        print("        ",x)
    i = i + 1


"""
for n in range(3,30):
    for r in range(1,50):
        poly = polygon(n, r, 75, 75)
        box = lsst.geom.Box2I(lsst.geom.Point2I(15, 15),
                    lsst.geom.Extent2I(115, 115))
        image = poly.createImage(box)
        diff = abs(1-image.getArray().sum()/poly.calculateArea())
        #print(n,r,image.getArray().sum(),poly.calculateArea())
        if n == 3 and (r == 23 or r == 25):
            i = 0;
            for x in image.getArray():
                if (r==23 and i == 79) or (r==25 and i == 81):
                    print("   ",i,x.sum())
                    print("        ",x)
                i = i + 1
"""
