import lsst.utils

def version():
    """Return current version. If a different version is setup, return that too"""

    HeadURL = r"$HeadURL$"
    return lsst.utils.version(HeadURL, "afw")

