import lsst.utils

def version():
    """Return current version. If a different version is setup, return that too"""

    HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/tags/3.5.2/python/lsst/afw/__init__.py $"
    return lsst.utils.version(HeadURL, "afw")

