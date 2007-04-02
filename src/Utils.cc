#include <iostream>
#include <string>
#include <stack>
#include <boost/regex.hpp>
#include "lsst/fw/Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
/*!
 * Return a version name given an SVN HeadURL
 *
 * Given a string of the form
 *   Dollar HeadURL: svn+ssh://svn.lsstcorp.org/DC2/fw/tags/1.1/foo Dollar
 * (where I've written Dollar to foil svn) try to guess the version.
 *
 *    If the string is misformed, return "(NOSVN)";
 *
 *    If the version appears to be on the trunk, return "svn"
 *    as this is presumably an untagged * version
 *
 *    If the version appears to be in branches, tags, or tickets return
 *    the version string (with " B", "", or " T" appended respectively)
 *
 *    Otherwise return the svn URL
 *
 * Note: for this to be set by svn, you'll have to set the svn property
 * svn:keywords to expand HeadURL in the file where the HeadURL originates.
 */
void guessSvnVersion(const std::string &headURL, //!< the HeadURL String
                     std::string &version //!< the desired version
                    ) {
    const boost::regex getURL("^\\$HeadURL:\\s+([^$ ]+)\\s*\\$$");
    boost::smatch matchObject;
    if (boost::regex_search(headURL, matchObject, getURL)) {
        version = matchObject[1];

        const boost::regex getVersion("(branches|tags|tickets|trunk)/([^/]+)");
        if (boost::regex_search(version, matchObject, getVersion)) {
            std::string type = matchObject[1];
            version = matchObject[2];
        
            if (type == "branches") {
                version += " B";
            } else if (type == "tickets") {
                version += " T" ;
            } else if (type == "trunk") {
                version = "svn";
            }
        }
    } else {
        version = "(NOSVN)";
        return;
    }
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
