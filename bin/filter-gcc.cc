#include "boost/regex.hpp"

#include <iostream>
#include <string>

/**
 *  @file filter-gcc.cc
 *
 *  This is a simple filter program (reads stdin, writes stdout) to simplify
 *  the horrendous error messages g++ spits out when things go wrong with
 *  Boost.Variant.  Most of what it does is replace the actual template arguments to
 *  boost::variant with "...".
 *
 *  To use it, you'll have to find a way to pipe it stderr, not stdout, because
 *  that's where the error messages appear.  The following will do the trick,
 *  but also merge stderr and stdout:
 *  @code
 *  scons 2>&1 | bin/filter-gcc
 *  @endcode
 */

void process(std::istream & input, std::ostream & output) {
    static boost::regex const voidListRegex("(, (T\\d+ = )boost::detail::variant::void_)+");
    std::string line;
    std::string replacement;
    std::getline(input, line);
    while (input) {
        line = boost::regex_replace(line, voidListRegex, ", ...");
        std::size_t i1 = line.find("boost::variant<");
        std::size_t i2 = line.find("boost::detail::variant::over_sequence<");
        std::size_t i = 0;
        if (i1 < i2) {
            i = i1;
            output << line.substr(0, i);
            output << "boost::variant<...>";
        } else if (i1 > i2) {
            i = i2;
            output << line.substr(0, i);
            output << "boost::detail::variant::over_sequence<...>";
        } else {
            output << line << std::endl;
            std::getline(input, line);
            continue;
        }
        int nBrackets = 0;
        while (++i < line.size()) {
            if (line[i] == '<') {
                ++nBrackets;
            } else if (line[i] == '>') {
                if (--nBrackets == 0) {
                    ++i;
                    line = line.substr(i, line.size() - i);
                    break;
                }
            }
        }
    }
}

int main() {
    process(std::cin, std::cout);
    return 0;
}
