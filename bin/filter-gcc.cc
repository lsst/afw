#include <iostream>
#include <string>

/**
 *  @file filter-gcc.cc
 *
 *  This is a simple filter program (reads stdin, writes stdout) to simplify
 *  the horrendous error messages g++ spits out when things go wrong with
 *  Boost.Variant.  All it does is replace the actual template arguments to
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
    std::string line;
    std::getline(input, line);
    while (input) {
        std::size_t i = line.find("boost::variant<");
        if (i == std::string::npos) {
            output << line << std::endl;
            std::getline(input, line);
            continue;
        }
        output << line.substr(0, i);
        int nBrackets = 0;
        while (++i < line.size()) {
            if (line[i] == '<') {
                ++nBrackets;
            } else if (line[i] == '>') {
                if (--nBrackets == 0) {
                    output << "boost::variant<...>";
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
