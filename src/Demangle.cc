#include <string>
#include <stack>
#include <boost/format.hpp>
#include "lsst/fw/Utils.h"
#include "lsst/fw/Trace.h"
#include "lsst/fw/Demangle.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
/*! \brief Try to demangle a C++ type
 *
 * Try to unmangle a type name; this is purely a convenience to the user
 * and need not succeed for the LSST to work
 *
 * A number of compilers (including gcc 3.2+) use
 a mangling scheme based on the "Itanium ABI";
 * see e.g.
 *    http://www.codesourcery.com/cxx-abi/abi.html#mangling
 * Contributing organisations include
 *   CodeSourcery, Compaq, EDG, HP, IBM, Intel, Red Hat, and SGI
 * which implies pretty wide acceptance.
 *
 * I chose to write my own as the demanglers on the web handle
 * much more than types (which is all I need) and have long tentacles
 * into e.g. the libiberty source.
 */
std::string demangleType(const std::string _typeName) {
#if 1
    std::string typeName("");
    const char *ptr = _typeName.c_str();

    if (*ptr == 'r' || *ptr == 'V' || *ptr == 'K') {
        ptr++;                          // (restrict/volatile/const)
    }
    
    if (*ptr == 'P') ptr++;             // We passed "this" which is (type *)

    bool startSymbolSequence = true;    // are in the middle of a set of symbs
    std::stack<char> typeStack;         // Did we last see an N or an I?

    while (*ptr != '\0') {
        switch (*ptr) {
          case 'E':
            ptr++;
            startSymbolSequence = true;

            if (typeStack.size() == 0) {
                Trace::trace("fw.Citizen.demangle", 0, boost::format("Tried to examine empty stack for %s at \"%s\"") % _typeName % ptr);
                typeStack.push('\a');   // at least don't crash
            }

            if (typeStack.top() == 'N') {
                ;
            } else if (typeStack.top() == 'I') {
                typeName += ">";
            }
            typeStack.pop();

            break;
          case 'I':
            typeStack.push(*ptr++);
            startSymbolSequence = true;

            typeName += "<";
            break;
          case 'N':
            typeStack.push(*ptr++);
            startSymbolSequence = true;
            break;
          case 'S':
            *ptr++;
            switch (*ptr) {
              case 't': typeName += "::std::"; break;
              case 'a': typeName += "::std::allocator"; break;
              case 'b': typeName += "::std::basic_string"; break;
              case 's': typeName += "::std::basic_string<char,::std::char_traits<char>,::std::allocator<char>>"; break;
              case 'i': typeName += "::std::basic_istream<char,  std::char_traits<char> >"; break;
              case 'o': typeName += "::std::basic_ostream<char,std::char_traits<char>>"; break;
              case 'd': typeName += "::std::basic_iostream<char,std::char_traits<char>>"; break;
              default:
                typeName += "[";
                if (*ptr == '_') {
                    typeName += "S0";
                } else if (isdigit(*ptr) || isupper(*ptr)) {
                    int subst = 0;
                    while (isdigit(*ptr) || isupper(*ptr)) {
                        if (isdigit(*ptr)) {
                            subst = 36*subst + (*ptr - '0');
                        } else {
                            subst = 36*subst + 10 + (*ptr - 'A');
                        }
                        *ptr++;
                    }
                    assert (*ptr == '_');
                    ptr++;
                    
                    typeName += (boost::format("S%d") % subst).str();
                }
                typeName += "]";
                break;
            }
            startSymbolSequence = true;
            break;
          case '0': case '1': case '2': case '3': case '4':
          case '5': case '6': case '7': case '8': case '9':
            if (!startSymbolSequence) {
                typeName += "::";
            }
            startSymbolSequence = false;
            
            {
                const int len = atoi(ptr++);
                while (isdigit(*ptr)) ptr++;
                
                for (int i = 0; *ptr != '\0' && i < len; i++) {
                    typeName += *ptr++;
                }
	    }
            break;
          case 'v': typeName += "void";		ptr++; break;
          case 'w': typeName += "wchar_t";	ptr++; break;
          case 'b': typeName += "bool";		ptr++; break;
          case 'c': typeName += "char";		ptr++; break;
          case 'a': typeName += "signed char";	ptr++; break;
          case 'h': typeName += "unsigned char"; ptr++; break;
          case 's': typeName += "short";	ptr++; break;
          case 't': typeName += "unsigned short"; ptr++; break;
          case 'i': typeName += "int";		ptr++; break;
          case 'j': typeName += "unsigned int";	ptr++; break;
          case 'l': typeName += "long";		ptr++; break;
          case 'm': typeName += "unsigned long"; ptr++; break;
          case 'x': typeName += "long long, __int64"; ptr++; break;
          case 'y': typeName += "unsigned long long, __int64"; ptr++; break;
          case 'n': typeName += "__int128";	ptr++; break;
          case 'o': typeName += "unsigned __int128"; ptr++; break;
          case 'f': typeName += "float";	ptr++; break;
          case 'd': typeName += "double";	ptr++; break;
          case 'e': typeName += "long double, __float80"; ptr++; break;
          case 'g': typeName += "__float128";	ptr++; break;
          case 'z': typeName += "...";		ptr++; break;
          case 'u': typeName += "vendor extended type"; ptr++; break;
          default:
            typeName += *ptr++;
        }
    }

    return typeName;
#else
    return _typeName;
#endif
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
