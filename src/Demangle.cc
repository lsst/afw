#include <string>
#include <stack>
#include <boost/format.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include "lsst/fw/Utils.h"
#include "lsst/fw/Trace.h"
#include "lsst/fw/Demangle.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
/*
 * Provide a symbol table for "substitutions" while mangling
 */
using boost::multi_index_container;
using namespace boost::multi_index;

class Symbol {
public:
    int n;                              // index of substitution
    std::string key;                    // key to saved string

    Symbol(std::string key) : n(n_next++), key(key) { }
    ~Symbol() {}

    static void reset() { n_next = 0; } // reset counter
    
    void print() const {
        std::cout << '\t' << n << " " << key << '\n';
    }
private:
    static int n_next;                  // the next value of n
};

/*
 * Tags for indices
 */
struct n {};                            // lookup by n
struct key {};                          // lookup by key

int Symbol::n_next = 0;                 // unique ID for each symbol

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
static bool interpret_typeletter(const char c, std::string &type) {
    switch (c) {
      case 'v': type = "void"; return true;
      case 'w': type = "wchar_t"; return true;
      case 'b': type = "bool"; return true;
      case 'c': type = "char"; return true;
      case 'a': type = "schar"; return true;
      case 'h': type = "uchar"; return true;
      case 's': type = "short"; return true;
      case 't': type = "ushort"; return true;
      case 'i': type = "int"; return true;
      case 'j': type = "uint"; return true;
      case 'l': type = "long"; return true;
      case 'm': type = "ulong"; return true;
      case 'x': type = "long long"; return true;
      case 'y': type = "ulong long"; return true;
      case 'n': type = "__int128"; return true;
      case 'o': type = "__uint128"; return true;
      case 'f': type = "float"; return true;
      case 'd': type = "double"; return true;
      case 'e': type = "long double"; return true;
      case 'g': type = "__float128"; return true;
      case 'z': type = "..."; return true;
      case 'u': type = "vendor extended type"; return true;
      default: return false;
    }
}


std::string demangleType(const std::string _typeName) {
#if 1
    typedef multi_index_container<
        Symbol,
        indexed_by<
            ordered_unique<tag<n>,
        	member<Symbol, int, &Symbol::n> >,
            ordered_unique<tag<key>,
        	member<Symbol, std::string, &Symbol::key> >
        >
    > SymbolTable;
    typedef SymbolTable::index<n>::type::iterator nIterator;
    typedef SymbolTable::index<key>::type::iterator keyIterator;
    Symbol::reset();
   
    // Here's my symbol table and its indices
    SymbolTable st;
    
    SymbolTable::index<n>::type &nIndex = st.get<n>();
    SymbolTable::index<key>::type &keyIndex = st.get<key>();
    //
    // Start mangling
    //
    std::string typeName("");
    const char *ptr = _typeName.c_str();

    if (*ptr == 'r' || *ptr == 'V' || *ptr == 'K') {
        ptr++;                          // (restrict/volatile/const)
    }
    
    if (*ptr == 'P') ptr++;             // We passed "this" which is (type *)

    std::string currentSymbol = "";     // Current symbol
    std::stack<char> typeStack;         // Did we last see an N or an I?

    while (*ptr != '\0') {
        switch (*ptr) {
          case 'E':
            ptr++;
            currentSymbol = "";

            if (typeStack.empty()) {
                Trace("fw.Citizen.demangle", 0,
                      boost::format("Tried to examine empty stack for %s at \"%s\"") % _typeName % ptr);
                typeStack.push('\a');   // at least don't crash
            }

            if (typeStack.top() == 'I') {
                typeName += '>';
            } else if (typeStack.top() == 'L') {
                ;
            } else if (typeStack.top() == 'N') {
                ;
            }
            typeStack.pop();

            if (!typeStack.empty() && typeStack.top() == 'I') {
                if (*ptr != 'E' && typeName[typeName.size() - 1] != '<') {
                    typeName += ',';
                }
            }
                        
            break;
          case 'I':
            typeStack.push(*ptr++);
            currentSymbol = "";

            typeName += '<';
            break;
          case 'L':
            typeStack.push(*ptr++);
            currentSymbol = "";
            {
                std::string type;
                if (interpret_typeletter(*ptr, type)) {
                    typeName += "(" + type + ')';
                } else {
                    typeName += 'c';
                }
                ptr++;
            }
            if (*ptr == 'n') {
                typeName += '-'; ptr++;
            }
            while (*ptr != '\0' && *ptr != 'E') {
                typeName += *ptr++;
            }
            break;
          case 'N':
            typeStack.push(*ptr++);
            currentSymbol = "";
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
                {
                    int subst = 0;      // number of substitution

                    if (*ptr == '_') {
                        ;                   // S_ => 0
                    } else if (isdigit(*ptr) || isupper(*ptr)) {
                        while (isdigit(*ptr) || isupper(*ptr)) {
                            if (isdigit(*ptr)) {
                                subst = 36*subst + (*ptr - '0');
                            } else {
                                subst = 36*subst + 10 + (*ptr - 'A');
                            }
                            *ptr++;
                        }
                        subst++;            // S_ == 0; S1_ == 1
                        assert (*ptr == '_');
                        ptr++;
                    }

                    nIterator sym = nIndex.find(subst);
                    if (sym == nIndex.end()) { // not found
                        typeName += (boost::format("[S%d]") % subst).str();
                    } else {
                        typeName += sym->key;
                    }

                }
                break;
            }
            currentSymbol = "";
            break;
          case '0': case '1': case '2': case '3': case '4':
          case '5': case '6': case '7': case '8': case '9':
            {
                const int len = atoi(ptr++);
                while (isdigit(*ptr)) ptr++;

                std::string name = "";
                for (int i = 0; *ptr != '\0' && i < len; i++) {
                    name += *ptr++;
                }

                if (currentSymbol != "") {
                    currentSymbol += "::";
                    typeName += "::";
                }
            
                currentSymbol += name;
                typeName += name;

                if (keyIndex.find(currentSymbol) == keyIndex.end()) {
                    st.insert(currentSymbol);
                }
	    }
            break;
          default:
            {
                std::string type;
                if (interpret_typeletter(*ptr, type)) {
                    typeName += type;
                } else {
                    typeName += *ptr;
                }
                ptr++;
            }
        }
    }

    static volatile bool dumpSymbolTable = false; // can be set from gdb
    if (dumpSymbolTable) {
        // The test on the iterator is paranoid, but they _could_
        // have deleted elements.  In this case, they didn't.
        for (unsigned int i = 0; i < st.size(); i++) {
            nIterator el = nIndex.find(2);
            if (el != nIndex.end()) {          // did we find it?
                el->print();
            }
        }
    }

    return typeName;
#else
    return _typeName;
#endif
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
