// -*- lsst-c++ -*-
#if !defined(LSST_SHARED_POINTERS)      //! multiple inclusion guard macro
#define LSST_SHARED_POINTERS 1

#include <vector>
#include <map>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/format.hpp>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
/*! \brief Citizen is a class that should be among all LSST
 * classes base classes, and handles basic memory management
 *
 * Instances of subclasses of Citizen will automatically be
 * given a unique id.  If you use the
 *   Citizen(const char *file, const int line)
 * constructor, the values of file and line will also be
 * recorded; this is typically achieved by using
 *   Citizen(__FILE__, __LINE__)
 * where the values are provided by the pre-processor.
 *
 * \sa NEW to include the file/line information when calling new
 *
 * You can ask for infomation about the currently allocated
 * Citizens using the census functions, request that
 * a function of your choice be called when a specific
 * block ID is allocated or freed, and check whether any
 * of the data blocks are known to be corrupted
 */
    class Citizen {
    public:
        //! Type of the block's ID
        typedef unsigned long memId;
        //! A function used to register a callback
        typedef memId (*memCallback)(const Citizen *ptr);

        Citizen(const char *file = "???", const int line = 0);
        ~Citizen();
        //
        std::string repr() const;

        static int census(int dummy);
        static void census(std::ostream &stream);
        static const std::vector<const Citizen *> *census();

        static bool checkCorruption();
        
        memId getId() const;
        
        static memId setNewCallbackId(memId id);
        static memId setDeleteCallbackId(memId id);
        static memCallback setNewCallback(memCallback func);
        static memCallback setDeleteCallback(memCallback func);
        static memCallback setCorruptionCallback(memCallback func);
        //
        const static int magicSentinel; //!< a magic known bit pattern
    private:
        typedef std::map<memId, const lsst::Citizen *> table;

        int _sentinel;                  // Initialised to _magicSentinel to detect overwritten memory
        const char *_file;              // file where pointer was allocated
        const int _line;                // line where pointer was allocated
        memId _id;                      // unique identifier for this pointer
        //
        // Book-keeping for _id
        //
        static memId _nextMemId;        // next unique identifier
        static table active_Citizens;
        //
        // Callbacks
        //
        static memId _newId;       // call _newCallback when _newID is allocated
        static memId _deleteId;    // call _deleteCallback when _deleteID is freed

        static memCallback _newCallback;
        static memCallback _deleteCallback;        
        static memCallback _corruptionCallback;        
        //
        bool _checkCorruption() const;
    };

/*! \brief A subclass of boost::SMART_PTR that also subclasses
 * Citizen, so it has a unique ID and can remember where it was allocated
 *
 * \sa SCOPED_PTR/SHARED_PTR to include the file/line information
 * for the scoped_ptr (not the target) automatically
 * \sa NEW to include the file/line information when calling new
 */
/*
 * We need to define a set of these, and RHL doesn't know how
 * to use templates to do this.  So he used the CPP, which confuses
 * doxygen, is inelegant, and works.
 */
#define LSST_DEFINE_SMART_PTR(PTR_KIND)                                 \
    template<typename T>                                                \
    class PTR_KIND : public boost::PTR_KIND<T>, Citizen {               \
    public:                                                             \
        /*! Create a boost::PTR_KIND from p */                          \
        PTR_KIND(T *p) : boost::PTR_KIND<T>(p) {                        \
            ;                                                           \
        }                                                               \
        /*! Create a boost::PTR_KIND from p that remembers file/line */ \
        PTR_KIND(const char *file,    /*! The current filename */       \
                 const int line, T *p) : /*! The current line number */ \
            boost::PTR_KIND<T>(p), Citizen(file, line) {                \
            ;                                                           \
        }                                                               \
    }
//
// Actually define them.  I won't define all possibilities now, as I don't
// want to pull in all the boost::{shared,...}_{ptr,array}.hpp headers
//
// If this becomes an issue, generate them as separate files, and
// consider using m4 not cpp to do the generation
//
LSST_DEFINE_SMART_PTR(scoped_ptr);
LSST_DEFINE_SMART_PTR(scoped_array);
LSST_DEFINE_SMART_PTR(shared_ptr);
LSST_DEFINE_SMART_PTR(shared_array);
    
#undef LSST_DEFINE_SMART_PTR

//! Declare an initialized scoped pointer that inherits from Citizen
#define SHARED_ARRAY(TYPE, VAR, PTR)                            \
    lsst::shared_array<TYPE> VAR(__FILE__, __LINE__, PTR)

#define SCOPED_ARRAY(TYPE, VAR, PTR)                            \
    lsst::scoped_array<TYPE> VAR(__FILE__, __LINE__, PTR)

#define SHARED_PTR(TYPE, VAR, PTR)                      \
    lsst::shared_ptr<TYPE> VAR(__FILE__, __LINE__, PTR)

#define SCOPED_PTR(TYPE, VAR, PTR)                      \
    lsst::scoped_ptr<TYPE> VAR(__FILE__, __LINE__, PTR)
//! \def SCOPED_PTR
//! TYPE is the desired type for VAR, initialised to PTR. E.g.<BR>
//! SCOPED_PTR(Shoe, y, new Shoe);<BR>
//! SCOPED_PTR(Shoe, x, NEW(Shoe));<BR>
//! lsst::Citizen::census(std::cout);<BR>
//!
//! The former knows which line the scoped pointer was allocated on;
//! the latter also knows where the Shoe was allocated.
//!
//! \note There is also SCOPED_ARRAY, SHARED_PTR, and SHARED_ARRAY

//! Return a new object of TYPE, subclassed from Citizen, initialised with one or more arguments
#define NEW(TYPE, ...)                                  \
    new TYPE(__FILE__, __LINE__, ## __VA_ARGS__)
//! \def NEW
//! The new object knows where it was born. E.g.<BR>
//! Shoe *z = NEW(Shoe);<BR>
//! lsst::Citizen::census(std::cout);<BR>
//! \note __VA_ARGS__ is standard C, but not [yet] shandard C++
//! \note The use of ## is a gcc extension.
LSST_END_NAMESPACE(lsst)
#endif
