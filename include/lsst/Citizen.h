// -*- lsst-c++ -*-
#if !defined(LSST_SHARED_POINTERS)      //! multiple inclusion guard macro
#define LSST_SHARED_POINTERS 1

#include <vector>
#include <map>
#include <iostream>

#include "Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)
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
 * block ID is allocated or deleted, and check whether any
 * of the data blocks are known to be corrupted
 */
    class Citizen {
    public:
        //! Type of the block's ID
        typedef unsigned long memId;
        //! A function used to register a callback
        typedef memId (*memCallback)(const Citizen *ptr);

        Citizen(const std::type_info &);
        ~Citizen();
        //
        std::string repr() const;

        static int census(int, memId startingMemId = 0);
        static void census(std::ostream &stream, memId startingMemId = 0);
        static const std::vector<const Citizen *> *census();

        static bool checkCorruption();
        
        memId getId() const;
        
        static memId getNextMemId();

        static memId setNewCallbackId(memId id);
        static memId setDeleteCallbackId(memId id);
        static memCallback setNewCallback(memCallback func);
        static memCallback setDeleteCallback(memCallback func);
        static memCallback setCorruptionCallback(memCallback func);
        //
        enum { magicSentinel = 0xdeadbeef }; //!< a magic known bit pattern
        static int init();
    private:
        typedef std::map<memId, const lsst::fw::Citizen *> table;

        int _sentinel;                  // Initialised to _magicSentinel to detect overwritten memory
        memId _id;                      // unique identifier for this pointer
        const char *_typeName;          // typeid()->name
        //
        // Book-keeping for _id
        //
        static memId _nextMemId;        // next unique identifier
        static table active_Citizens;
        //
        // Callbacks
        //
        static memId _newId;       // call _newCallback when _newID is allocated
        static memId _deleteId;    // call _deleteCallback when _deleteID is deleted

        static memCallback _newCallback;
        static memCallback _deleteCallback;        
        static memCallback _corruptionCallback;        
        //
        bool _checkCorruption() const;
    };

//! \def lsstFactory
//! The new object knows its type. E.g.<BR>
//! Shoe *z = lsstFactory(Shoe);<BR>
//! lsst::Citizen::census(std::cout);<BR>
//! \note __VA_ARGS__ is standard C, but not [yet] shandard C++
//! \note The use of ## is a gcc extension.

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
