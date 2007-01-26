// -*- lsst-c++ -*-
#if !defined(LSST_SHARED_POINTERS)
#define LSST_SHARED_POINTERS 1

#include <vector>
#include <map>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>

namespace lsst {
    class citizen {
    public:
        typedef unsigned long memId;
        typedef memId (*memCallback)(const citizen *ptr);

        citizen(const char *file = "???", const int line = 0);
        ~citizen();

        std::string repr() const;
        
        static int census(int dummy);   // the int allows overloading
        static void census(std::ostream &stream);
        static const std::vector<const citizen *> *census();
        //
        // Return private state
        //
        memId getId() const;
        //
        // Set/retrieve callbackIds and their callbacks
        //
        static memId setNewCallbackId(memId id);
        static memId setDeleteCallbackId(memId id);

        static memCallback setNewCallback(memCallback func);
        static memCallback setDeleteCallback(memCallback func);
    private:
        typedef std::map<memId, const lsst::citizen *> table;

        const char *_file;              // file where pointer was allocated
        const int _line;                // line where pointer was allocated
        memId _id;                      // unique identifier for this pointer
        //
        // Book-keeping for _id
        //
        static memId _nextMemId;        // next unique identifier
        static table active_citizens;
        //
        // Callbacks
        //
        static memId _newId;       // call _newCallback when _newID is allocated
        static memId _deleteId;    // call _deleteCallback when _deleteID is freed

        static memCallback _newCallback;
        static memCallback _deleteCallback;        
    };

    template<typename T>
    class scoped_ptr : public boost::scoped_ptr<T>, citizen {
    public:
        scoped_ptr(T *p) : boost::scoped_ptr<T>(p) {
            ;
        }
        scoped_ptr(const char *file,  const int line, T *p) :
            boost::scoped_ptr<T>(p), citizen(file, line) {
            ;
        }
    };
    // Declare a scoped pointer that doesn't inherit from citizen
    #define SCOPED_PTR0(TYPE, VAR) \
        scoped_ptr<TYPE> VAR
    // Declare an initialized scoped pointer that inherits from citizen
    #define SCOPED_PTR(TYPE, VAR, PTR) \
        scoped_ptr<TYPE> VAR(__FILE__, __LINE__, PTR)
    #define NEW(TYPE, ...) \
        new TYPE(__FILE__, __LINE__, ## __VA_ARGS__)
}
#endif
