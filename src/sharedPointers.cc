#include <iostream>
#include "sharedPointers.h"

namespace lsst {
    //
    // Con/Destructors
    //
    citizen::citizen(const char *file,  const int line) :
        _file(file), _line(line) {
        _id = ++_nextMemId;
        active_citizens[_id] = this;

        if (_id == _newId) {
            _newId += _newCallback(this);
        }
    }
    citizen::~citizen() {
        if (_id == _deleteId) {
            _deleteId += _deleteCallback(this);
        }

        active_citizens.erase(_id);
    }
    //
    // Return (some) private state
    //
    citizen::memId citizen::getId() const {
        return _id;
    }
    //
    // Return a string representation of a citizen
    //
    std::string citizen::repr() const {
        return boost::str(boost::format("%d: %p %-20s") % _id % this %
                          (boost::format("%s:%d") % _file % _line));
    }
    //
    // How many active citizens are there?
    //
    int citizen::census(int dummy) {    // the int argument allows overloading
        return active_citizens.size();
    }
    //
    // Print a list of all active citizens to stdout
    //
    void citizen::census(std::ostream &stream) {
        for (table::iterator cur = active_citizens.begin();
             cur != active_citizens.end(); cur++) {
            stream << cur->second->repr() << "\n";
        }
    }
    //
    // Return a new std::vector of all active citizens
    //
    const std::vector<const citizen *> *citizen::census() {
        std::vector<const citizen *> *vec =
            new std::vector<const citizen *>(0);
        vec->reserve(active_citizens.size());

        for (table::iterator cur = active_citizens.begin();
             cur != active_citizens.end(); cur++) {
            vec->push_back(dynamic_cast<const citizen *>(cur->second));
        }
        
        return vec;
    }
    //
    // Set callback Ids, returning old values
    //
    citizen::memId citizen::setNewCallbackId(citizen::memId id) {
        citizen::memId oldId = _newId;
        _newId = id;

        return oldId;
    }

    citizen::memId citizen::setDeleteCallbackId(citizen::memId id) {
        citizen::memId oldId = _deleteId;
        _deleteId = id;

        return oldId;
    }
    //
    // Install new callbacks
    //
    citizen::memCallback citizen::setNewCallback(citizen::memCallback func) {
        citizen::memCallback old = _newCallback;
        _newCallback = func;

        return old;
    }
    citizen::memCallback citizen::setDeleteCallback(citizen::memCallback func) {
        citizen::memCallback old = _deleteCallback;
        _deleteCallback = func;

        return old;
    }
    
    //
    // Default callbacks.  Note that these may well be the
    // target of debugger breakpoints, so e.g. dId may well
    // be changed behind our back
    //
    citizen::memId defaultNewCallback(const citizen *ptr) {
        static int dId = 0;             // how much to incr memId
        std::cerr << boost::format("Allocating memId %s\n") % ptr->repr();

        return dId;
    }
    citizen::memId defaultDeleteCallback(const citizen *ptr) {
        static int dId = 0;             // how much to incr memId
        std::cerr << boost::format("Freeing memId %s\n") % ptr->repr();

        return dId;
    }
    //
    // Initialise static members.  Could use singleton pattern
    //
    citizen::memId citizen::_nextMemId = 0;
    citizen::table citizen::active_citizens = citizen::table();

    citizen::memId citizen::_newId = 0;
    citizen::memId citizen::_deleteId = 0;

    citizen::memCallback citizen::_newCallback = defaultNewCallback;
    citizen::memCallback citizen::_deleteCallback = defaultDeleteCallback;
}
