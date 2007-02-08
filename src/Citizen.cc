//! \file
//! \brief Implementation of Citizen

#include <iostream>
#include "lsst/Citizen.h"
#include "lsst/Exception.h"

using namespace lsst;
//
// Con/Destructors
//
Citizen::Citizen(const char *file,  //!< File where Citizen was allocated
                 const int line    //!< Line where Citizen was allocated
    ) : _file(file), _line(line) {
    _id = ++_nextMemId;
    active_Citizens[_id] = this;

    _sentinel = magicSentinel;

    if (_id == _newId) {
        _newId += _newCallback(this);
    }
}
Citizen::~Citizen() {
    if (_id == _deleteId) {
        _deleteId += _deleteCallback(this);
    }

    (void)_checkCorruption();
    
    active_Citizens.erase(_id);
}
//
// Return (some) private state
//
//! Return the Citizen's ID
Citizen::memId Citizen::getId() const {
    return _id;
}

//! Return a string representation of a Citizen
//
std::string Citizen::repr() const {
    return boost::str(boost::format("%d: %p %-20s") % _id % this %
                      (boost::format("%s:%d") % _file % _line));
}

//! \name Census
//! Provide a list of current Citizens
//@{
//
//
//! How many active Citizens are there?
//
int Citizen::census(
    int dummy                           //! the int argument allows overloading
    ) {
    return active_Citizens.size();
}
//
//! Print a list of all active Citizens to stream
//
void Citizen::census(
    std::ostream &stream                //! stream to print to
    ) {
    for (table::iterator cur = active_Citizens.begin();
         cur != active_Citizens.end(); cur++) {
        stream << cur->second->repr() << "\n";
    }
}
//
//! Return a (newly allocated) std::vector of active Citizens
//
//! You are responsible for freeing it; or you can say
//!    boost::scoped_ptr<const std::vector<const Citizen *> >
//!					leaks(Citizen::census());
//! and not bother
//
const std::vector<const Citizen *> *Citizen::census() {
    std::vector<const Citizen *> *vec =
        new std::vector<const Citizen *>(0);
    vec->reserve(active_Citizens.size());

    for (table::iterator cur = active_Citizens.begin();
         cur != active_Citizens.end(); cur++) {
        vec->push_back(dynamic_cast<const Citizen *>(cur->second));
    }
        
    return vec;
}
//@}

//! Check for corruption
//! Return true if the block is corrupted, but
//! only after calling the corruptionCallback
bool Citizen::_checkCorruption() const {
    if (_sentinel == magicSentinel) {
        return false;
    }

    (void)_corruptionCallback(this);
    return true;
}

//! Check all allocated blocks for corruption
bool Citizen::checkCorruption() {
    for (table::iterator cur = active_Citizens.begin();
         cur != active_Citizens.end(); cur++) {
        if (cur->second->_checkCorruption()) {
            return true;
        }
    }

    return false;
}

//! \name callbackIDs
//! Set callback Ids. The old Id is returned
//@{
//
//! Call the NewCallback when block is allocated
Citizen::memId Citizen::setNewCallbackId(
    Citizen::memId id                   //!< Desired ID
    ) {
    Citizen::memId oldId = _newId;
    _newId = id;

    return oldId;
}

//! Call the current DeleteCallback when block is deleted
Citizen::memId Citizen::setDeleteCallbackId(
    Citizen::memId id                   //!< Desired ID
    ) {
    Citizen::memId oldId = _deleteId;
    _deleteId = id;

    return oldId;
}
//@}

//! \name callbacks
//! Set the New/Delete callback functions; in each case
//! the previously installed callback is returned. These
//! callback functions return a value which is Added to
//! the previously registered id.
//!
//! The default callback functions are called
//! default{New,Delete}Callback; you may want to set a break
//! point in these callbacks from your favourite debugger
//

//@{
//! Set the NewCallback function

Citizen::memCallback Citizen::setNewCallback(
    Citizen::memCallback func //! The new function to be called when a designated block is allocated
    ) {
    Citizen::memCallback old = _newCallback;
    _newCallback = func;

    return old;
}

//! Set the DeleteCallback function
Citizen::memCallback Citizen::setDeleteCallback(
    Citizen::memCallback func           //!< function be called when desired block is freed
    ) {
    Citizen::memCallback old = _deleteCallback;
    _deleteCallback = func;

    return old;
}
    
//! Set the DeleteCallback function
Citizen::memCallback Citizen::setCorruptionCallback(
	Citizen::memCallback func //!< function be called when block is found to be corrupted
                                                   ) {
    Citizen::memCallback old = _corruptionCallback;
    _corruptionCallback = func;

    return old;
}
    
//! Default callbacks.
//!
//! Note that these may well be the target of debugger breakpoints, so e.g. dId
//! may well be changed behind our back
//@{
//! Default NewCallback
Citizen::memId defaultNewCallback(const Citizen *ptr //!< Just-allocated Citizen
                                 ) {
    static int dId = 0;             // how much to incr memId
    std::cerr << boost::format("Allocating memId %s\n") % ptr->repr();

    return dId;
}

//! Default DeleteCallback
Citizen::memId defaultDeleteCallback(const Citizen *ptr //!< About-to-be freed Citizen
                                    ) {
    static int dId = 0;             // how much to incr memId
    std::cerr << boost::format("Freeing memId %s\n") % ptr->repr();

    return dId;
}

//! Default CorruptionCallback
Citizen::memId defaultCorruptionCallback(const Citizen *ptr //!< About-to-be freed Citizen
                              ) {
    throw lsst::Memory(str(boost::format("Memory block \"%s\" is corrupted") % ptr->repr()));

    return ptr->getId();                // NOTREACHED
}

//@}
//
// Initialise static members
//
const int Citizen::magicSentinel = 0xdeadbeef;

Citizen::memId Citizen::_nextMemId = 0;
Citizen::table Citizen::active_Citizens = Citizen::table();

Citizen::memId Citizen::_newId = 0;
Citizen::memId Citizen::_deleteId = 0;

Citizen::memCallback Citizen::_newCallback = defaultNewCallback;
Citizen::memCallback Citizen::_deleteCallback = defaultDeleteCallback;
Citizen::memCallback Citizen::_corruptionCallback = defaultCorruptionCallback;
