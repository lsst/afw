%{
    #include <boost/serialization/serialization.hpp>
    #include <boost/archive/binary_oarchive.hpp>
    #include <boost/archive/binary_iarchive.hpp>
    #include <sstream>
%}

%include "std_string.i"

%define %boost_picklable(cls...)
    %extend cls {
        std::string __getstate__()
        {
            std::stringstream ss;
            boost::archive::binary_oarchive ar(ss);
            ar << *($self);
            return ss.str();
        }

        void __setstate_internal(std::string const& sState)
        {
            std::stringstream ss(sState);
            boost::archive::binary_iarchive ar(ss);
            ar >> *($self);
        }


        %pythoncode %{
            def __setstate__(self, sState):
                self.__init__()
                self.__setstate_internal(sState)
        %}
    }
%enddef
