// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_FUNCTION_H
#define LSST_AFW_MATH_FUNCTION_H
/**
 * @file
 *
 * @brief Define the basic Function classes.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <stdexcept>
#include <sstream>
#include <vector>

#include "boost/format.hpp"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/void_cast.hpp"
#include "boost/serialization/export.hpp"

#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

#ifndef SWIG
using boost::serialization::make_nvp;
#endif

    /**
     * @brief Basic Function class.
     *
     * Function objects are functions whose parameters may be read and changed using
     * getParameters and setParameters. They were designed for use with the Kernel class.
     *
     * These are simple functors with the restrictions that:
     * - Function arguments and parameters are double precision
     * - The return type is templated
     *
     * To create a function for a particular equation, subclass Function
     * or (much more likely) Function1 or Function2. Your subclass must:
     * - Have one or more constructors, all of which must initialize _params
     * - Define operator() with code to compute the function
     *   using this->_params or this->getParams() to reference the parameters
     *
     * Design Notes:
     * The reason these functions exist (rather than using a pre-existing function class,
     * such as Functor in VisualWorkbench) is because the Kernel class requires function
     * objects with a standard interface for setting and getting function parameters.
     *
     * To do:
     * - Implement separable functions
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Function : public lsst::daf::data::LsstBase {
    
    public:
        /**
         * @brief Construct a Function given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function(
            unsigned int nParams)   ///< number of function parameters
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _params(nParams)
        {}

        /**
         * @brief Construct a Function given the function parameters.
         */
        explicit Function(
            std::vector<double> const &params)
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _params(params)   ///< function parameters
        {}
        
        virtual ~Function() {};
    
        /**
         * @brief Return the number of function parameters
         *
         * @return the number of function parameters
         */
        unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * @brief Get one function parameter without range checking
         *
         * @return the specified function parameter
         */
        virtual double getParameter(
            unsigned int ind)    ///< index of parameter
        const {
            return _params[ind];
        }
        
        /**
         * @brief Return all function parameters
         *
         * @return the function parameters as a vector
         */
        std::vector<double> const &getParameters() const {
            return _params;
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _params[ind] = newValue;
        };
        
        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameterException
         *        if the wrong number of parameters is supplied.
         */
        void setParameters(
            std::vector<double> const &params)   ///< vector of function parameters
        {
            if (_params.size() != params.size()) {
                throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                    (boost::format("params has %d entries instead of %d") % \
                    params.size() % _params.size()).str());
            }
            _params = params;
        }
    
        /**
         * @brief Return a string representation of the function
         *
         * @return a string representation of the function
         */
        virtual std::string toString(std::string const&) const {
            std::stringstream os;
            os << "parameters: [ ";
            for (std::vector<double>::const_iterator i = _params.begin(); i != _params.end(); ++i) {
                if (i != _params.begin()) os << ", ";
                os << *i;
            }
            os << " ]";
            return os.str();
        };

    protected:
        std::vector<double> _params;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
        };
    };   

    
    /**
     * @brief A Function taking one argument.
     *
     * Subclass and override operator() to do useful work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Function1 : public Function<ReturnT> {
    public:
        typedef boost::shared_ptr<Function1<ReturnT> > Ptr;

        /**
         * @brief Construct a Function1 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function1(
            unsigned int nParams)   ///< number of function parameters
        :
            Function<ReturnT>(nParams)
        {}

        /**
         * @brief Construct a Function1 given the function parameters.
         */
        explicit Function1(
            std::vector<double> const &params)   ///< function parameters
        :
            Function<ReturnT>(params)
        {}
        
        virtual ~Function1() {};
        
        /**
         * @brief Return a pointer to a deep copy of this function
         *
         * This function exists instead of a copy constructor
         * so one can obtain a copy of an actual function
         * instead of a useless copy of the base class.
         *
         * Every non-virtual function must override this method.
         *
         * @return a pointer to a deep copy of the function
         */
        virtual Ptr clone() const = 0; 
    
        virtual ReturnT operator() (double x) const = 0;

        virtual std::string toString(std::string const& prefix="") const {
            return std::string("Function1: ") + Function<ReturnT>::toString(prefix);
        };

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            boost::serialization::void_cast_register<
                Function1<ReturnT>, Function<ReturnT> >(
                    static_cast< Function1<ReturnT>* >(0),
                    static_cast< Function<ReturnT>* >(0));
        };
    };    

    
    /**
     * @brief A Function taking two arguments.
     *
     * Subclass and override operator() to do useful work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Function2 : public Function<ReturnT> {
    public:
        typedef boost::shared_ptr<Function2<ReturnT> > Ptr;

        /**
         * @brief Construct a Function2 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function2(
            unsigned int nParams)   ///< number of function parameters
        :
            Function<ReturnT>(nParams)
        {}

        /**
         * @brief Construct a Function2 given the function parameters.
         *
         * The number of function parameters is set to the length of params.
         */
        explicit Function2(
            std::vector<double> const &params)   ///< function parameters
        :
            Function<ReturnT>(params)
        {}
        
        virtual ~Function2() {};
        
        /**
         * @brief Return a pointer to a deep copy of this function
         *
         * This function exists instead of a copy constructor
         * so one can obtain a copy of an actual function
         * instead of a useless copy of the base class.
         *
         * Every non-virtual function must override this method.
         *
         * @return a pointer to a deep copy of the function
         */
        virtual Ptr clone() const = 0; 
    
        virtual ReturnT operator() (double x, double y) const = 0;

        virtual std::string toString(std::string const& prefix="") const {
            return std::string("Function2: ") + Function<ReturnT>::toString(prefix);
        };

    private:
        friend class boost::serialization::access;
#ifndef SWIG
        template <class Archive>
        void serialize(Archive& ar, unsigned const int version) {
            boost::serialization::void_cast_register<
                Function2<ReturnT>, Function<ReturnT> >(
                    static_cast< Function2<ReturnT>* >(0),
                    static_cast< Function<ReturnT>* >(0));
        };
#endif
    };


    /**
     * @brief a class used in function calls to indicate that no Function1 is being provided
     */
    template<typename ReturnT>
    class NullFunction1 : public Function1<ReturnT> {
    public:
        explicit NullFunction1() : Function1<ReturnT>(0) {}
        typename Function1<ReturnT>::Ptr clone() const {
            return typename Function1<ReturnT>::Ptr(new NullFunction1()); }

    private:
        ReturnT operator() (double) const { return static_cast<ReturnT>(0); }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            ar & make_nvp("fn",
                          boost::serialization::base_object<
                          Function<ReturnT> >(*this));
        };
    };

    /**
     * @brief a class used in function calls to indicate that no Function2 is being provided
     */
    template<typename ReturnT>
    class NullFunction2 : public Function2<ReturnT> {
    public:
        explicit NullFunction2() : Function2<ReturnT>(0) {}
        typename Function2<ReturnT>::Ptr clone() const {
            return typename Function2<ReturnT>::Ptr(new NullFunction2()); }

    private:
        ReturnT operator() (double, double) const { return static_cast<ReturnT>(0); }

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned int const version) {
            ar & make_nvp("fn",
                          boost::serialization::base_object<
                          Function<ReturnT> >(*this));
        };
    };

}}}   // lsst::afw::math

namespace boost {
namespace serialization {

template <class Archive, typename ReturnT>
inline void save_construct_data(Archive& ar,
                                lsst::afw::math::Function<ReturnT> const* f,
                                unsigned int const version) {
    ar << make_nvp("params", f->getParameters());
};

template <class Archive, typename ReturnT>
inline void load_construct_data(Archive& ar,
                                lsst::afw::math::Function<ReturnT>* f,
                                unsigned int const version) {
    std::vector<double> params;
    ar >> make_nvp("params", params);
    ::new(f) lsst::afw::math::Function<ReturnT>(params);
};

}}

#endif // #ifndef LSST_AFW_MATH_FUNCTION_H
