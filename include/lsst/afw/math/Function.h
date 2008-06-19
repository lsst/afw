// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_FUNCTION_H
#define LSST_AFW_MATH_FUNCTION_H
/**
 * \file
 *
 * \brief Define the basic Function classes.
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <stdexcept>
#include <sstream>
#include <vector>

#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

    /**
     * \brief Basic Function class.
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
     * \ingroup afw
     */
    template<typename ReturnT>
    class Function : public lsst::daf::data::LsstBase {
    
    public:
        typedef boost::shared_ptr<Function<ReturnT> > Ptr;
        /**
         * \brief Construct a Function given the number of function parameters.
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
         * \brief Construct a Function given the function parameters.
         */
        explicit Function(
            std::vector<double> const &params)
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _params(params)   ///< function parameters
        {}
        
        virtual ~Function() {};
        
        /**
         * \brief Return a pointer to a deep copy of this function
         *
         * This function exists instead of a copy constructor
         * so one can obtain a copy of an actual function
         * instead of a useless copy of the base class.
         *
         * Every non-virtual function must override this method.
         *
         * \return a pointer to a deep copy of the function
         */
        virtual Ptr copy() const = 0; 
    
        /**
         * \brief Return the number of function parameters
         *
         * \return the number of function parameters
         */
        unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * \brief Return the function parameters
         *
         * \return the function parameters
         */
        std::vector<double> const &getParameters() const {
            return _params;
        }
        
        /**
         * \brief Set the function parameters
         *
         * \throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        void setParameters(std::vector<double> const &params) {
            if (_params.size() != params.size()) {
                throw lsst::pex::exceptions::InvalidParameter("Wrong number of parameters");
            }
            _params = params;
        }
    
        /**
         * \brief Return a string representation of the function
         *
         * \return a string representation of the function
         */
        virtual std::string toString(void) const {
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
    };   
    
    /**
     * \brief A Function taking one argument.
     *
     * Subclass and override operator() to do useful work.
     *
     * \ingroup afw
     */
    template<typename ReturnT>
    class Function1 : public Function<ReturnT> {
    public:
        /**
         * \brief Construct a Function1 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function1(
            unsigned int nParams)   ///< number of function parameters
        :
            Function<ReturnT>(nParams)
        {}

        /**
         * \brief Construct a Function1 given the function parameters.
         */
        explicit Function1(
            std::vector<double> const &params)   ///< function parameters
        :
            Function<ReturnT>(params)
        {}
        
        virtual ~Function1() {};
    
        virtual ReturnT operator() (double x) const = 0;

        virtual std::string toString(void) const {
            return std::string("Function1: ") + Function<ReturnT>::toString();
        };
    };    
    
    /**
     * \brief A Function taking two arguments.
     *
     * Subclass and override operator() to do useful work.
     *
     * \ingroup afw
     */
    template<typename ReturnT>
    class Function2 : public Function<ReturnT> {
    public:
        /**
         * \brief Construct a Function2 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function2(
            unsigned int nParams)   ///< number of function parameters
        :
            Function<ReturnT>(nParams)
        {}

        /**
         * \brief Construct a Function2 given the function parameters.
         *
         * The number of function parameters is set to the length of params.
         */
        explicit Function2(
            std::vector<double> const &params)   ///< function parameters
        :
            Function<ReturnT>(params)
        {}
        
        virtual ~Function2() {};
    
        virtual ReturnT operator() (double x, double y) const = 0;

        virtual std::string toString(void) const {
            return std::string("Function2: ") + Function<ReturnT>::toString();
        };
    };

}}}   // lsst::afw::math

#endif // #ifndef LSST_AFW_MATH_FUNCTION_H
