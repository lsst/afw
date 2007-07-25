// -*- LSST-C++ -*-
#ifndef LSST_FW_Function_H
#define LSST_FW_Function_H
/**
 * \file
 *
 * \brief Define the basic Function classes.
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <stdexcept>
#include <sstream>
#include <vector>

#include <boost/format.hpp>

#include <lsst/fw/LsstBase.h>

namespace lsst {
namespace fw {
namespace function {

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
     * \ingroup fw
     */
    template<typename ReturnT>
    class Function : private LsstBase {
    
    public:
        /**
         * \brief Construct a Function given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit Function(
            unsigned int nParams)   ///< number of function parameters
        :
            LsstBase(typeid(this)),
            _params(nParams)
        {}

        /**
         * \brief Construct a Function given the function parameters.
         */
        explicit Function(
            std::vector<double> const &params)
        :
            LsstBase(typeid(this)),
            _params(params)   ///< function parameters
        {}
        
        virtual ~Function() {};
    
        /**
         * \brief Return the number of function parameters
         */
        virtual unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * \brief Return the function parameters
         */
        virtual std::vector<double> const &getParameters() const {
            return _params;
        }
        
        /**
         * \brief Set the function parameters
         *
         * \throw std::invalid_argument if the wrong number of parameters is supplied.
         */
        virtual void setParameters(std::vector<double> const &params) {
            if (_params.size() != params.size()) {
                throw std::invalid_argument(str(boost::format(
                    "setParameters called with %d parameters instead of %d")
                    % params.size() % _params.size()));
            }
            _params = params;
        }
    
    protected:
        std::vector<double> _params;
    };
    
    
    /**
     * \brief A Function taking one argument.
     *
     * Subclass and override operator() to do useful work.
     *
     * \ingroup fw
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
    };
    
    
    /**
     * \brief A Function taking two arguments.
     *
     * Subclass and override operator() to do useful work.
     *
     * \ingroup fw
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
    };

}   // namespace functions
}   // namespace fw
}   // namespace lsst

#endif // #ifndef LSST_FW_Function_H
