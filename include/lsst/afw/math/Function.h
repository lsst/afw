// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_FUNCTION_H
#define LSST_AFW_MATH_FUNCTION_H
/**
 * @file
 *
 * @brief Define the basic Function classes.
 *
 * Function objects are functions whose parameters may be read and changed using
 * getParameters and setParameters. They were designed for use with the Kernel class.
 *
 * These are simple functors with the restrictions that:
 * - Function arguments and parameters are double precision
 * - The return type is templated
 *
 * At present support exists for 1-d functions, 2-d functions
 * and 2-d separable functions. It would be easy to add 3-d support.
 *
 * To create a simple (nonseparable) function for a particular equation, subclass one of
 * SimpleFunction1 or SimpleFunction2. Your subclass must:
 * - Have one or more constructors, all of which must initialize the parent class.
 * - Define operator() with code to compute the function
 *   using this->_params or this->getParams() to reference the parameters
 *
 * To create a separable function for a particular equation, you have two choices:
 * - Create a SeparableFunction2 with the appropriate arguments
 * - Subclass SeparableFunction2
 *
 * Design Notes:
 * The reason these functions exist (rather than using a pre-existing function class,
 * such as Functor in VisualWorkbench) is because the Kernel class requires function
 * objects with a standard interface for setting and getting function parameters.
 *
 * @todo
 * - Implement function cloning.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Parameters.h"

namespace lsst {
namespace afw {
namespace math {
    /**
     * @brief Basic Function class.
     *
     * @todo
     * - Implement separable functions
     * - Implement deepCopy method
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
            std::string const name="Function")  ///< function name (for toString)
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _name(name)
        {}
        
        virtual ~Function() {};

        /**
         * @brief Return the number of function parameters
         */
        virtual unsigned int getNParameters() const = 0;
        
        /**
         * @brief Get one function parameter without range checking
         */
        virtual double getParameter(
            unsigned int ind)    ///< index of parameter
        const = 0;
        
        /**
         * @brief Return all function parameters
         */
        virtual std::vector<double> const getParameters() const = 0;
        
        /**
         * @brief Set one function parameter without range checking
         */
        virtual void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        = 0;

        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        virtual void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        = 0;

        
        /**
         * @brief Get the names of the parameters
         */
        virtual std::vector<std::string> getParameterNames() const;
    
        /**
         * @brief Return a string description
         */
        virtual std::string toString(void) const;

    protected:
        std::string _name;
    };



    /**
     * @brief A virtual Function taking one argument.
     *
     * Intended as a type specifier for variables
     * (as in foo takes a reference to a Function1).
     * See SimpleFunction1 for a class to do actual work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Function1 : public Function<ReturnT> {
    public:
        typedef boost::shared_ptr<Function1> PtrType;

        /**
         * @brief Construct a Function1.
         */
        explicit Function1(
            std::string const name="Function1") ///< function name (for toString)
        :
            Function<ReturnT>(name)
        {}

        virtual ~Function1() {};
    
        virtual ReturnT operator() (double x) const = 0;
    };


    /**
     * @brief A virtual Function taking two arguments.
     *
     * Intended as a type specifier for variables
     * (as in foo takes a reference to a Function2)
     * See SimpleFunction2 and SeparableFunction2 for classes to do actual work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class Function2 : public Function<ReturnT> {
    public:
        typedef boost::shared_ptr<Function2> PtrType;
        /**
         * @brief Construct a Function2.
         */
        explicit Function2(
            std::string const name="Function2") ///< function name (for toString)
        :
            Function<ReturnT>(name)
        {}

        virtual ~Function2() {};
    
        virtual ReturnT operator() (double x, double y) const = 0;
    };

    
    /**
     * @brief A Function taking one argument.
     *
     * Subclass and override operator() to do useful work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class SimpleFunction1 : public Function1<ReturnT> {
    public:
        /**
         * @brief Construct a Function1 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit SimpleFunction1(
            unsigned int nParams,   ///< number of function parameters
            std::string const name="SimpleFunction1")   ///< function name (for toString)
        :
            Function1<ReturnT>(name),
            _params(nParams)
        {}

        /**
         * @brief Construct a Function1 given the function parameters.
         */
        explicit SimpleFunction1(
            std::vector<double> const &params,   ///< function parameters
            std::string const name="SimpleFunction1")   ///< function name (for toString)
        :
            Function1<ReturnT>(name),
            _params(params)
        {}
        
        virtual ~SimpleFunction1() {};

        /**
         * @brief Return the number of function parameters
         */
        virtual unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * @brief Get one function parameter without range checking
         */
        virtual double getParameter(
            unsigned int ind)    ///< index of parameter
        const {
            return _params[ind];
        }
        
        /**
         * @brief Return all function parameters
         */
        virtual std::vector<double> const getParameters() const {
            return _params.getParameters();
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        virtual void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _params.setParameter(ind, newValue);
        };

        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        virtual void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        {
            _params.setParameters(params);
        };
    
    protected:
        SimpleParameters<ReturnT> _params;
    };

    
    /**
     * @brief A simple (nonseparable) Function taking two arguments.
     *
     * Subclass and override operator() to do useful work.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class SimpleFunction2 : public Function2<ReturnT> {
    public:
        /**
         * @brief Construct a Function1 given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit SimpleFunction2(
            unsigned int nParams,   ///< number of function parameters
            std::string const name="SimpleFunction2")   ///< function name (for toString)
        :
            Function2<ReturnT>(name),
            _params(nParams)
        {}

        /**
         * @brief Construct a Function1 given the function parameters.
         */
        explicit SimpleFunction2(
            std::vector<double> const &params,   ///< function parameters
            std::string const name="SimpleFunction2")   ///< function name (for toString)
        :
            Function2<ReturnT>(name),
            _params(params)
        {}
        
        virtual ~SimpleFunction2() {};

        /**
         * @brief Return the number of function parameters
         */
        virtual unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * @brief Get one function parameter without range checking
         */
        virtual double getParameter(
            unsigned int ind)    ///< index of parameter
        const {
            return _params[ind];
        }
        
        /**
         * @brief Return all function parameters
         */
        virtual std::vector<double> const getParameters() const {
            return _params.getParameters();
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        virtual void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _params.setParameter(ind, newValue);
        };

        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        virtual void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        {
            _params.setParameters(params);
        };
    protected:
        SimpleParameters<ReturnT> _params;
    };

    
    /**
     * @brief A SeparableFunction whose result is the product of two basis functions.
     *
     * Note: the basis functions must both be instances of Function1
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class SeparableFunction2 : public Function2<ReturnT>  {
    public:
        typedef typename Function1<ReturnT>::PtrType Function1PtrType;
        typedef std::vector<Function1PtrType> Function1ListType;
        explicit SeparableFunction2(
            Function1ListType functionList,  ///< list of Function1 basis functions
            std::string const name="SeparableFunction2")   ///< function name
        :
            Function2<ReturnT>(name),
            _params(functionList)
        {
            if (functionList.size() != 2) {
                throw lsst::pex::exceptions::InvalidParameter("Must supply exactly two functions");
            }
        }
        
        ReturnT operator() (double x, double y) const {
            return _params.solveFunction(0, x) * _params.solveFunction(1, y);
        }


        /**
         * @brief Return the number of function parameters
         */
        virtual unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * @brief Get one function parameter without range checking
         */
        virtual double getParameter(
            unsigned int ind)    ///< index of parameter
        const {
            return _params[ind];
        }
        
        /**
         * @brief Return all function parameters
         */
        virtual std::vector<double> const getParameters() const {
            return _params.getParameters();
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        virtual void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _params.setParameter(ind, newValue);
        };

        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        virtual void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        {
            _params.setParameters(params);
        };

    protected:
        SeparableParameters<ReturnT> _params;
    };

}}}   // lsst::afw::math

#endif // #ifndef LSST_AFW_MATH_FUNCTION_H
