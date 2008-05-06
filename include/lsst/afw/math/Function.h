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
#include <utility> // for std::pair

#include <lsst/daf/data/LsstBase.h>
#include <lsst/pex/exceptions.h>

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
     * \todo
     * - Implement separable functions
     * - Implement deepCopy method
     *
     * \ingroup afw
     */
    template<typename ReturnT>
    class Function : public lsst::daf::data::LsstBase {
    
    public:
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
         * \brief Return the number of function parameters
         */
        unsigned int getNParameters() const {
            return _params.size();
        }
        
        /**
         * \brief Return the function parameters
         */
        std::vector<double> const getParameters() const {
            return _params;
        }
        
        /**
         * \brief Access the specified function parameter WITHOUT range checking
         *
         * Warning: no range checking is performed
         */
        double &operator[] (unsigned int ind) {
            return _params[ind];
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

    
    /**
     * \brief A Function whose result is the product of one or more basis functions.
     *
     * Subclass and override operator() to do useful work.
     *
     * Note that the internal _params vector is not used.
     *
     * \ingroup afw
     */
    template<typename ReturnT>
    class SeparableFunction : public Function<ReturnT> {
    public:
        typedef std::vector<boost::shared_ptr<Function1<ReturnT> > > functionListType;
        /**
         * \brief Construct a SeparableFunction from a list of basis functions
         */
        explicit SeparableFunction(
            functionListType &functionList)   ///< list of functions
        :
            Function<ReturnT>(0),
            _functionList(functionList)
        {
            unsigned int indexOffset = 0;
            for (typename functionListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const unsigned int nParams = (*funcIter)->getNParameters();
                for (unsigned int ii = 0; ii < nParams; ++ii) {
                    _funcIndexOffsetList.push_back(_functionIndexOffsetPairType(*funcIter, indexOffset));
                }
                indexOffset += nParams;
            }
        }
        
        virtual ~SeparableFunction() {};
    
        /**
         * \brief Return the basis functions.
         *
         * Warning: this is a shallow copy: if you modify the parameters of a returned basis function
         * it modifies the corresponding parameters of this SeparableFunction.
         */
        functionListType getFunctions() const {
            return _functionList;
        }
    
        /**
         * \brief Return the number of basis functions.
         *
         * Warning: this is a shallow copy: if you modify the parameters of a returned basis function
         * it modifies the corresponding parameters of this SeparableFunction.
         */
        unsigned int getNFunctions() const {
            return _functionList.size();
        }
    
        /**
         * \brief Return the number of function parameters
         */
        unsigned int getNParameters() const {
            return _funcIndexOffsetList.size();
        }
        
        /**
         * \brief Return the function parameters
         */
        std::vector<double> const getParameters() const {
            std::vector<double> params(this->getNParameters());
            std::vector<double>::iterator paramsIter = params.begin();
            for (typename functionListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const std::vector<double> funcParams = (*funcIter)->getParameters();
                paramsIter = std::copy(funcParams.begin(), funcParams.end(), paramsIter);
            }
            return params;
        }
        
        /**
         * \brief Access the specified function parameter WITHOUT range checking
         *
         * Warning: no range checking is performed
         */
        double &operator[] (unsigned int ind) {
            _functionIndexOffsetPairType funcOffsetPair = _funcIndexOffsetList[ind];
            return (*(funcOffsetPair.first))[ind - funcOffsetPair.second];
        }
        
        /**
         * \brief Set the function parameters
         *
         * \throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        void setParameters(std::vector<double> const &params) {
            if (params.size() != _funcIndexOffsetList.size()) {
                throw lsst::pex::exceptions::InvalidParameter("Wrong number of parameters");
            }
            
            std::vector<double>::const_iterator paramsIter = params.begin();
            for (typename functionListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const unsigned int nFuncParams = (*funcIter)->getNParameters();
                for (unsigned int ind = 0; ind < nFuncParams; ++ind) {
                    (**funcIter)[ind] = *paramsIter++;
                }
            }
        }

        virtual std::string toString(void) const {
            std::stringstream os;
            os << "SeparableFunction(";
            bool isFirst = true;
            for (typename functionListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                if (isFirst) {
                    isFirst = false;
                } else {
                    os << ",";
                }
                os << (*funcIter)->toString();
            }
            os << ")";
            return os.str();
        };

    protected:
        typedef std::pair<boost::shared_ptr<Function1<ReturnT> >, unsigned int> _functionIndexOffsetPairType;
        functionListType _functionList;
        std::vector<_functionIndexOffsetPairType> _funcIndexOffsetList; ///< parameter index -> function, index offset
    };
    
    /**
     * \brief A SeparableFunction whose result is the product of two basis functions.
     *
     * \ingroup afw
     */
    template<typename ReturnT>
    class SeparableFunction2 : public SeparableFunction<ReturnT> {
    public:
        typedef std::vector<boost::shared_ptr<Function1<ReturnT> > > functionListType;
        explicit SeparableFunction2(
            functionListType &functionList)   ///< list of functions
        :
            SeparableFunction<ReturnT>(functionList)
        {
            if (functionList.size() != 2) {
                throw lsst::pex::exceptions::InvalidParameter("Must supply exactly two functions");
            }
        }
        
        ReturnT operator() (double x, double y) const {
            return (*(this->_functionList[0]))(x) * (*(this->_functionList[1]))(y);
        }
    };

}}}   // lsst::afw::math

#endif // #ifndef LSST_AFW_MATH_FUNCTION_H
