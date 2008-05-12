// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_PARAMETERS_H
#define LSST_AFW_MATH_PARAMETERS_H
/**
 * @file
 *
 * @brief Define classes to manage parameters for Functions.
 *
 * Parameters emulate vectors for getting parameters
 * However Parameters emulate Functions for setting parameters.
 *
 * This was done for several reasons:
 * * Using [] and size makes the code in subclasses of Function compact and easy to read.
 *   It also improved backwards compatibility for code that was written when _params was a vector.
 * * However, using [] for set requires exposing references to internals (a bad idea)
 *   and it would be needlessly tricky to implement for SeparableParams.
 * * Emulating the Function calls for setting parameters makes the delegation code
 *   in Function a bit clearer and easier to write.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility> // for std::pair

#include <lsst/daf/data/LsstBase.h>
#include <lsst/pex/exceptions.h>

namespace lsst {
namespace afw {
namespace math {

    /**
     * @brief Manage a vector of Function parameters.
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class SimpleParameters : public lsst::daf::data::LsstBase {
    
    public:
        /**
         * @brief Construct a SimpleParameters given the number of function parameters.
         *
         * The function parameters are initialized to 0.
         */
        explicit SimpleParameters(
            unsigned int nParams)   ///< number of function parameters
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _params(nParams)
        {}

        /**
         * @brief Construct a SimpleParameters given the function parameters.
         */
        explicit SimpleParameters(
            std::vector<double> const &params)  ///< function parameters
        :
            lsst::daf::data::LsstBase(typeid(this)),
            _params(params)
        {}
        
        virtual ~SimpleParameters() {};
        
        /**
         * @brief Get all function parameters
         *
         * @return all parameters as a vector
         */
        std::vector<double> const getParameters() const {
            return _params;
        }
        
        /**
         * @brief Get one function parameter without range checking
         *
         * @return the value of the specified parameter
         */
        double operator[] (
            unsigned int ind)   ///< index of parameter
        const {
            return _params[ind];
        }
    
        /**
         * @brief Get the number of function parameters
         *
         * @return the number of function parameters
         */
        unsigned int size() const {
            return _params.size();
        }
        
        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        {
            if (_params.size() != params.size()) {
                throw lsst::pex::exceptions::InvalidParameter("Wrong number of parameters");
            }
            _params = params;
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _params[ind] = newValue;
        }

    protected:
        std::vector<double> _params;
    };

    template<typename ReturnT>
    class Function1;
    
    /**
     * @brief Manage parameters for separable functions
     *
     * @ingroup afw
     */
    template<typename ReturnT>
    class SeparableParameters: public lsst::daf::data::LsstBase {
    public:
        typedef typename Function1<ReturnT>::PtrType Function1PtrType;
        typedef std::vector<Function1PtrType> Function1ListType;
        /**
         * @brief Construct a SeparableParameters from a list of basis functions
         */
        explicit SeparableParameters(
            Function1ListType const &functionList    ///< list of Function1 basis functions
        ) :
            lsst::daf::data::LsstBase(typeid(this)),
            _functionList(functionList)
        
        {
            unsigned int indexOffset = 0;
            for (typename Function1ListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const unsigned int nParams = (*funcIter)->getNParameters();
                for (unsigned int ii = 0; ii < nParams; ++ii) {
                    _funcIndexOffsetList.push_back(_FunctionIndexOffsetType(*funcIter, indexOffset));
                }
                indexOffset += nParams;
            }
        }
        
        virtual ~SeparableParameters() {};
        
        /**
         * @brief Return all function parameters
         *
         * @return all parameters as a vector
         */
        std::vector<double> const getParameters() const {
            std::vector<double> params(this->size());
            std::vector<double>::iterator paramsIter = params.begin();
            for (typename Function1ListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const std::vector<double> funcParams = (*funcIter)->getParameters();
                paramsIter = std::copy(funcParams.begin(), funcParams.end(), paramsIter);
            }
            return params;
        }
        
        /**
         * @brief Get one function parameter without range checking
         *
         * @return the value of the specified parameter
         */
        double operator[] (
            unsigned int ind)   ///< index of parameter
        const {
            _FunctionIndexOffsetType funcOffsetPair = _funcIndexOffsetList[ind];
            return funcOffsetPair.first->getParameter(ind - funcOffsetPair.second);
        }
    
        /**
         * @brief Return the number of function parameters
         *
         * @return the number of function parameters
         */
        unsigned int size() const {
            return _funcIndexOffsetList.size();
        }

        /**
         * @brief Set all function parameters
         *
         * @throw lsst::pex::exceptions::InvalidParameter if the wrong number of parameters is supplied.
         */
        void setParameters(
            std::vector<double> const &params)  ///< new function parameters
        {
            if (params.size() != _funcIndexOffsetList.size()) {
                throw lsst::pex::exceptions::InvalidParameter("Wrong number of parameters");
            }
            
            std::vector<double>::const_iterator paramsIter = params.begin();
            for (typename Function1ListType::const_iterator funcIter = _functionList.begin();
                 funcIter != _functionList.end();  ++funcIter) {
                const unsigned int nFuncParams = (*funcIter)->getNParameters();
                for (unsigned int ind = 0; ind < nFuncParams; ++ind) {
                    (*funcIter)->setParameter(ind, *paramsIter++);
                }
            }
        }
        
        /**
         * @brief Set one function parameter without range checking
         */
        void setParameter(
            unsigned int ind,   ///< index of parameter
            double newValue)    ///< new value for parameter
        {
            _FunctionIndexOffsetType funcOffsetPair = _funcIndexOffsetList[ind];
            return funcOffsetPair.first->setParameter(ind - funcOffsetPair.second, newValue);
        }

    
        /**
         * @brief Get the basis functions
         *
         * Warning: returns the original basis functions, not copies.
         * If you modify the parameters of a returned basis function it also modifies
         * the contained basis function.
         *
         * @return the basis functions: a vector of shared_ptr to Function1
         */
        Function1ListType getFunctions() const {
            return _functionList;
        }
    
        /**
         * @brief Return the number of basis functions.
         */
        unsigned int getNFunctions() const {
            return _functionList.size();
        }
            
        /**
         * @brief Solve the specified function
         *
         * @return the value of the specified function at the specified point
         *
         */
        ReturnT solveFunction(
            unsigned int ind,   ///< index of function
            double x            ///< argument for function
        ) const {
            return (*_functionList[ind])(x);
        }

    protected:
        Function1ListType _functionList;
        typedef std::pair<boost::shared_ptr<Function1<ReturnT> >, unsigned int> _FunctionIndexOffsetType;
        std::vector<_FunctionIndexOffsetType> _funcIndexOffsetList; ///< parameter index -> function, index offset
    };

}}}   // lsst::afw::math

#endif // #ifndef LSST_AFW_MATH_PARAMETERS_H
