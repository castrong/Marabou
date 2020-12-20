/*********************                                                        */
/*! \file CostFunctionManager.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "CostFunctionManager.h"
#include "Debug.h"
#include "FloatUtils.h"
#include "ITableau.h"
#include "MarabouError.h"
#include "TableauRow.h"

CostFunctionManager::CostFunctionManager( ITableau *tableau )
    : _tableau( tableau )
    , _costFunction( NULL )
    , _basicCosts( NULL )
    , _multipliers( NULL )
    , _n( 0 )
    , _m( 0 )
    , _costFunctionStatus( COST_FUNCTION_INVALID )
    , _ANColumn( NULL )
{
}

CostFunctionManager::~CostFunctionManager()
{
    freeMemoryIfNeeded();
}

void CostFunctionManager::freeMemoryIfNeeded()
{
    if ( _multipliers )
    {
        delete[] _multipliers;
        _multipliers = NULL;
    }

    if ( _basicCosts )
    {
        delete[] _basicCosts;
        _basicCosts = NULL;
    }

    if ( _costFunction )
    {
        delete[] _costFunction;
        _costFunction = NULL;
    }
}

void CostFunctionManager::initialize()
{
    _n = _tableau->getN();
    _m = _tableau->getM();

    freeMemoryIfNeeded();

    _costFunction = new double[_n - _m];
    if ( !_costFunction )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "CostFunctionManager::costFunction" );

    _basicCosts = new double[_m];
    if ( !_basicCosts )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "CostFunctionManager::basicCosts" );

    _multipliers = new double[_m];
    if ( !_multipliers )
        throw MarabouError( MarabouError::ALLOCATION_FAILED, "CostFunctionManager::multipliers" );

    invalidateCostFunction();
}

void CostFunctionManager::computeCostFunction( const Map<unsigned, double> &heuristicCost )
{
    /*
      A heuristic-based cost function is computed by computing the core
      cost function and adding to it the provided heuristic cost.

      The heuristic cost may include variables that are basic and variables
      that are non-basic. The basic variables are added to the vector of basic
      costs, which is normally used in computing the core cost fuction.
      Afterwards, once the modified core cost function has been computed,
      the remaining, non-basic variables are added.
    */

    // Reset cost function
    std::fill( _costFunction, _costFunction + _n - _m, 0.0 );

    // Compute the core basic costs
    computeBasicOOBCosts();

    // Iterate over the heuristic costs. Add any basic variables to the basic
    // cost vector, and the rest directly to the cost function.
    for ( const auto &variableCost : heuristicCost )
    {
        unsigned variable = variableCost.first;
        double cost = variableCost.second;
        unsigned variableIndex = _tableau->variableToIndex( variable );
        if ( _tableau->isBasic( variable ) )
            _basicCosts[variableIndex] += cost;
        else
            _costFunction[variableIndex] += cost;
    }

    // Complete the calculation of the modified core cost function
    computeMultipliers();
    computeReducedCosts();
}

void CostFunctionManager::computeCoreCostFunction()
{
    /*
      The core cost function is computed in three steps:

      1. Compute the basic costs c.
         These costs indicate whether a basic variable's row in
         the tableau should be added as is (variable too great;
         cost = 1), should be added negatively (variable is too
         small; cost = -1), or should be ignored (variable
         within bounds; cost = 0).

      2. Compute the multipliers p.
         p = c' * inv(B)
         This is solved by invoking BTRAN for pB = c'

      3. Compute the non-basic (reduced) costs.
         These are given by -p * AN

      Comment: the correctness follows from the fact that

      xB = inv(B)(b - AN xN)

      we ignore b because the constants don't matter for the cost
      function, and we omit xN because we want the function and not an
      evaluation thereof on a specific point.
    */

    // Once linear is solved, we won't need to check anymore since it will stay
    if( _optimize)
    {
      updateLinearSolved();
    }

    //printf("Linear solved: %d\n", _linearSolved);

    // fill the cost function with zeros
    std::fill( _costFunction, _costFunction + _n - _m, 0.0 );

    // Phase I - find a feasible solution
    if(!_linearSolved)
    {
      computeBasicOOBCosts();
      computeMultipliers();
      computeReducedCosts();
    }
    // Phase II - optimize our objective function
    else
    {
      //printf("Linear is solved, generating cost function to optimize\n");
      //printf("Opt var: %d\n", _optimizationVariable);
      //printf("Value of opt var: %f\n", _tableau->getValue(_optimizationVariable));
      //printf("Val via basic assignment: %f\n", _tableau->getBasicAssignment( _tableau->variableToIndex(_optimizationVariable) ));
      //printf("Our opt variable is basic: %d\n", _tableau->isBasic(_optimizationVariable));

      std::fill( _basicCosts, _basicCosts + _m, 0.0 );
      if (_tableau->isBasic(_optimizationVariable))
      {
          _basicCosts[_tableau->variableToIndex(_optimizationVariable)] = -1.0;
          computeMultipliers();
          computeReducedCosts();
      }
      // If non-basic, then it's just itself
      else
      {
          // _basicCosts should be all 0 since no basics are in the equation
          // The cost function is just given by the negative optimization variable
          // (assuming we are trying to maximize the original variable)
          _costFunction[_tableau->variableToIndex(_optimizationVariable)] = -1.0;
      }

    }

    _costFunctionStatus = ICostFunctionManager::COST_FUNCTION_JUST_COMPUTED;
}

void CostFunctionManager::updateLinearSolved()
{
      _linearSolved = !_tableau->existsBasicOutOfBounds();
}

void CostFunctionManager::adjustBasicCostAccuracy()
{
    if( _optimize)
    {
      updateLinearSolved();
    }

    // TODO (Chris Strong): Find what changes need to be made for this to work with optimization
    // FOR NOW JUST THROW THIS TO THE GENERAL COMPUTE IT ALL FUNCTION
    // fix this eventually? only need compute 1 thing so probably can just compute every time?

    bool needToRecompute = false;

    // Stage II just one possible basic variable contributing to the cost.
    if (_linearSolved && _optimize)
    {
      // If our optimization variable is basic set its coefficient
      unsigned optBasicIndex = _tableau->variableToIndex(_optimizationVariable);
      if (_tableau->isBasic(_optimizationVariable) && _basicCosts[optBasicIndex] != -1.0)
      {
          _basicCosts[optBasicIndex] = -1.0;
          needToRecompute = true;
      }
    }
    else
    {
        unsigned variable;
        double assignment, lb, relaxedLb, ub, relaxedUb;

        for ( unsigned i = 0; i < _m; ++i )
        {
            variable = _tableau->basicIndexToVariable( i );
            assignment = _tableau->getBasicAssignment( i );

            if ( _basicCosts[i] < 0 )
            {
                lb = _tableau->getLowerBound( variable );
                relaxedLb =
                    lb -
                    ( GlobalConfiguration::BASIC_COSTS_ADDITIVE_TOLERANCE +
                      GlobalConfiguration::BASIC_COSTS_MULTIPLICATIVE_TOLERANCE * FloatUtils::abs( lb ) );

                // Current basic cost is negative, assignment should be too small
                if ( assignment >= relaxedLb )
                {
                    needToRecompute = true;
                    _basicCosts[i] = 0;
                }
            }
            else if ( _basicCosts[i] > 0 )
            {
                ub = _tableau->getUpperBound( variable );
                relaxedUb =
                    ub +
                    ( GlobalConfiguration::BASIC_COSTS_ADDITIVE_TOLERANCE +
                      GlobalConfiguration::BASIC_COSTS_MULTIPLICATIVE_TOLERANCE * FloatUtils::abs( ub ) );

                // Current basic cost is positive, assignment should be too large
                if ( assignment <= relaxedUb )
                {
                    needToRecompute = true;
                    _basicCosts[i] = 0;
                }
            }
            else
            {
                /*
                  It seems to make sense to adjust anything that had cost 0 but should be
                  1 or -1, but apparently this leads to cycling.
                */
            }

        }
    }

    if ( needToRecompute )
    {
        computeMultipliers();
        computeReducedCosts();

        _costFunctionStatus = ICostFunctionManager::COST_FUNCTION_JUST_COMPUTED;
    }
}

void CostFunctionManager::computeBasicOOBCosts()
{
    unsigned variable;
    double assignment, lb, relaxedLb, ub, relaxedUb;
    for ( unsigned i = 0; i < _m; ++i )
    {
        variable = _tableau->basicIndexToVariable( i );
        assignment = _tableau->getBasicAssignment( i );

        lb = _tableau->getLowerBound( variable );
        relaxedLb =
            lb -
            ( GlobalConfiguration::BASIC_COSTS_ADDITIVE_TOLERANCE +
              GlobalConfiguration::BASIC_COSTS_MULTIPLICATIVE_TOLERANCE * FloatUtils::abs( lb ) );

        if ( assignment < relaxedLb )
        {
            _basicCosts[i] = -1;
            continue;
        }

        ub = _tableau->getUpperBound( variable );
        relaxedUb =
            ub +
            ( GlobalConfiguration::BASIC_COSTS_ADDITIVE_TOLERANCE +
              GlobalConfiguration::BASIC_COSTS_MULTIPLICATIVE_TOLERANCE * FloatUtils::abs( ub ) );

        if ( assignment > relaxedUb )
        {
            _basicCosts[i] = 1;
            continue;
        }

        _basicCosts[i] = 0;
    }
}

void CostFunctionManager::computeMultipliers()
{
    _tableau->backwardTransformation( _basicCosts, _multipliers );
}

void CostFunctionManager::computeReducedCosts()
{
    for ( unsigned i = 0; i < _n - _m; ++i )
        computeReducedCost( i );
}

void CostFunctionManager::computeReducedCost( unsigned nonBasic )
{
    unsigned nonBasicIndex = _tableau->nonBasicIndexToVariable( nonBasic );
    _ANColumn = _tableau->getSparseAColumn( nonBasicIndex );

    for ( const auto &entry : *_ANColumn )
        _costFunction[nonBasic] -= ( _multipliers[entry._index] * entry._value );
}

void CostFunctionManager::dumpCostFunction() const
{
    printf( "Cost function:\n\t" );

    for ( unsigned i = 0; i < _n - _m; ++i )
    {
        double coefficient = _costFunction[i];
        if ( FloatUtils::isZero( coefficient ) )
            continue;

        if ( FloatUtils::isPositive( coefficient ) )
            printf( "+" );
        printf( "%lfx%u ", coefficient, _tableau->nonBasicIndexToVariable( i ) );
    }

    printf( "\n" );
}

ICostFunctionManager::CostFunctionStatus CostFunctionManager::getCostFunctionStatus() const
{
    return _costFunctionStatus;
}

void CostFunctionManager::setCostFunctionStatus( ICostFunctionManager::CostFunctionStatus status )
{
    _costFunctionStatus = status;
}

const double *CostFunctionManager::getCostFunction() const
{
    return _costFunction;
}

void CostFunctionManager::setOptimize( bool optimize )
{
    _optimize = optimize;
}

void CostFunctionManager::setOptimizationVariable( unsigned variable )
{
    _optimizationVariable = variable;
}
bool CostFunctionManager::getOptimize()
{
    return _optimize;
}
unsigned CostFunctionManager::getOptimizationVariable()
{
    return _optimizationVariable;
}
bool CostFunctionManager::getLinearSolved()
{
    return _linearSolved;
}

double CostFunctionManager::updateCostFunctionForPivot( unsigned enteringVariableIndex,
                                                        unsigned leavingVariableIndex,
                                                        double pivotElement,
                                                        const TableauRow *pivotRow,
                                                        const double *changeColumn
                                                        )
{
      if( _optimize)
      {
        updateLinearSolved();
      }


      // TODO (Chris Strong): Find what changes need to be made for this to work with optimization
      // FOR NOW JUST THROW THIS TO THE GENERAL COMPUTE IT ALL FUNCTION
      // if (_linearSolved && _optimize)
      // {
      //   computeCoreCostFunction();
      //   return -1.0;
      // }


    /*
      This method is invoked when the non-basic _enteringVariable and
      basic _leaving variable are about to be swapped. It updates the
      reduced costs, without recomputing them from scratch.

      Suppose the pivot row is y = 5x + 3z, where x is entering and y
      is leaving. This corresponds to a new equation, x = 1/5y - 3/5z.
      If the previous cost for x was 10, then the new cost for y should
      be ( 10 / pivotElement ) = 10 / 5 = 2. This means that 2 units of y
      have the same cost as 10 units of x.

      However, now that we have 2 units of y, we have actually added 6 units
      of z to the overall cost - so the price of z needs to be adjusted by -6.

      Finally, we need to adjust the cost to reflect the pivot operation itself.
      The entering variable was, and remains, within bounds. The leaving variable
      is pressed against one of its bounds. So, if it was previously out-of-bounds
      (and contributed to the cost function), this needs to be removed.
    */

    ASSERT( _tableau->getM() == _m );
    ASSERT( _tableau->getN() == _n );

    /*
      The current reduced cost of the entering variable is stored in
      _costFunction, but since we have the change column we can compute a
      more accurate version from scratch
    */
    double enteringVariableCost = 0;
    for ( unsigned i = 0; i < _m; ++i )
        enteringVariableCost -= _basicCosts[i] * changeColumn[i];

    double normalizedError =
        FloatUtils::abs( enteringVariableCost - _costFunction[enteringVariableIndex] ) /
        ( FloatUtils::abs( enteringVariableCost ) + 1.0 );

    // Update the cost of the new non-basic
    _costFunction[enteringVariableIndex] = enteringVariableCost / pivotElement;

    for ( unsigned i = 0; i < _n - _m; ++i )
    {
        if ( i != enteringVariableIndex )
            _costFunction[i] -= (*pivotRow)[i] * _costFunction[enteringVariableIndex];
    }

    if (!(_linearSolved && _optimize))
    {
        _costFunction[enteringVariableIndex] -= _basicCosts[leavingVariableIndex];
    }

    // If we're in stage II, then we can have a basic cost without an out of bounds
    // variable, which means we won't need this extra update.
    if (_linearSolved && _optimize)
    {
        // If the entering variable is the optimization variable, set the basic cost to -1.0
        // The reduced cost should already have been updated appropriately
        if(_tableau->nonBasicIndexToVariable(enteringVariableIndex) == _optimizationVariable)
        {
            _basicCosts[leavingVariableIndex] = -1.0;
        }
        else
        {
            _basicCosts[leavingVariableIndex] = 0.0;
        }
    }
    else
    {
        /*
          The leaving variable might have contributed to the cost function, but it will
          soon be made within bounds. So, we adjust the reduced costs accordingly.
        */
        _costFunction[enteringVariableIndex] -= _basicCosts[leavingVariableIndex];
        // The entering varibale is non-basic, so it is within bounds.
        _basicCosts[leavingVariableIndex] = 0;
    }

    _costFunctionStatus = ICostFunctionManager::COST_FUNCTION_UPDATED;

    return normalizedError;
}

bool CostFunctionManager::costFunctionInvalid() const
{
    return _costFunctionStatus == ICostFunctionManager::COST_FUNCTION_INVALID;
}

void CostFunctionManager::invalidateCostFunction()
{
    _costFunctionStatus = ICostFunctionManager::COST_FUNCTION_INVALID;
}

bool CostFunctionManager::costFunctionJustComputed() const
{
    return _costFunctionStatus == ICostFunctionManager::COST_FUNCTION_JUST_COMPUTED;
}

double CostFunctionManager::getBasicCost( unsigned basicIndex ) const
{
    return _basicCosts[basicIndex];
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
