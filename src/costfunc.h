#ifndef COSTFUNC_H
#define COSTFUNC_H

#include "actfunc.h"
#include "neuron.h"
#include <cmath>
#include <map>
#include <memory>
#include <string>

using namespace std;

class CostFunction {
public:
    virtual double computeOutputNeuronCost(
        Neuron       *neuron, 
        const double &desiredOutput) = 0;
    
    virtual double computeOutputNeuronError(
        Neuron             *neuron, 
        ActivationFunction *activationFunction, 
        const double       &desiredOutput) = 0;
};

class QuadraticCostFunction : public CostFunction {
public:
    virtual double computeOutputNeuronCost(
        Neuron       *neuron, 
        const double &desiredOutput) override 
    {
        double error = neuron->getOutput() - desiredOutput;
        return 0.5 * error * error;
    }
    
    virtual double computeOutputNeuronError(
        Neuron             *neuron, 
        ActivationFunction *activationFunction, 
        const double       &desiredOutput) override 
    {
        return (neuron->getOutput() - desiredOutput) * 
            activationFunction->computeDifferentialOutput(neuron->getInput());
    }
};

class CrossEntropyCostFunction : public CostFunction {
public:
    virtual double computeOutputNeuronCost(
        Neuron       *neuron, 
        const double &desiredOutput) override 
    {
        return -(
            desiredOutput       * log(neuron->getOutput())     + 
            (1 - desiredOutput) * log(1 - neuron->getOutput())
        );
    }
    
    virtual double computeOutputNeuronError(
        Neuron             *neuron, 
        ActivationFunction *activationFunction, 
        const double       &desiredOutput) override 
    {
        return neuron->getOutput() - desiredOutput;
    }
};

inline const map<string, shared_ptr<CostFunction>> *getCostFunctions() {
    static const map<string, shared_ptr<CostFunction>> COST_FUNCTIONS = {
        {"quadratic",    newInstance<QuadraticCostFunction>()}, 
        {"crossEntropy", newInstance<CrossEntropyCostFunction>()}, 
    };
    return &COST_FUNCTIONS;
}

#endif
