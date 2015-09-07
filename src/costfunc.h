#ifndef COSTFUNC_H
#define COSTFUNC_H

#include "actfunc.h"
#include "help.h"
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
        const double       &desiredOutput, 
        ActivationFunction *activationFunction) = 0;
};

class QuadraticFunction : public CostFunction {
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
        const double       &desiredOutput, 
        ActivationFunction *activationFunction) override 
    {
        return (neuron->getOutput() - desiredOutput) * 
            activationFunction->computeDifferentialOutput(neuron->getInput());
    }
};

class CrossEntropyFunction : public CostFunction {
public:
    virtual double computeOutputNeuronCost(
        Neuron       *neuron, 
        const double &desiredOutput) override 
    {
        return -(
            desiredOutput              * log(neuron->getOutput())              + 
            negateRatio(desiredOutput) * log(negateRatio(neuron->getOutput()))
        );
    }
    
    virtual double computeOutputNeuronError(
        Neuron             *neuron, 
        const double       &desiredOutput, 
        ActivationFunction *activationFunction) override 
    {
        return neuron->getOutput() - desiredOutput;
    }
};

inline const map<string, shared_ptr<CostFunction>> *getCostFunctions() {
    static const map<string, shared_ptr<CostFunction>> COST_FUNCTIONS = {
        {"quadratic",    newInstance<QuadraticFunction>()}, 
        {"crossEntropy", newInstance<CrossEntropyFunction>()}, 
    };
    return &COST_FUNCTIONS;
}

#endif
