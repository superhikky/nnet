#ifndef ACTFUNC_H
#define ACTFUNC_H

#include "help.h"
#include "neuron.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;

class ActivationFunction {
public:
    virtual double computeOutput(
        const double               &input, 
        vector<shared_ptr<Neuron>> *neurons) = 0;
    virtual double computeDifferentialOutput(
        const double &input, 
        vector<shared_ptr<Neuron>> *neurons) = 0;
};

class SigmoidFunction : public ActivationFunction {
public:
    virtual double computeOutput(
        const double               &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        return invert(1.0 + exp(-input));
    }
    
    virtual double computeDifferentialOutput(
        const double &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        double o = computeOutput(input, neurons);
        return o * negateRatio(o);
    }
};

class TanhFunction : public ActivationFunction {
public:
    virtual double computeOutput(
        const double               &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        return 0.5 * (1.0 + tanh(0.5 * input));
    }
    
    virtual double computeDifferentialOutput(
        const double &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        double t = tanh(0.5 * input);
        return 0.25 * negateRatio(t * t);
    }
};

class SoftmaxFunction : public ActivationFunction {
public:
    virtual double computeOutput(
        const double               &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        double inputExpsSum = 0.0;
        for (auto n : *neurons) 
        inputExpsSum += exp(n->getInput());
        return exp(input) / inputExpsSum;
    }
    
    virtual double computeDifferentialOutput(
        const double &input, 
        vector<shared_ptr<Neuron>> *neurons) override 
    {
        double o = computeOutput(input, neurons);
        return o * negateRatio(o);
    }
};

inline const map<string, shared_ptr<ActivationFunction>> *getActivationFunctions() {
    static const map<string, shared_ptr<ActivationFunction>> ACTIVATION_FUNCTIONS = {
        {"sigmoid", newInstance<SigmoidFunction>()}, 
        {"tanh",    newInstance<TanhFunction>()}, 
        {"softmax", newInstance<SoftmaxFunction>()}, 
    };
    return &ACTIVATION_FUNCTIONS;
}

#endif
