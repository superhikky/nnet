#ifndef ACTFUNC_H
#define ACTFUNC_H

#include "help.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>

using namespace std;

class ActivationFunction {
public:
    virtual double computeOutput(const double &input) = 0;
    virtual double computeDifferentialOutput(const double &input) = 0;
};

class SigmoidFunction : public ActivationFunction {
public:
    virtual double computeOutput(const double &input) override {
        return invert(1.0 + exp(-input));
    }
    
    virtual double computeDifferentialOutput(const double &input) override {
        double o = computeOutput(input);
        return o * negateRatio(o);
    }
};

class TanhFunction : public ActivationFunction {
public:
    virtual double computeOutput(const double &input) override {
        return 0.5 * (1.0 + tanh(0.5 * input));
    }
    
    virtual double computeDifferentialOutput(const double &input) override {
        double t = tanh(0.5 * input);
        return 0.25 * negateRatio(t * t);
    }
};

inline const map<string, shared_ptr<ActivationFunction>> *getActivationFunctions() {
    static const map<string, shared_ptr<ActivationFunction>> ACTIVATION_FUNCTIONS = {
        {"sigmoid", newInstance<SigmoidFunction>()}, 
        {"tanh",    newInstance<TanhFunction>()}, 
    };
    return &ACTIVATION_FUNCTIONS;
}

#endif
