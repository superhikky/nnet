#ifndef ACTFUNC_H
#define ACTFUNC_H

#include "help.h"
#include <cmath>

using namespace std;

class ActivationFunction {
public:
    virtual double computeOutput(const double &input) = 0;
    virtual double computeDifferentialOutput(const double &input) = 0;
};

class SigmoidActivationFunction : public ActivationFunction {
public:
    virtual double computeOutput(const double &input) override {
        return invert(1.0 + exp(-input));
    }
    
    virtual double computeDifferentialOutput(const double &input) override {
        double o = computeOutput(input);
        return o * negateRatio(o);
    }
};

#endif
