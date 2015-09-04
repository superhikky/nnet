#ifndef REGRIZ_H
#define REGRIZ_H

#include "layer.h"
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;

class Regularization {
public:
    virtual double computeWeightsCost(
        vector<shared_ptr<Layer>> *layers, 
        const double              &weightDecayRate, 
        const double              &dropoutRatio) = 0;
    virtual double computeDecayedWeight(
        const double &weight, 
        const double &learningRate, 
        const double &weightDecayRate, 
        const size_t &imagesNumber) = 0;
};

class NullRegularization : public Regularization {
public:
    virtual double computeWeightsCost(
        vector<shared_ptr<Layer>> *layers, 
        const double              &weightDecayRate, 
        const double              &dropoutRatio) override
        { return 0.0; }
    virtual double computeDecayedWeight(
        const double &weight, 
        const double &learningRate, 
        const double &weightDecayRate, 
        const size_t &imagesNumber) override 
        { return weight; }
};

class L1Regularization : public Regularization {
public:
    virtual double computeWeightsCost(
        vector<shared_ptr<Layer>> *layers, 
        const double              &weightDecayRate, 
        const double              &dropoutRatio) override
    {
        double absoluteWeightsSum = 0.0;
        for (auto l = layers->begin() + 1; l != layers->end(); l++) {
            for (auto n : (*(*l)->getNeurons())) {
                for (auto s : *n->getInputSynapses()) {
                    double actualWeight = s->getWeight() * (1.0 - dropoutRatio);
                    absoluteWeightsSum += fabs(actualWeight);
                }
            }
        }
        return weightDecayRate * absoluteWeightsSum;
    }
    
    virtual double computeDecayedWeight(
        const double &weight, 
        const double &learningRate, 
        const double &weightDecayRate, 
        const size_t &imagesNumber) override 
    {
        double weightSign = 0.0;
        if (weight > 0.0) 
            weightSign = +1.0;
        else if (weight < 0.0) 
            weightSign = -1.0;
        return weight - 
            weightSign * 
            learningRate * 
            weightDecayRate / 
            (double)imagesNumber;
    }
};

class L2Regularization : public Regularization {
public:
    virtual double computeWeightsCost(
        vector<shared_ptr<Layer>> *layers, 
        const double              &weightDecayRate, 
        const double              &dropoutRatio) override
    {
        double squaredWeightsSum = 0.0;
        for (auto l = layers->begin() + 1; l != layers->end(); l++) {
            for (auto n : (*(*l)->getNeurons())) {
                for (auto s : *n->getInputSynapses()) {
                    double actualWeight = s->getWeight() * (1.0 - dropoutRatio);
                    squaredWeightsSum += actualWeight * actualWeight;
                }
            }
        }
        return (weightDecayRate / 2.0) * squaredWeightsSum;
    }
    
    virtual double computeDecayedWeight(
        const double &weight, 
        const double &learningRate, 
        const double &weightDecayRate, 
        const size_t &imagesNumber) override 
    {
        return (
            1.0 - (
                learningRate * 
                weightDecayRate / 
                (double)imagesNumber
            )
        ) * weight;
    }
};

inline const map<string, shared_ptr<Regularization>> *getRegularizations() {
    static const map<string, shared_ptr<Regularization>> REGULARIZATIONS = {
        {"null", newInstance<NullRegularization>()}, 
        {"l1",   newInstance<L1Regularization>()}, 
        {"l2",   newInstance<L2Regularization>()}, 
    };
    return &REGULARIZATIONS;
}

#endif
