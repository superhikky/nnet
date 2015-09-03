#ifndef LAYER_H
#define LAYER_H

#include "actfunc.h"
#include "help.h"
#include "mnist.h"
#include "neuron.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class Layer {
protected:
    vector<shared_ptr<Neuron>> neurons;
public:
    vector<shared_ptr<Neuron>> *getNeurons()  
        { return &this->neurons; }
    virtual ActivationFunction *getActivationFunction() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void connect(Layer *sourceLayer) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    
    void read(istream &is) {
        for (auto n : this->neurons) 
            n->read(is);
    }
    
    void write(ostream &os) {
        for (auto n : this->neurons) 
            n->write(os);
    }
};

class InputLayer : public virtual Layer {
public:
    InputLayer() {
        for (auto i = 0; i < IMAGE_AREA; i++) 
            this->neurons.push_back(newInstance<InputNeuron>());
    }
};

class NotInputLayer : public virtual Layer {
protected:
    shared_ptr<ActivationFunction> activationFunction;
    
    NotInputLayer(const shared_ptr<ActivationFunction> &activationFunction) : 
        activationFunction(activationFunction) {}
public:
    virtual ActivationFunction *getActivationFunction() override 
        { return this->activationFunction.get(); }
};

class FullyConnectedLayer : public virtual Layer {
public:
    virtual void connect(Layer *sourceLayer) override {
        for (auto src : *sourceLayer->getNeurons()) {
            for (auto dest : this->neurons) {
                auto s = newInstance<Synapse>(src.get(), dest.get());
                src->getOutputSynapses()->push_back(s);
                dest->getInputSynapses()->push_back(s);
            }
        }
    }
};

class OutputLayer : public NotInputLayer, public FullyConnectedLayer {
public:
    OutputLayer(const shared_ptr<ActivationFunction> &activationFunction) : 
        NotInputLayer(activationFunction) 
    {
        for (auto i = 0; i < LABEL_VALUES_NUMBER; i++) 
            this->neurons.push_back(newInstance<OutputNeuron>());
    }
};

class HiddenLayer : public NotInputLayer {
protected:
    HiddenLayer(
        const size_t                         &neuronsNumber, 
        const shared_ptr<ActivationFunction> &activationFunction) : 
        NotInputLayer(activationFunction) 
    {
        for (auto i = 0; i < neuronsNumber; i++) 
            this->neurons.push_back(newInstance<HiddenNeuron>());
    }
};

class FullyConnectedHiddenLayer : public HiddenLayer, public FullyConnectedLayer {
public:
    FullyConnectedHiddenLayer(
        const size_t                         &neuronsNumber, 
        const shared_ptr<ActivationFunction> &activationFunction) : 
        HiddenLayer(neuronsNumber, activationFunction) {}
};

#endif
