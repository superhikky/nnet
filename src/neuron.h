#ifndef NNET_H
#define NNET_H

#include "help.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class Neuron;

class Synapse {
protected:
    Neuron *source;
    Neuron *destination;
    double  weight;
    double  weightGradient;
public:
    Synapse(Neuron *source, Neuron *destination) : 
        source(source), 
        destination(destination), 
        weight(Random::getInstance()->normalDistribution<double>(0.0, 1.0)) {}
    Neuron *getSource() 
        { return this->source; }
    Neuron *getDestination() 
        { return this->destination; }
    double getWeight() 
        { return this->weight; }
    void setWeight(const double &weight) 
        { this->weight = weight; }
    void multiplyWeight(const double &multiplier) 
        { this->weight *= multiplier; }
    double getWeightGradient() 
        { return this->weightGradient; }
    void clearWeightGradient() 
        { this->weightGradient = 0.0; }
    void addWeightGradient(const double &addend) 
        { this->weightGradient += addend; }
    
    void read(istream &is) {
        is.read((char *)&this->weight, sizeof(double));
        if (is.gcount() < sizeof(double)) 
            throw describe(__FILE__, "(", __LINE__, "): " , "パラメータを読み込めません。");
    }
    
    void write(ostream &os) {
        os.write((char *)&this->weight, sizeof(double));
    }
};

class Neuron {
protected:
    double output;
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual double getBias() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void setBias(const double &bias) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    double getOutput()  
        { return this->output; }
    void setOutput(const double &output)  
        { this->output = output; }
    virtual double getInput() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void clearInput() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void addInput(const double &addend) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual double getError() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void setError(const double &error) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual double getBiasGradient() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void clearBiasGradient() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void addBiasGradient(const double &addend) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual bool wasDropped() 
        { return false; }
    virtual void drop() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void restore() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    
    void read(istream &is) {
        double bias;
        is.read((char *)&bias, sizeof(double));
        if (is.gcount() < sizeof(double)) 
            throw describe(__FILE__, "(", __LINE__, "): " , "パラメータを読み込めません。");
        setBias(bias);
        for (auto inputSynapse : *getInputSynapses()) 
            inputSynapse->read(is);
    }
    
    void write(ostream &os) {
        double b = getBias();
        os.write((char *)&b, sizeof(double));
        for (auto inputSynapse : *getInputSynapses()) 
            inputSynapse->write(os);
    }
};

class NotOutputNeuron : public virtual Neuron {
protected:
    vector<shared_ptr<Synapse>> outputSynapses;
    bool dropped;
    
    NotOutputNeuron() : dropped(false) {}
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() override 
        { return &this->outputSynapses; }
    virtual bool wasDropped() override 
        { return this->dropped; }
    virtual void drop() override 
        { this->dropped = true; }
    virtual void restore() override 
        { this->dropped = false; }
};

class InputNeuron : public NotOutputNeuron {};

class NotInputNeuron : public virtual Neuron {
protected:
    vector<shared_ptr<Synapse>> inputSynapses;
    double                      bias;
    double                      input;
    double                      error;
    double                      biasGradient;
    
    NotInputNeuron() : 
        bias(Random::getInstance()->normalDistribution<double>(0.0, 1.0)) {}
public:
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() override 
        { return &this->inputSynapses; }
    virtual double getBias() override 
        { return this->bias; }
    virtual void setBias(const double &bias) override 
        { this->bias = bias; }
    virtual double getInput() override 
        { return this->input; }
    virtual void clearInput() override 
        { this->input = 0.0; }
    virtual void addInput(const double &addend) override 
        { this->input += addend; }
    virtual double getError() override 
        { return this->error; }
    virtual void setError(const double &error) override 
        { this->error = error; }
    virtual double getBiasGradient() override 
        { return this->biasGradient; }
    virtual void clearBiasGradient() override 
        { this->biasGradient = 0.0; }
    virtual void addBiasGradient(const double &addend) override 
        { this->biasGradient += addend; }
};

class OutputNeuron : public NotInputNeuron {};
class HiddenNeuron : public NotOutputNeuron, public NotInputNeuron {};

#endif
