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
    double weight;
    double weightGradient;
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
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual double getBias() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void setBias(const double &bias) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual double getOutput() 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
    virtual void setOutput(const double &output) 
        { throw describe(__FILE__, "(", __LINE__, "): ", "不正な呼び出しです。"); }
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
    
    void read(istream &is) {
        double bias;
        is.read((char *)&bias, sizeof(double));
        if (is.gcount() < sizeof(double)) 
            throw describe(__FILE__, "(", __LINE__, "): " , "パラメータを読み込めません。");
        setBias(bias);
        for (auto s : *getInputSynapses()) {
            s->read(is);
        }
    }
    
    void write(ostream &os) {
        double b = getBias();
        os.write((char *)&b, sizeof(double));
        for (auto s : *getInputSynapses()) {
            s->write(os);
        }
    }
};

class InputNeuron : public Neuron {
protected:
    vector<shared_ptr<Synapse>> outputSynapses;
    double output;
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() override 
        { return &this->outputSynapses; }
    virtual double getOutput() override 
        { return this->output; }
    virtual void setOutput(const double &output) override 
        { this->output = output; }
};

class NotInputNeuron : public Neuron {
protected:
    vector<shared_ptr<Synapse>> inputSynapses;
    double bias;
    double output;
    double input;
    double error;
    double biasGradient;
    
    NotInputNeuron() : 
        bias(Random::getInstance()->normalDistribution<double>(0.0, 1.0)) {}
public:
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() override 
        { return &this->inputSynapses; }
    virtual double getBias() override 
        { return this->bias; }
    virtual void setBias(const double &bias) override 
        { this->bias = bias; }
    virtual double getOutput() override 
        { return this->output; }
    virtual void setOutput(const double &output) override 
        { this->output = output; }
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

class HiddenNeuron : public NotInputNeuron {
protected:
    vector<shared_ptr<Synapse>> outputSynapses;
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() override 
        { return &this->outputSynapses; }
};

#endif
