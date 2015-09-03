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
            throw describe(__FILE__, "(", __LINE__, "): " , "�p�����[�^��ǂݍ��߂܂���B");
    }
    
    void write(ostream &os) {
        os.write((char *)&this->weight, sizeof(double));
    }
};

class Neuron {
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() = 0;
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() = 0;
    virtual double getBias() = 0;
    virtual void setBias(const double &bias) = 0;
    virtual double getOutput() = 0;
    virtual void setOutput(const double &output) = 0;
    virtual double getInput() = 0;
    virtual void clearInput() = 0;
    virtual void addInput(const double &addend) = 0;
    virtual double getError() = 0;
    virtual void setError(const double &error) = 0;
    virtual double getBiasGradient() = 0;
    virtual void clearBiasGradient() = 0;
    virtual void addBiasGradient(const double &addend) = 0;
    
    void read(istream &is) {
        double bias;
        is.read((char *)&bias, sizeof(double));
        if (is.gcount() < sizeof(double)) 
            throw describe(__FILE__, "(", __LINE__, "): " , "�p�����[�^��ǂݍ��߂܂���B");
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
    virtual vector<shared_ptr<Synapse>> *getInputSynapses() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual double getBias() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void setBias(const double &bias) override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual double getOutput() override 
        { return this->output; }
    virtual void setOutput(const double &output) override 
        { this->output = output; }
    virtual double getInput() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void clearInput() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void addInput(const double &addend) override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual double getError() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void setError(const double &error) override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual double getBiasGradient() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void clearBiasGradient() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
    virtual void addBiasGradient(const double &addend) override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
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

class OutputNeuron : public NotInputNeuron {
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() override 
        { throw describe(__FILE__, "(", __LINE__, "): ", "�s���ȌĂяo���ł��B"); }
};

class HiddenNeuron : public NotInputNeuron {
protected:
    vector<shared_ptr<Synapse>> outputSynapses;
public:
    virtual vector<shared_ptr<Synapse>> *getOutputSynapses() override 
        { return &this->outputSynapses; }
};

#endif
