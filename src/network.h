#ifndef NETWORK_H
#define NETWORK_H

#include "costfunc.h"
#include "help.h"
#include "layer.h"
#include "mnist.h"
#include "neuron.h"
#include "regriz.h"
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

using namespace std;

struct HyperParameters {
    CostFunction   *costFunction;
    Regularization *regularization;
    double          weightDecayRate;
    double          dropoutRatio;
    double          learningRate;
};

struct Log {
    function<void(
        size_t epochIndex, 
        size_t trainCorrectAnswersNumber, 
        double trainCost, 
        size_t evalCorrectAnswersNumber, 
        double evalCost)> doneTrainEpoch;
    function<void(
        size_t trainCorrectAnswersNumber, 
        double trainCostsAverage, 
        size_t evalCorrectAnswersNumber, 
        double evalCostsAverage)> doneTrain;
    function<void(
        size_t inferIndex, 
        size_t imageIndex, 
        size_t label, 
        size_t answer)> doneInferImage;
    function<void(
        size_t correctAnswersNumber, 
        double cost)> doneInfer;
    
    Log() : 
        doneTrainEpoch([](size_t, size_t, double, size_t, double) {}), 
        doneTrain([](size_t, double, size_t, double) {}), 
        doneInferImage([](size_t, size_t, size_t, size_t) {}), 
        doneInfer([](size_t, double) {}) {}
};

class Network {
protected:
    shared_ptr<vector<shared_ptr<Layer>>>  layers;
    HyperParameters                       *hyperParameters;
    shared_ptr<Log>                        log;
    
    void beginBatch() {
        for (auto l = this->layers->begin() + 1; l < this->layers->end() - 1; l++) 
            (*l)->dropNeurons(this->hyperParameters->dropoutRatio);
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->clearBiasGradient();
                for (auto s : *n->getInputSynapses()) 
                    s->clearWeightGradient();
            }
        }
    }
    
    void feedForward(Image *image, const double &outputRatio) {
        auto inputNeurons = this->layers->front()->getNeurons();
        for (auto i = 0; i < IMAGE_AREA; i++) 
            (*inputNeurons)[i]->setOutput((double)(*image->getIntensities())[i] / 255.0);
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            for (auto n : *(*l)->getNeurons()) {
                if (n->wasDropped()) 
                    continue;
                n->clearInput();
                for (auto s : *n->getInputSynapses()) {
                    auto src = s->getSource();
                    if (src->wasDropped()) 
                        continue;
                    n->addInput(s->getWeight() * src->getOutput());
                }
                n->addInput(n->getBias());
                n->setOutput(
                    (*l)->getActivationFunction()->computeOutput(n->getInput()) * 
                    outputRatio);
            }
        }
    }
    
    size_t getAnswer() {
        size_t answer = 0;
        double maxOutput = 0.0;
        auto outputNeurons = this->layers->back()->getNeurons();
        for (auto i = 0; i < outputNeurons->size(); i++) {
            double o = (*outputNeurons)[i]->getOutput();
            if (o > maxOutput) {
                answer = i;
                maxOutput = o;
            }
        }
        return answer;
    }
    
    double computeImageCost(const size_t &label) {
        double cost = 0.0;
        auto outputNeurons = this->layers->back()->getNeurons();
        for (auto i = 0; i < outputNeurons->size(); i++) 
            cost += this->hyperParameters->costFunction->computeOutputNeuronCost(
                (*outputNeurons)[i].get(), 
                getDesiredOutput(i, label));
        return cost;
    }
    
    void propagateBackward(const size_t &label) {
        for (auto l = this->layers->rbegin(); l != this->layers->rend() - 1; l++) {
            if (l == this->layers->rbegin()) {
                for (auto i = 0; i < (*l)->getNeurons()->size(); i++) {
                    auto n = (*(*l)->getNeurons())[i];
                    n->setError(this->hyperParameters->costFunction->computeOutputNeuronError(
                        n.get(), 
                        (*l)->getActivationFunction(), 
                        getDesiredOutput(i, label)));
                }
            } else {
                for (auto n : (*(*l)->getNeurons())) {
                    if (n->wasDropped()) 
                        continue;
                    double error = 0.0;
                    for (auto s : *n->getOutputSynapses()) {
                        auto dest = s->getDestination();
                        if (dest->wasDropped()) 
                            continue;
                        error += s->getWeight() * dest->getError();
                    }
                    error *= (*l)->getActivationFunction()->computeDifferentialOutput(n->getInput());
                    n->setError(error);
                }
            }
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->addBiasGradient(n->getError());
                for (auto s : *n->getInputSynapses()) {
                    auto src = s->getSource();
                    if (src->wasDropped()) 
                        continue;
                    s->addWeightGradient(src->getOutput() * n->getError());
                }
            }
        }
    }
    
    void endBatch(const size_t &imagesNumber, const size_t &batchSize) {
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->setBias(
                    n->getBias() - 
                    this->hyperParameters->learningRate * n->getBiasGradient() / batchSize);
                for (auto s : *n->getInputSynapses()) 
                    s->setWeight(
                        this->hyperParameters->regularization->computeDecayedWeight(
                            s->getWeight(), 
                            this->hyperParameters->learningRate, 
                            this->hyperParameters->weightDecayRate, 
                            imagesNumber) - 
                        this->hyperParameters->learningRate * s->getWeightGradient() / batchSize);
            }
        }
        for (auto l = this->layers->begin() + 1; l < this->layers->end() - 1; l++) 
            (*l)->restoreNeurons();
    }
    
    static double getDesiredOutput(const size_t &index, const size_t &label) {
        return index == label ? 1.0 : 0.0;
    }
public:
    Network(
        const shared_ptr<vector<shared_ptr<Layer>>> &layers, 
        HyperParameters                             *hyperParameters, 
        const shared_ptr<Log>                       &log) : 
            layers         (layers), 
            hyperParameters(hyperParameters), 
            log            (log) 
    {
        for (auto i = 1; i < this->layers->size(); i++) 
            (*this->layers)[i]->connect((*this->layers)[i - 1].get());
    }
    
    void train(
        const size_t   &epochsNumber, 
        const size_t   &batchSize, 
        MNIST          *trainingMNIST, 
        const size_t   &trainImagesOffset, 
        const size_t   &trainImagesNumber, 
        MNIST          *evalMNIST, 
        const size_t   &evalImagesOffset, 
        const size_t   &evalImagesNumber) 
    {
        size_t trainCorrectAnswersNumber = 0;
        double trainCostsSum = 0.0;
        size_t evalCorrectAnswersNumber = 0;
        double evalCostsSum = 0.0;
        vector<size_t> imageIndices(trainImagesNumber);
        for (auto i = 0; i < epochsNumber; i++) {
            size_t epochTrainCorrectAnswersNumber = 0;
            double trainCost = 0.0;
            for (auto j = 0; j < trainImagesNumber; j++) 
                imageIndices[j] = trainImagesOffset + j;
            for (auto j = 0;; j++) {
                if (j % batchSize == 0 || j == trainImagesNumber) {
                    if (j != 0) 
                        endBatch(trainImagesNumber, batchSize);
                    if (j == trainImagesNumber) 
                        break;
                    beginBatch();
                }
                size_t k = Random::getInstance()->uniformDistribution<size_t>(
                    0, trainImagesNumber - j - 1);
                size_t imageIndex = imageIndices[k];
                feedForward((*trainingMNIST)[imageIndex].get(), 1.0);
                size_t label = (*trainingMNIST)[imageIndex]->getLabel();
                if (getAnswer() == label) 
                    epochTrainCorrectAnswersNumber++;
                trainCost += computeImageCost(label);
                propagateBackward(label);
                imageIndices[k] = imageIndices[trainImagesNumber - j - 1];
            }
            trainCost += this->hyperParameters->regularization->computeWeightsCost(
                this->layers.get(), 
                this->hyperParameters->weightDecayRate, 
                trainImagesNumber, 
                this->hyperParameters->dropoutRatio);
            
            size_t epochEvalCorrectAnswersNumber = 0;
            double evalCost = 0.0;
            for (auto j = 0; j < evalImagesNumber; j++) {
                feedForward(
                    (*evalMNIST)[j].get(), 
                    1.0 - this->hyperParameters->dropoutRatio);
                size_t label = (*evalMNIST)[j]->getLabel();
                if (getAnswer() == label) 
                    epochEvalCorrectAnswersNumber++;
                evalCost += computeImageCost(label);
            }
            evalCost += this->hyperParameters->regularization->computeWeightsCost(
                this->layers.get(), 
                this->hyperParameters->weightDecayRate, 
                evalImagesNumber, 
                this->hyperParameters->dropoutRatio);
            
            trainCorrectAnswersNumber += epochTrainCorrectAnswersNumber;
            trainCostsSum += trainCost;
            evalCorrectAnswersNumber += epochEvalCorrectAnswersNumber;
            evalCostsSum += evalCost;
            this->log->doneTrainEpoch(
                i, 
                epochTrainCorrectAnswersNumber, 
                trainCost / (double)trainImagesNumber, 
                epochEvalCorrectAnswersNumber, 
                evalCost  / (double)evalImagesNumber);
        }
        this->log->doneTrain(
            trainCorrectAnswersNumber, 
            trainCostsSum / (double)epochsNumber / (double)trainImagesNumber, 
            evalCorrectAnswersNumber, 
            evalCostsSum  / (double)epochsNumber / (double)evalImagesNumber);
    }
    
    void infer(
        MNIST        *mnist, 
        const size_t &imagesOffset, 
        const size_t &imagesNumber)
    {
        size_t correctAnswersNumber = 0;
        double cost = 0.0;
        for (auto i = 0; i < imagesNumber; i++) {
            size_t imageIndex = imagesOffset + i;
            feedForward(
                (*mnist)[i].get(), 
                1.0 - this->hyperParameters->dropoutRatio);
            size_t label = (*mnist)[i]->getLabel();
            if (getAnswer() == label) 
                correctAnswersNumber++;
            cost += computeImageCost(label);
            this->log->doneInferImage(
                i, 
                imageIndex, 
                label, 
                getAnswer());
        }
        cost += this->hyperParameters->regularization->computeWeightsCost(
            this->layers.get(), 
            this->hyperParameters->weightDecayRate, 
            imagesNumber, 
            this->hyperParameters->dropoutRatio);
        this->log->doneInfer(
            correctAnswersNumber, 
            cost / (double)imagesNumber);
    }
    
    void read(istream &is) {
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) 
            (*l)->read(is);
    }
    
    void write(ostream &os) {
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) 
            (*l)->write(os);
    }
};

class NetworkBuilder {
protected:
    NetworkBuilder() = default;
    
    using MakeLayerProc = function<shared_ptr<Layer>(vector<string> *)>;
    static map<string, MakeLayerProc> *getMakeLayerProcs() {
        static map<string, MakeLayerProc> MAKE_LAYER_PROCS = {
            {"input",          &makeInputLayer}, 
            {"fullyConnected", &makeFullyConnectedHiddenLayer}, 
            {"output",         &makeOutputLayer}, 
        };
        return &MAKE_LAYER_PROCS;
    }
    
    static shared_ptr<Layer> makeInputLayer(vector<string> *tokens) {
        return newInstance<InputLayer>();
    }
    
    static shared_ptr<Layer> makeFullyConnectedHiddenLayer(vector<string> *tokens) {
        if (tokens->size() < 2) 
            throw describe(__FILE__, "(", __LINE__, "): " , "ÉjÉÖÅ[ÉçÉìÇÃêîÇèëÇ¢ÇƒÇ≠ÇæÇ≥Ç¢ÅB");
        size_t neuronsNumber = s2ul((*tokens)[1]);
        if (neuronsNumber == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "ÉjÉÖÅ[ÉçÉìÇÕ1å¬à»è„Ç≈Ç»ÇØÇÍÇŒÇ»ÇËÇ‹ÇπÇÒÅB");
        return newInstance<FullyConnectedHiddenLayer>(
            neuronsNumber, 
            newInstance<SigmoidActivationFunction>());
    }
    
    static shared_ptr<Layer> makeOutputLayer(vector<string> *tokens) {
        return newInstance<OutputLayer>(newInstance<SigmoidActivationFunction>());
    }
public:
    shared_ptr<Network> build(
        istream               *is, 
        HyperParameters       *hyperParameters, 
        const shared_ptr<Log> &log) 
    {
        auto layers = newInstance<vector<shared_ptr<Layer>>>();
        if (is) {
            string line;
            while (getLineAndChopCR(*is, line)) {
                if (line.empty() || line.at(0) == '#') 
                    continue;
                vector<string> tokens;
                tokenize(line, " \t", true, [&tokens](const string &token) {
                    tokens.push_back(token);
                });
                if (tokens.size() < 1) 
                    continue;
                if (getMakeLayerProcs()->count(tokens[0]) == 0) 
                    throw describe(__FILE__, "(", __LINE__, "): " , "'", tokens[0], "'Ç∆Ç¢Ç§ëwÇÕÇ†ÇËÇ‹ÇπÇÒÅB");
                layers->push_back(getMakeLayerProcs()->at(tokens[0])(&tokens));
            }
        }
        if (layers->size() < 1) {
            layers->push_back(newInstance<InputLayer>());
            layers->push_back(newInstance<OutputLayer>(newInstance<SigmoidActivationFunction>()));
        } else {
            if (!dynamic_cast<InputLayer *>(layers->front().get())) 
                throw describe(__FILE__, "(", __LINE__, "): " , "ç≈èâÇÃëwÇÕì¸óÕëwÇ≈Ç»ÇØÇÍÇŒÇ»ÇËÇ‹ÇπÇÒÅB");
            if (layers->size() < 2 || !dynamic_cast<OutputLayer *>(layers->back().get())) 
                throw describe(__FILE__, "(", __LINE__, "): " , "ç≈å„ÇÃëwÇÕèoóÕëwÇ≈Ç»ÇØÇÍÇŒÇ»ÇËÇ‹ÇπÇÒÅB");
            for (auto l = layers->begin() + 1; l != layers->end() - 1; l++) {
                if (!dynamic_cast<HiddenLayer *>(l->get())) 
                    throw describe(__FILE__, "(", __LINE__, "): " , "íÜä‘ÇÃëwÇÕâBÇÍëwÇ≈Ç»ÇØÇÍÇŒÇ»ÇËÇ‹ÇπÇÒÅB");
            }
        }
        return newInstance<Network>(layers, hyperParameters, log);
    }
    
    static NetworkBuilder *getInstance() {
        static NetworkBuilder instance;
        return &instance;
    }
};

#endif
