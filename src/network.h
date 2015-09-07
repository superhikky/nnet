#ifndef NETWORK_H
#define NETWORK_H

#include "costfunc.h"
#include "help.h"
#include "layer.h"
#include "mnist.h"
#include "neuron.h"
#include "regriz.h"
#include "wgtinit.h"
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

using namespace std;

#define DEFAULT_INPUT_DROPOUT_RATIO "0.0"

#define DEFAULT_FULLY_CONNECTED_NEURONS_NUMBER "30"
#define DEFAULT_FULLY_CONNECTED_DROPOUT_RATIO  "0.0"

struct HyperParameters {
    WeightInitialization *weightInitialization;
    CostFunction         *costFunction;
    Regularization       *regularization;
    double                weightDecayRate;
    double                learningRate;
};

struct Log {
    function<void(
        size_t epochIndex, 
        size_t trainCorrectAnswersNumber, 
        double trainCost, 
        size_t evalCorrectAnswersNumber, 
        double evalCost)> doneTrainEpoch;
    function<void(
        size_t totalTrainCorrectAnswersNumber, 
        double trainCostsAverage, 
        size_t totalEvalCorrectAnswersNumber, 
        double evalCostsAverage)> doneTrain;
    function<void(
        size_t inferImageIndex, 
        size_t imageIndex, 
        size_t label, 
        size_t answer)> doneInferImage;
    function<void(
        size_t correctAnswersNumber, 
        double costsSum)> doneInfer;
    
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
    
    void beginEpoch() {
        for (auto l = this->layers->begin(); l != this->layers->end() - 1; l++) {
            for (auto n : (*(*l)->getNeurons())) {
                for (auto os : *n->getOutputSynapses()) 
                    os->multiplyWeight(invert(negateRatio((*l)->getDropoutRatio())));
            }
        }
    }
    
    void beginBatch() {
        for (auto l = this->layers->begin(); l != this->layers->end() - 1; l++) 
            (*l)->dropNeurons();
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->clearBiasGradient();
                for (auto is : *n->getInputSynapses()) {
                    if (is->getSource()->wasDropped()) 
                        continue;
                    is->clearWeightGradient();
                }
            }
        }
    }
    
    void propagateForward(Image *image) {
        auto inputNeurons = this->layers->front()->getNeurons();
        for (auto i = 0; i < IMAGE_AREA; i++) {
            auto n = (*inputNeurons)[i];
            if (n->wasDropped()) 
                continue;
            n->setOutput((double)(*image->getIntensities())[i] / 255.0);
        }
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            for (auto n : *(*l)->getNeurons()) {
                if (n->wasDropped()) 
                    continue;
                n->clearInput();
                for (auto is : *n->getInputSynapses()) {
                    auto src = is->getSource();
                    if (src->wasDropped()) 
                        continue;
                    n->addInput(is->getWeight() * src->getOutput());
                }
                n->addInput(n->getBias());
                n->setOutput((*l)->getActivationFunction()->computeOutput(n->getInput()));
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
        double costsSum = 0.0;
        auto outputNeurons = this->layers->back()->getNeurons();
        for (auto i = 0; i < outputNeurons->size(); i++) 
            costsSum += this->hyperParameters->costFunction->computeOutputNeuronCost(
                (*outputNeurons)[i].get(), 
                getDesiredOutput(i, label));
        return costsSum;
    }
    
    void propagateBackward(const size_t &label) {
        for (auto l = this->layers->rbegin(); l != this->layers->rend() - 1; l++) {
            if (l == this->layers->rbegin()) {
                for (auto i = 0; i < (*l)->getNeurons()->size(); i++) {
                    auto n = (*(*l)->getNeurons())[i];
                    n->setError(this->hyperParameters->costFunction->computeOutputNeuronError(
                        n.get(), 
                        getDesiredOutput(i, label), 
                        (*l)->getActivationFunction()));
                }
            } else {
                for (auto n : (*(*l)->getNeurons())) {
                    if (n->wasDropped()) 
                        continue;
                    double error = 0.0;
                    for (auto os : *n->getOutputSynapses()) {
                        auto dest = os->getDestination();
                        if (dest->wasDropped()) 
                            continue;
                        error += os->getWeight() * dest->getError();
                    }
                    error *= (*l)->getActivationFunction()->computeDifferentialOutput(n->getInput());
                    n->setError(error);
                }
            }
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->addBiasGradient(n->getError());
                for (auto is : *n->getInputSynapses()) {
                    auto src = is->getSource();
                    if (src->wasDropped()) 
                        continue;
                    is->addWeightGradient(src->getOutput() * n->getError());
                }
            }
        }
    }
    
    void endBatch(const size_t &imagesNumber, const size_t &batchSize) {
        double imageLearningRate = 
            this->hyperParameters->learningRate / 
            (double)batchSize;
        for (auto l = this->layers->begin() + 1; l != this->layers->end(); l++) {
            double outputLearningRate = 
                imageLearningRate * 
                invert(negateRatio((*l)->getDropoutRatio()));
            double inputLearningRate = 
                outputLearningRate * 
                invert(negateRatio((*(l - 1))->getDropoutRatio()));
            for (auto n : (*(*l)->getNeurons())) {
                if (n->wasDropped()) 
                    continue;
                n->setBias(
                    n->getBias() - 
                    outputLearningRate * n->getBiasGradient());
                for (auto is : *n->getInputSynapses()) {
                    if (is->getSource()->wasDropped())
                        continue;
                    is->setWeight(
                        this->hyperParameters->regularization->computeDecayedWeight(
                            is->getWeight(), 
                            inputLearningRate, 
                            this->hyperParameters->weightDecayRate, 
                            imagesNumber) - 
                        inputLearningRate * is->getWeightGradient());
                }
            }
        }
        for (auto l = this->layers->begin(); l != this->layers->end() - 1; l++) 
            (*l)->restoreNeurons();
    }
    
    void endEpoch() {
        for (auto l = this->layers->begin(); l != this->layers->end() - 1; l++) {
            for (auto n : (*(*l)->getNeurons())) {
                for (auto os : *n->getOutputSynapses()) 
                    os->multiplyWeight(negateRatio((*l)->getDropoutRatio()));
            }
        }
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
            log            (log) {}
    
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
        size_t totalTrainCorrectAnswersNumber = 0;
        double totalTrainCostsSum             = 0.0;
        size_t totalEvalCorrectAnswersNumber  = 0;
        double totalEvalCostsSum              = 0.0;
        vector<size_t> imageIndices(trainImagesNumber);
        for (auto i = 0; i < epochsNumber; i++) {
            size_t epochTrainCorrectAnswersNumber = 0;
            double epochTrainCostsSum = 0.0;
            beginEpoch();
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
                propagateForward((*trainingMNIST)[imageIndex].get());
                size_t label = (*trainingMNIST)[imageIndex]->getLabel();
                if (getAnswer() == label) 
                    epochTrainCorrectAnswersNumber++;
                epochTrainCostsSum += computeImageCost(label);
                propagateBackward(label);
                imageIndices[k] = imageIndices[trainImagesNumber - j - 1];
            }
            endEpoch();
            epochTrainCostsSum += this->hyperParameters->regularization->computeWeightsCost(
                this->layers.get(), 
                this->hyperParameters->weightDecayRate);
            
            size_t epochEvalCorrectAnswersNumber = 0;
            double epochEvalCostsSum = 0.0;
            for (auto j = 0; j < evalImagesNumber; j++) {
                propagateForward((*evalMNIST)[j].get());
                size_t label = (*evalMNIST)[j]->getLabel();
                if (getAnswer() == label) 
                    epochEvalCorrectAnswersNumber++;
                epochEvalCostsSum += computeImageCost(label);
            }
            epochEvalCostsSum += this->hyperParameters->regularization->computeWeightsCost(
                this->layers.get(), 
                this->hyperParameters->weightDecayRate);
            
            this->log->doneTrainEpoch(
                i, 
                epochTrainCorrectAnswersNumber, 
                epochTrainCostsSum / (double)trainImagesNumber, 
                epochEvalCorrectAnswersNumber, 
                epochEvalCostsSum  / (double)evalImagesNumber);
            
            totalTrainCorrectAnswersNumber += epochTrainCorrectAnswersNumber;
            totalTrainCostsSum             += epochTrainCostsSum;
            totalEvalCorrectAnswersNumber  += epochEvalCorrectAnswersNumber;
            totalEvalCostsSum              += epochEvalCostsSum;
        }
        this->log->doneTrain(
            totalTrainCorrectAnswersNumber, 
            totalTrainCostsSum / ((double)epochsNumber * (double)trainImagesNumber), 
            totalEvalCorrectAnswersNumber, 
            totalEvalCostsSum  / ((double)epochsNumber * (double)evalImagesNumber));
    }
    
    void infer(
        MNIST        *mnist, 
        const size_t &imagesOffset, 
        const size_t &imagesNumber)
    {
        size_t correctAnswersNumber = 0;
        double costsSum = 0.0;
        for (auto i = 0; i < imagesNumber; i++) {
            size_t imageIndex = imagesOffset + i;
            propagateForward((*mnist)[i].get());
            size_t label = (*mnist)[i]->getLabel();
            if (getAnswer() == label) 
                correctAnswersNumber++;
            costsSum += computeImageCost(label);
            this->log->doneInferImage(
                i, 
                imageIndex, 
                label, 
                getAnswer());
        }
        costsSum += this->hyperParameters->regularization->computeWeightsCost(
            this->layers.get(), 
            this->hyperParameters->weightDecayRate);
        this->log->doneInfer(
            correctAnswersNumber, 
            costsSum / (double)imagesNumber);
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
    
    static shared_ptr<Layer> makeInputLayer(vector<string> *args) {
        auto conf = newInstance<map<string, string>>();
        (*conf)["dropoutRatio"] = DEFAULT_INPUT_DROPOUT_RATIO;
        setConfig(args->size() - 1, args->begin() + 1, conf.get());
        double dropoutRatio = s2d((*conf)["dropoutRatio"]);
        if (dropoutRatio < 0.0 || dropoutRatio >= 1.0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "ドロップアウト率は0.0以上かつ1.0未満でなければなりません。");
        return newInstance<InputLayer>(dropoutRatio);
    }
    
    static shared_ptr<Layer> makeFullyConnectedHiddenLayer(vector<string> *args) {
        auto conf = newInstance<map<string, string>>();
        (*conf)["neuronsNumber"] = DEFAULT_FULLY_CONNECTED_NEURONS_NUMBER;
        (*conf)["dropoutRatio"]  = DEFAULT_FULLY_CONNECTED_DROPOUT_RATIO;
        setConfig(args->size() - 1, args->begin() + 1, conf.get());
        size_t neuronsNumber = s2ul((*conf)["neuronsNumber"]);
        if (neuronsNumber == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "ニューロンは1個以上でなければなりません。");
        double dropoutRatio = s2d((*conf)["dropoutRatio"]);
        if (dropoutRatio < 0.0 || dropoutRatio >= 1.0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "ドロップアウト率は0.0以上かつ1.0未満でなければなりません。");
        return newInstance<FullyConnectedHiddenLayer>(
            neuronsNumber, 
            dropoutRatio, 
            newInstance<SigmoidActivationFunction>());
    }
    
    static shared_ptr<Layer> makeOutputLayer(vector<string> *args) {
        return newInstance<OutputLayer>(newInstance<SigmoidActivationFunction>());
    }
public:
    shared_ptr<Network> build(
        istream               &is, 
        HyperParameters       *hyperParameters, 
        const shared_ptr<Log> &log) 
    {
        auto layers = newInstance<vector<shared_ptr<Layer>>>();
        string line;
        while (getLineAndChopCR(is, line)) {
            if (line.empty()) 
                continue;
            vector<string> args;
            tokenize(line, " \t", true, [&args](const string &token) {
                args.push_back(token);
            });
            if (args.size() < 1 || 
                args[0].at(0) == '#') 
                continue;
            string layerType = args[0];
            if (getMakeLayerProcs()->count(layerType) == 0) 
                throw describe(__FILE__, "(", __LINE__, "): " , "'", layerType, "'という層はありません。");
            layers->push_back(getMakeLayerProcs()->at(layerType)(&args));
        }
        if (layers->size() < 1 || !dynamic_cast<InputLayer *>(layers->front().get())) 
            throw describe(__FILE__, "(", __LINE__, "): " , "最初の層は入力層でなければなりません。");
        if (layers->size() < 2 || !dynamic_cast<OutputLayer *>(layers->back().get())) 
            throw describe(__FILE__, "(", __LINE__, "): " , "最後の層は出力層でなければなりません。");
        for (auto l = layers->begin() + 1; l != layers->end() - 1; l++) {
            if (!dynamic_cast<HiddenLayer *>(l->get())) 
                throw describe(__FILE__, "(", __LINE__, "): " , "中間の層は隠れ層でなければなりません。");
        }
        for (auto i = 1; i < layers->size(); i++) 
            (*layers)[i]->connect(
                (*layers)[i - 1].get(), 
                hyperParameters->weightInitialization);
        return newInstance<Network>(layers, hyperParameters, log);
    }
    
    static NetworkBuilder *getInstance() {
        static NetworkBuilder INSTANCE;
        return &INSTANCE;
    }
};

#endif
