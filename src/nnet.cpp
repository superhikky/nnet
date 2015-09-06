#include "costfunc.h"
#include "help.h"
#include "layer.h"
#include "mnist.h"
#include "network.h"
#include "regriz.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

using namespace std;

#define DEFAULT_NETWORK_FILE        "default.network"
#define DEFAULT_PARAMETERS_FILE     "default.parameters"
#define DEFAULT_COST_FUNCTION       "quadratic"
#define DEFAULT_REGULARIZATION      "null"
#define DEFAULT_WEIGHT_DECAY_RATE   "0.1"
#define DEFAULT_TRAIN_IMAGES_FILE   "data/train.images"
#define DEFAULT_TRAIN_LABELS_FILE   "data/train.labels"
#define DEFAULT_EVAL_IMAGES_FILE    "data/infer.images"
#define DEFAULT_EVAL_LABELS_FILE    "data/infer.labels"
#define DEFAULT_TRAIN_IMAGES_OFFSET "0"
#define DEFAULT_TRAIN_IMAGES_NUMBER "1000"
#define DEFAULT_EVAL_IMAGES_OFFSET  "0"
#define DEFAULT_EVAL_IMAGES_NUMBER  "100"
#define DEFAULT_READ_PARAMETERS     "yes"
#define DEFAULT_EPOCHS_NUMBER       "10"
#define DEFAULT_BATCH_SIZE          "10"
#define DEFAULT_LEARNING_RATE       "5.0"
#define DEFAULT_INFER_IMAGES_FILE   "data/infer.images"
#define DEFAULT_INFER_LABELS_FILE   "data/infer.labels"
#define DEFAULT_INFER_IMAGES_OFFSET "0"
#define DEFAULT_INFER_IMAGES_NUMBER "100"

const string USAGE = 
"nnet�̓j���[�����l�b�g���[�N�����A�菑�������摜�ɂ��P���Ɛ�����s���܂��B\n"
"�g����: ./nnet ���� �ݒ�...\n"
"  ���߂ɂ͎��s���鏈�����w�肵�܂��B\n"
"  �ݒ�̏�����'<���ږ�>=<���e>'�ł��B\n"
"  �Ⴆ��'trainImagesFile=data/train.images'�̂悤�ɏ����܂��B\n"
"  �܂��A'@<�t�@�C����>'�ŃR���t�B�O�t�@�C������ǂݍ��ނ��Ƃ��ł��܂��B\n"
"  �Ⴆ��'@nnet.config'�̂悤�ɏ����܂��B\n"
"  �R���t�B�O�t�@�C���ɂ͍s���Ƃɐݒ�������܂��B\n"
"  �󔒍s��'#'�Ŏn�܂�s�͖������܂��B\n"
"  ����default.config�����݂���΍ŏ��ɓǂݍ��݂܂��B\n"
"���߂̈ꗗ\n"
"  train �l�b�g���[�N���P������\n"
"  infer �摜�̃��x���𐄒肷��\n"
"train���߂�infer���߂ɋ��ʂ̐ݒ荀�ڂ̈ꗗ\n"
"  networkFile     �l�b�g���[�N���`�����t�@�C���B\n"
"                  �ȗ��Ȃ�" DEFAULT_NETWORK_FILE "�B\n"
"                  �������݂��Ȃ���΃f�t�H���g�̃l�b�g���[�N���g���܂��B\n"
"  parametersFile  �p�����[�^�̃t�@�C���B\n"
"                  �ȗ��Ȃ�" DEFAULT_PARAMETERS_FILE "\n"
"  costFunction    �R�X�g�֐��B�ȗ��Ȃ�" DEFAULT_COST_FUNCTION "\n"
"  regularization  �������B�ȗ��Ȃ�" DEFAULT_REGULARIZATION "\n"
"  weightDecayRate �d�ݕ␳���B�ȗ��Ȃ�" DEFAULT_WEIGHT_DECAY_RATE "\n"
"train���߂̐ݒ荀�ڂ̈ꗗ\n"
"  trainImagesFile   �P���Ɏg���菑�������摜�̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_TRAIN_IMAGES_FILE "\n"
"  trainLabelsFile   �P���Ɏg�����x���̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_TRAIN_LABELS_FILE "\n"
"  trainImagesOffset �P���Ɏg���摜�̃I�t�Z�b�g�B�ȗ��Ȃ�" DEFAULT_TRAIN_IMAGES_OFFSET "\n"
"  trainImagesNumber �P���Ɏg���摜�̐��B�ȗ��Ȃ�" DEFAULT_TRAIN_IMAGES_NUMBER "\n"
"  evalImagesFile    �]���Ɏg���菑�������摜�̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_EVAL_IMAGES_FILE "\n"
"  evalLabelsFile    �]���Ɏg�����x���̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_EVAL_LABELS_FILE "\n"
"  evalImagesOffset  �]���Ɏg���摜�̃I�t�Z�b�g�B�ȗ��Ȃ�" DEFAULT_EVAL_IMAGES_OFFSET "\n"
"  evalImagesNumber  �]���Ɏg���摜�̐��B�ȗ��Ȃ�" DEFAULT_EVAL_IMAGES_NUMBER "\n"
"  readParameters    �p�����[�^��ǂݍ��ނ��ǂ����Byes�܂���no�B�ȗ��Ȃ�" DEFAULT_READ_PARAMETERS "\n"
"  epochsNumber      ����̐��B�ȗ��Ȃ�" DEFAULT_EPOCHS_NUMBER "\n"
"  batchSize         �o�b�`�̑傫���B�ȗ��Ȃ�" DEFAULT_BATCH_SIZE "\n"
"  learningRate      �w�K���B�ȗ��Ȃ�" DEFAULT_LEARNING_RATE "\n"
"infer���߂̐ݒ荀�ڂ̈ꗗ\n"
"  inferImagesFile   ����Ɏg���菑�������摜�̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_INFER_IMAGES_FILE "\n"
"  inferLabelsFile   ����Ɏg�����x���̃t�@�C���B\n"
"                    �ȗ��Ȃ�" DEFAULT_INFER_LABELS_FILE "\n"
"  inferImagesOffset ����Ɏg���摜�̃I�t�Z�b�g�B�ȗ��Ȃ�" DEFAULT_INFER_IMAGES_OFFSET "\n"
"  inferImagesNumber ����Ɏg���摜�̐��B�ȗ��Ȃ�" DEFAULT_INFER_IMAGES_NUMBER "\n"
"�l�b�g���[�N�̒�`\n"
"  �s���Ƃɑw���`���܂��B������'�w�̎�� �ݒ�...'�ł��B\n"
"  �Ⴆ��'fullyConnected neuronsNumber=30'�̂悤�ɏ����܂��B\n"
"  �摜�f�[�^�͏�̑w���牺�̑w�Ɍ������Ď��X�Ɠ`�d���܂��B\n"
"  �ŏ��͓��͑w�A�Ō�͏o�͑w�A���Ԃ͂���ȊO�̎�ނ̑w�łȂ���΂Ȃ�܂���B\n"
"  �󔒍s��'#'�Ŏn�܂�s�͖������܂��B\n"
"�w�̎�ނ̈ꗗ\n"
"  input          ���͑w\n"
"    �ݒ荀�ڂ̈ꗗ\n"
"      dropoutRatio �h���b�v�A�E�g���B>= 0.0 && < 1.0�B�ȗ��Ȃ�" DEFAULT_INPUT_DROPOUT_RATIO "\n"
"  output         �o�͑w\n"
"  fullyConnected �S�ڑ��w\n"
"    �ݒ荀�ڂ̈ꗗ\n"
"      neuronsNumber �j���[�����̐��B�ȗ��Ȃ�" DEFAULT_FULLY_CONNECTED_NEURONS_NUMBER "\n"
"      dropoutRatio  �h���b�v�A�E�g���B>= 0.0 && < 1.0�B�ȗ��Ȃ�" DEFAULT_FULLY_CONNECTED_DROPOUT_RATIO "\n"
"�f�t�H���g�̃l�b�g���[�N: ���͑w�Əo�͑w��������܂���B\n"
"  input\n"
"  output\n"
"�R�X�g�֐��̈ꗗ\n"
"  quadratic    ���ϓ��덷\n"
"  crossEntropy �N���X�G���g���s�[\n"
"�������̈ꗗ\n"
"  null �Ȃ�\n"
"  l1   L1 ������\n"
"  l2   L2 ������\n"
"�W���o��: ���O���o�͂��܂��B\n"
"  �s���Ƃ̏�����'���O�̎�� �f�[�^...'�ł��B�^�u�ŋ�؂�܂��B\n"
"���O�̎�ނ̈ꗗ\n"
"  doneTrainEpoch ����̌P��������\n"
"    �f�[�^�̈ꗗ\n"
"      ����̔ԍ�\n"
"      �P���̐���\n"
"      �P���̃R�X�g\n"
"      �]���̐���\n"
"      �]���̃R�X�g\n"
"  doneTrain      �P��������\n"
"    �f�[�^�̈ꗗ\n"
"      �P���̑�����\n"
"      �P���̕��σR�X�g\n"
"      �]���̑�����\n"
"      �]���̕��σR�X�g\n"
"  doneInferImage �摜�̐��������\n"
"    �f�[�^�̈ꗗ\n"
"      �摜�̐���̔ԍ�\n"
"      �摜�̔ԍ�\n"
"      �����̃��x��\n"
"      �l�b�g���[�N�����肵������\n"
"  doneInfer      ���������\n"
"    �f�[�^�̈ꗗ\n"
"      ����\n"
"      �R�X�g\n"
;

const string DEFAULT_NETWORK = 
"input\n"
"output\n"
;

void train(map<string, string> *conf, HyperParameters *hyperParameters);
void infer(map<string, string> *conf, HyperParameters *hyperParameters);

using CommandProc = function<void(map<string, string> *, HyperParameters *)>;
inline const map<string, CommandProc> *getCommandProcs() {
    static const map<string, CommandProc> COMMAND_PROCS = {
        {"train", &train}, 
        {"infer", &infer}, 
    };
    return &COMMAND_PROCS;
}

int main(int argc, char **argv) {
    int result = 0;
    try {
        if (argc == 1 || 
            string(argv[1]) == "-h" || 
            string(argv[1]) == "/?") 
            throw USAGE;
        string command = argv[1];
        if (getCommandProcs()->count(command) == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "'", command, "'�Ƃ������߂͂���܂���B");
        auto conf = newInstance<map<string, string>>();
        (*conf)["networkFile"]       = DEFAULT_NETWORK_FILE;
        (*conf)["parametersFile"]    = DEFAULT_PARAMETERS_FILE;
        (*conf)["costFunction"]      = DEFAULT_COST_FUNCTION;
        (*conf)["regularization"]    = DEFAULT_REGULARIZATION;
        (*conf)["weightDecayRate"]   = DEFAULT_WEIGHT_DECAY_RATE;
        (*conf)["trainImagesFile"]   = DEFAULT_TRAIN_IMAGES_FILE;
        (*conf)["trainLabelsFile"]   = DEFAULT_TRAIN_LABELS_FILE;
        (*conf)["trainImagesOffset"] = DEFAULT_TRAIN_IMAGES_OFFSET;
        (*conf)["trainImagesNumber"] = DEFAULT_TRAIN_IMAGES_NUMBER;
        (*conf)["evalImagesFile"]    = DEFAULT_EVAL_IMAGES_FILE;
        (*conf)["evalLabelsFile"]    = DEFAULT_EVAL_LABELS_FILE;
        (*conf)["evalImagesOffset"]  = DEFAULT_EVAL_IMAGES_OFFSET;
        (*conf)["evalImagesNumber"]  = DEFAULT_EVAL_IMAGES_NUMBER;
        (*conf)["readParameters"]    = DEFAULT_READ_PARAMETERS;
        (*conf)["epochsNumber"]      = DEFAULT_EPOCHS_NUMBER;
        (*conf)["batchSize"]         = DEFAULT_BATCH_SIZE;
        (*conf)["learningRate"]      = DEFAULT_LEARNING_RATE;
        (*conf)["inferImagesFile"]   = DEFAULT_INFER_IMAGES_FILE;
        (*conf)["inferLabelsFile"]   = DEFAULT_INFER_LABELS_FILE;
        (*conf)["inferImagesOffset"] = DEFAULT_INFER_IMAGES_OFFSET;
        (*conf)["inferImagesNumber"] = DEFAULT_INFER_IMAGES_NUMBER;
        if (fileExist("default.config")) 
            setConfig(*openFile<ifstream>("default.config", ios::in), conf.get());
        setConfig(argc - 2, argv + 2, conf.get());
        if (getCostFunctions()->count((*conf)["costFunction"]) == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "'", (*conf)["costFunction"], "'�Ƃ����R�X�g�֐��͂���܂���B");
        if (getRegularizations()->count((*conf)["regularization"]) == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "'", (*conf)["regularization"], "'�Ƃ����������͂���܂���B");
        auto hyperParameters = newInstance<HyperParameters>();
        hyperParameters->costFunction    = getCostFunctions()->at((*conf)["costFunction"]).get();
        hyperParameters->regularization  = getRegularizations()->at((*conf)["regularization"]).get();
        hyperParameters->weightDecayRate = s2d((*conf)["weightDecayRate"]);
        getCommandProcs()->at(command)(conf.get(), hyperParameters.get());
    } catch (const string &message) {
        cerr << message << endl;
        result = 1;
    }
    return result;
}

void train(map<string, string> *conf, HyperParameters *hyperParameters) {
    if (YES_OR_NO.count((*conf)["readParameters"]) == 0) 
        throw describe(__FILE__, "(", __LINE__, "): " , "'readParameters'��yes�܂���no�łȂ���΂Ȃ�܂���B");
    
    auto trainMNIST = readMNIST(
        *openFile<ifstream>((*conf)["trainImagesFile"], ios::in | ios::binary), 
        *openFile<ifstream>((*conf)["trainLabelsFile"], ios::in | ios::binary));
    auto evalMNIST = readMNIST(
        *openFile<ifstream>((*conf)["evalImagesFile"], ios::in | ios::binary), 
        *openFile<ifstream>((*conf)["evalLabelsFile"], ios::in | ios::binary));
    
    shared_ptr<istream> networkIS;
    if ((*conf)["networkFile"].empty() || 
        !fileExist((*conf)["networkFile"])) 
        networkIS = newInstance<stringstream>(DEFAULT_NETWORK);
    else
        networkIS = openFile<ifstream>((*conf)["networkFile"], ios::in);
    hyperParameters->learningRate = s2d((*conf)["learningRate"]);
    auto log = newInstance<Log>();
    auto net = NetworkBuilder::getInstance()->build(
        *networkIS, 
        hyperParameters, 
        log);
    if (fileExist((*conf)["parametersFile"]) && 
        YES_OR_NO.at((*conf)["readParameters"])) 
        net->read(*openFile<ifstream>((*conf)["parametersFile"], ios::in | ios::binary));
    
    log->doneTrainEpoch = [](
        const size_t &epochIndex, 
        const size_t &trainCorrectAnswersNumber, 
        const double &trainCost, 
        const size_t &evalCorrectAnswersNumber, 
        const double &evalCost) 
    {
        cout << 
            "doneTrainEpoch"          << "\t" << 
            epochIndex                << "\t" << 
            trainCorrectAnswersNumber << "\t" << 
            trainCost                 << "\t" << 
            evalCorrectAnswersNumber  << "\t" << 
            evalCost                  << endl;
    };
    log->doneTrain = [](
        const size_t &totalTrainCorrectAnswersNumber, 
        const double &trainCostAverage, 
        const size_t &totalEvalCorrectAnswersNumber, 
        const double &evalCostAverage) 
    {
        cout << 
            "doneTrain"                    << "\t" << 
            totalTrainCorrectAnswersNumber << "\t" << 
            trainCostAverage               << "\t" << 
            totalEvalCorrectAnswersNumber  << "\t" << 
            evalCostAverage                << endl;
    };
    net->train(
        s2ul((*conf)["epochsNumber"]), 
        s2ul((*conf)["batchSize"]), 
        trainMNIST.get(), 
        s2ul((*conf)["trainImagesOffset"]), 
        s2ul((*conf)["trainImagesNumber"]), 
        evalMNIST.get(), 
        s2ul((*conf)["evalImagesOffset"]), 
        s2ul((*conf)["evalImagesNumber"]));
    
    net->write(*openFile<ofstream>((*conf)["parametersFile"], ios::out | ios::binary | ios::trunc));
}

void infer(map<string, string> *conf, HyperParameters *hyperParameters) {
    auto mnist = readMNIST(
        *openFile<ifstream>((*conf)["inferImagesFile"], ios::in | ios::binary), 
        *openFile<ifstream>((*conf)["inferLabelsFile"], ios::in | ios::binary));
    
    auto log = newInstance<Log>();
    auto net = NetworkBuilder::getInstance()->build(
        *openFile<ifstream>((*conf)["networkFile"], ios::in), 
        hyperParameters, 
        log);
    net->read(*openFile<ifstream>((*conf)["parametersFile"], ios::in | ios::binary));
    
    log->doneInferImage = [](
        const size_t &inferImageIndex, 
        const size_t &imageIndex, 
        const size_t &label, 
        const size_t &answer) 
    {
        cout << 
            "doneInferImage" << "\t" << 
            inferImageIndex  << "\t" << 
            imageIndex       << "\t" << 
            label            << "\t" << 
            answer           << endl;
    };
    log->doneInfer = [](
        const size_t &correctAnswersNumber, 
        const double &cost) 
    {
        cout << 
            "doneInfer"          << "\t" << 
            correctAnswersNumber << "\t" << 
            cost                 << endl;
    };
    net->infer(
        mnist.get(), 
        s2ul((*conf)["inferImagesOffset"]), 
        s2ul((*conf)["inferImagesNumber"]));
}
