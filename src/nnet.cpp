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
"nnetはニューラルネットワークを作り、手書き数字画像による訓練と推定を行います。\n"
"使い方: ./nnet 命令 設定...\n"
"  命令には実行する処理を指定します。\n"
"  設定の書式は'<項目名>=<内容>'です。\n"
"  例えば'trainImagesFile=data/train.images'のように書きます。\n"
"  また、'@<ファイル名>'でコンフィグファイルから読み込むことができます。\n"
"  例えば'@nnet.config'のように書きます。\n"
"  コンフィグファイルには行ごとに設定を書きます。\n"
"  空白行と'#'で始まる行は無視します。\n"
"  もしdefault.configが存在すれば最初に読み込みます。\n"
"命令の一覧\n"
"  train ネットワークを訓練する\n"
"  infer 画像のラベルを推定する\n"
"train命令とinfer命令に共通の設定項目の一覧\n"
"  networkFile     ネットワークを定義したファイル。\n"
"                  省略なら" DEFAULT_NETWORK_FILE "。\n"
"                  もし存在しなければデフォルトのネットワークを使います。\n"
"  parametersFile  パラメータのファイル。\n"
"                  省略なら" DEFAULT_PARAMETERS_FILE "\n"
"  costFunction    コスト関数。省略なら" DEFAULT_COST_FUNCTION "\n"
"  regularization  正則化。省略なら" DEFAULT_REGULARIZATION "\n"
"  weightDecayRate 重み補正率。省略なら" DEFAULT_WEIGHT_DECAY_RATE "\n"
"train命令の設定項目の一覧\n"
"  trainImagesFile   訓練に使う手書き数字画像のファイル。\n"
"                    省略なら" DEFAULT_TRAIN_IMAGES_FILE "\n"
"  trainLabelsFile   訓練に使うラベルのファイル。\n"
"                    省略なら" DEFAULT_TRAIN_LABELS_FILE "\n"
"  trainImagesOffset 訓練に使う画像のオフセット。省略なら" DEFAULT_TRAIN_IMAGES_OFFSET "\n"
"  trainImagesNumber 訓練に使う画像の数。省略なら" DEFAULT_TRAIN_IMAGES_NUMBER "\n"
"  evalImagesFile    評価に使う手書き数字画像のファイル。\n"
"                    省略なら" DEFAULT_EVAL_IMAGES_FILE "\n"
"  evalLabelsFile    評価に使うラベルのファイル。\n"
"                    省略なら" DEFAULT_EVAL_LABELS_FILE "\n"
"  evalImagesOffset  評価に使う画像のオフセット。省略なら" DEFAULT_EVAL_IMAGES_OFFSET "\n"
"  evalImagesNumber  評価に使う画像の数。省略なら" DEFAULT_EVAL_IMAGES_NUMBER "\n"
"  readParameters    パラメータを読み込むかどうか。yesまたはno。省略なら" DEFAULT_READ_PARAMETERS "\n"
"  epochsNumber      世代の数。省略なら" DEFAULT_EPOCHS_NUMBER "\n"
"  batchSize         バッチの大きさ。省略なら" DEFAULT_BATCH_SIZE "\n"
"  learningRate      学習率。省略なら" DEFAULT_LEARNING_RATE "\n"
"infer命令の設定項目の一覧\n"
"  inferImagesFile   推定に使う手書き数字画像のファイル。\n"
"                    省略なら" DEFAULT_INFER_IMAGES_FILE "\n"
"  inferLabelsFile   推定に使うラベルのファイル。\n"
"                    省略なら" DEFAULT_INFER_LABELS_FILE "\n"
"  inferImagesOffset 推定に使う画像のオフセット。省略なら" DEFAULT_INFER_IMAGES_OFFSET "\n"
"  inferImagesNumber 推定に使う画像の数。省略なら" DEFAULT_INFER_IMAGES_NUMBER "\n"
"ネットワークの定義\n"
"  行ごとに層を定義します。書式は'層の種類 設定...'です。\n"
"  例えば'fullyConnected neuronsNumber=30'のように書きます。\n"
"  画像データは上の層から下の層に向かって次々と伝播します。\n"
"  最初は入力層、最後は出力層、中間はそれ以外の種類の層でなければなりません。\n"
"  空白行と'#'で始まる行は無視します。\n"
"層の種類の一覧\n"
"  input          入力層\n"
"    設定項目の一覧\n"
"      dropoutRatio ドロップアウト率。>= 0.0 && < 1.0。省略なら" DEFAULT_INPUT_DROPOUT_RATIO "\n"
"  output         出力層\n"
"  fullyConnected 全接続層\n"
"    設定項目の一覧\n"
"      neuronsNumber ニューロンの数。省略なら" DEFAULT_FULLY_CONNECTED_NEURONS_NUMBER "\n"
"      dropoutRatio  ドロップアウト率。>= 0.0 && < 1.0。省略なら" DEFAULT_FULLY_CONNECTED_DROPOUT_RATIO "\n"
"デフォルトのネットワーク: 入力層と出力層しかありません。\n"
"  input\n"
"  output\n"
"コスト関数の一覧\n"
"  quadratic    平均二乗誤差\n"
"  crossEntropy クロスエントロピー\n"
"正則化の一覧\n"
"  null なし\n"
"  l1   L1 正則化\n"
"  l2   L2 正則化\n"
"標準出力: ログを出力します。\n"
"  行ごとの書式は'ログの種類 データ...'です。タブで区切ります。\n"
"ログの種類の一覧\n"
"  doneTrainEpoch 世代の訓練を完了\n"
"    データの一覧\n"
"      世代の番号\n"
"      訓練の正解数\n"
"      訓練のコスト\n"
"      評価の正解数\n"
"      評価のコスト\n"
"  doneTrain      訓練を完了\n"
"    データの一覧\n"
"      訓練の総正解数\n"
"      訓練の平均コスト\n"
"      評価の総正解数\n"
"      評価の平均コスト\n"
"  doneInferImage 画像の推定を完了\n"
"    データの一覧\n"
"      画像の推定の番号\n"
"      画像の番号\n"
"      正解のラベル\n"
"      ネットワークが推定した答え\n"
"  doneInfer      推定を完了\n"
"    データの一覧\n"
"      正解数\n"
"      コスト\n"
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
            throw describe(__FILE__, "(", __LINE__, "): " , "'", command, "'という命令はありません。");
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
            throw describe(__FILE__, "(", __LINE__, "): " , "'", (*conf)["costFunction"], "'というコスト関数はありません。");
        if (getRegularizations()->count((*conf)["regularization"]) == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "'", (*conf)["regularization"], "'という正則化はありません。");
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
        throw describe(__FILE__, "(", __LINE__, "): " , "'readParameters'はyesまたはnoでなければなりません。");
    
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
