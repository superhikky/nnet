#include "help.h"
#include "mnist.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace std;

#define DEFAULT_INFER_IMAGES_FILE "data/infer.images"
#define DEFAULT_INFER_LABELS_FILE "data/infer.labels"
#define DEFAULT_ONLY_MISTAKE      "yes"

const string USAGE = 
"infviewはnnetの推定ログからテキストアートを表示します。\n"
"使い方: ./infview 設定...\n"
"  設定の書式は'<項目名>=<内容>'です。\n"
"  例えば'inferImagesFile=data/infer.images'のように書きます。\n"
"  また、'@<ファイル名>'でコンフィグファイルから読み込むことができます。\n"
"  例えば'@nnet.config'のように書きます。\n"
"  コンフィグファイルには行ごとに設定を書きます。\n"
"  空白行と'#'で始まる行は無視します。\n"
"  もしdefault.configが存在すれば最初に読み込みます。\n"
"設定項目の一覧\n"
"  inferImagesFile 推定に使った手書き数字画像のファイル。\n"
"                  省略なら" DEFAULT_INFER_IMAGES_FILE "\n"
"  inferLabelsFile 推定に使ったラベルのファイル。\n"
"                  省略なら" DEFAULT_INFER_LABELS_FILE "\n"
"  onlyMistake     不正解のみを表示するかどうか。yesまたはno。省略なら" DEFAULT_ONLY_MISTAKE "\n"
"標準入力: 推定で出力されたログを入力します。\n"
;

int main(int argc, char **argv) {
    int result = 0;
    try {
        if (argc >= 2 && (
                string(argv[1]) == "-h" || 
                string(argv[1]) == "/?"
            )) throw USAGE;
        auto conf = newInstance<map<string, string>>();
        (*conf)["inferImagesFile"]   = DEFAULT_INFER_IMAGES_FILE;
        (*conf)["inferLabelsFile"]   = DEFAULT_INFER_LABELS_FILE;
        (*conf)["onlyMistake"] = DEFAULT_ONLY_MISTAKE;
        if (fileExist("default.config")) 
            setConfig(*openFile<ifstream>("default.config", ios::in), conf.get());
        setConfig(argc - 1, argv + 1, conf.get());
        if (YES_OR_NO.count((*conf)["onlyMistake"]) == 0) 
            throw describe(__FILE__, "(", __LINE__, "): " , "'onlyMistake'はyesまたはnoでなければなりません。");
        
        auto inferMNIST = readMNIST(
            *openFile<ifstream>((*conf)["inferImagesFile"], ios::in | ios::binary), 
            *openFile<ifstream>((*conf)["inferLabelsFile"], ios::in | ios::binary));
        
        string line;
        while (getLineAndChopCR(cin, line)) {
            vector<string> tokens;
            tokenize(line, " \t", true, [&tokens](const string &token) {
                tokens.push_back(token);
            });
            if (tokens.size() < 1 || 
                tokens[0] != "doneInferImage") 
                continue;
            if (tokens.size() != 5) 
                throw describe(__FILE__, "(", __LINE__, "): " , "ログの形式が不正です。");
            size_t inferIndex = s2ul(tokens[1]);
            size_t imageIndex = s2ul(tokens[2]);
            size_t label = s2ul(tokens[3]);
            size_t answer = s2ul(tokens[4]);
            if (label != answer || 
                !YES_OR_NO.at((*conf)["onlyMistake"])) 
            {
                (*inferMNIST)[imageIndex]->putTextArt(cout);
                cout << 
                    "infer="  << inferIndex << " "  << 
                    "image="  << imageIndex << " "  << 
                    "label="  << label      << " "  << 
                    "answer=" << answer     << endl << endl;
            }
        }
    } catch (const string &message) {
        cerr << message << endl;
        result = 1;
    }
    return result;
}
