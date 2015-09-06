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
"infview��nnet�̐��胍�O����e�L�X�g�A�[�g��\�����܂��B\n"
"�g����: ./infview �ݒ�...\n"
"  �ݒ�̏�����'<���ږ�>=<���e>'�ł��B\n"
"  �Ⴆ��'inferImagesFile=data/infer.images'�̂悤�ɏ����܂��B\n"
"  �܂��A'@<�t�@�C����>'�ŃR���t�B�O�t�@�C������ǂݍ��ނ��Ƃ��ł��܂��B\n"
"  �Ⴆ��'@nnet.config'�̂悤�ɏ����܂��B\n"
"  �R���t�B�O�t�@�C���ɂ͍s���Ƃɐݒ�������܂��B\n"
"  �󔒍s��'#'�Ŏn�܂�s�͖������܂��B\n"
"  ����default.config�����݂���΍ŏ��ɓǂݍ��݂܂��B\n"
"�ݒ荀�ڂ̈ꗗ\n"
"  inferImagesFile ����Ɏg�����菑�������摜�̃t�@�C���B\n"
"                  �ȗ��Ȃ�" DEFAULT_INFER_IMAGES_FILE "\n"
"  inferLabelsFile ����Ɏg�������x���̃t�@�C���B\n"
"                  �ȗ��Ȃ�" DEFAULT_INFER_LABELS_FILE "\n"
"  onlyMistake     �s�����݂̂�\�����邩�ǂ����Byes�܂���no�B�ȗ��Ȃ�" DEFAULT_ONLY_MISTAKE "\n"
"�W������: ����ŏo�͂��ꂽ���O����͂��܂��B\n"
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
            throw describe(__FILE__, "(", __LINE__, "): " , "'onlyMistake'��yes�܂���no�łȂ���΂Ȃ�܂���B");
        
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
                throw describe(__FILE__, "(", __LINE__, "): " , "���O�̌`�����s���ł��B");
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
