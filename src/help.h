#ifndef HELP_H
#define HELP_H

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>

using namespace std;

const map<string, bool> YES_OR_NO = {
    {"yes", true}, 
    {"no",  false}, 
};

template <typename Type, typename ...Arguments> 
shared_ptr<Type> newInstance(Arguments&&... arguments) {
    return shared_ptr<Type>(new Type(static_cast<Arguments&&>(arguments)...));
}

inline void describeTo(std::stringstream &ss) {}

template <typename Type, typename ...Arguments> 
void describeTo(stringstream &ss, const Type &value, Arguments&&... arguments) {
    ss << value;
    describeTo(ss, arguments...);
}

template <typename ...Arguments> 
string describe(Arguments&&... arguments) {
    stringstream ss;
    describeTo(ss, arguments...);
    return ss.str();
}

inline bool fileExist(const string &name) {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
}

template <class Stream> 
shared_ptr<Stream> openFile(const string &name, const ios_base::openmode &mode) {
    auto s = newInstance<Stream>(name.c_str(), mode);
    if (!*s) 
        throw describe(__FILE__, "(", __LINE__, "): " , "ファイル'", name , "'を開けません。");
    return s;
}

inline istream &getLineAndChopCR(istream &is, string &line) {
    istream &result = getline(is, line);
    if (!line.empty() && line.at(line.length() - 1) == '\r') 
        line = line.substr(0, line.length() - 1);
    return result;
}

inline pair<string, string> parseArgument(const string &arg) {
    string name;
    string value;
    auto p = arg.find_first_of("=");
    if (p == string::npos) {
        name = arg;
        value = "";
    } else {
        name = arg.substr(0, p);
        value = arg.substr(p + 1);
    }
    return make_pair(name, value);
}

inline void setConfig(
    const string        &arg, 
    map<string, string> *conf) 
{
    auto nameAndValue = parseArgument(arg);
    (*conf)[nameAndValue.first] = nameAndValue.second;
}

inline void setConfig(
    istream             &is, 
    map<string, string> *conf) 
{
    string line;
    while (getLineAndChopCR(is, line)) {
        if (line.empty() || 
            line.at(0) == '#') 
            continue;
        setConfig(line, conf);
    }
}

template <typename Iterator> 
void setConfig(
    const int            &argc, 
    Iterator              argv, 
    map<string, string>  *conf) 
{
    for (auto i = 0; i < argc; i++) {
        string arg = *argv++;
        if (arg.empty()) 
            continue;
        if (arg.at(0) == '@') 
            setConfig(*openFile<ifstream>(arg.substr(1), ios::in), conf);
        else 
            setConfig(arg, conf);
    }
}

class Random {
protected:
    mt19937 engine;
    
    Random() : 
        engine(chrono::system_clock::now().time_since_epoch().count()) {}
public:
    template <typename Type> 
    Type normalDistribution(const Type &a, const Type &b) {
        return normal_distribution<>(a, b)(this->engine);
    }
    
    template <typename Type> 
    Type uniformDistribution(const Type &a, const Type &b) {
        return uniform_int_distribution<>(a, b)(this->engine);
    }
    
    static Random *getInstance() {
        static Random INSTANCE;
        return &INSTANCE;
    }
};

inline void tokenize(
    const string           &str, 
    const string           &delimiters, 
    const bool             &ignoreEmpty, 
    function<void(string)>  putToken) 
{
    for (size_t pos = 0;;) {
        size_t delimPos = str.find_first_of(delimiters, pos);
        string token;
        if (delimPos == string::npos) 
            token = str.substr(pos);
        else 
            token = str.substr(pos, delimPos - pos);
        if (!token.empty() || !ignoreEmpty) 
            putToken(token);
        if (delimPos == string::npos) 
            break;
        pos = delimPos + 1;
    }
}

inline unsigned long s2ul(const string &s) {
    char *end;
    unsigned long ul = strtoul(s.c_str(), &end, 0);
    if (s.empty() || *end) 
        throw describe(__FILE__, "(", __LINE__, "): " , "'", s , "'を符号無し整数に変換できません。");
    return ul;
}

inline double s2d(const string &s) {
    char *end;
    double d = strtod(s.c_str(), &end);
    if (s.empty() || *end) 
        throw describe(__FILE__, "(", __LINE__, "): " , "'", s , "'を浮動小数点数に変換できません。");
    return d;
}

inline unsigned long reverseByteOrder(const unsigned long &ul) {
    unsigned long result = 0;
    result |= ((ul >> 24) & 0xff) <<  0;
    result |= ((ul >> 16) & 0xff) <<  8;
    result |= ((ul >>  8) & 0xff) << 16;
    result |= ((ul >>  0) & 0xff) << 24;
    return result;
}

inline double negateRatio(const double &ratio) {
    return 1.0 - ratio;
}

inline double invert(const double &d) {
    return 1.0 / d;
}

#endif
