#ifndef MNIST_H
#define MNIST_H

#include "help.h"
#include <fstream>
#include <memory>
#include <vector>

using namespace std;

struct LetterIntensity {
    double upperIntensity;
    double lowerIntensity;
};

constexpr size_t IMAGE_SIDE_LENGTH   = 28;
constexpr size_t IMAGE_AREA          = IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH;
constexpr size_t LABEL_VALUES_NUMBER = 10;

constexpr int    FIRST_PRINTABLE_LETTER   = ' ';
constexpr size_t PRINTABLE_LETTERS_NUMBER = 95;

constexpr LetterIntensity PRINTABLE_LETTER_INTENSITIES[PRINTABLE_LETTERS_NUMBER] = {
    {  0.000000,   0.000000}, 
    { 72.857143,  57.954545}, 
    { 72.857143,   0.000000}, 
    {191.250000, 208.636364}, 
    {200.357143, 197.045455}, 
    {182.142857, 208.636364}, 
    {136.607143, 197.045455}, 
    { 54.642857,   0.000000}, 
    { 81.964286,  92.727273}, 
    { 81.964286,  92.727273}, 
    {100.178571, 115.909091}, 
    {100.178571,  46.363636}, 
    {  0.000000,  69.545455}, 
    { 63.750000,   0.000000}, 
    {  0.000000,  46.363636}, 
    { 81.964286,  81.136364}, 
    {154.821429, 150.681818}, 
    { 91.071429,  69.545455}, 
    {118.392857, 139.090909}, 
    {136.607143, 127.500000}, 
    {127.500000, 162.272727}, 
    {173.035714, 127.500000}, 
    {173.035714, 150.681818}, 
    {127.500000,  69.545455}, 
    {163.928571, 150.681818}, 
    {163.928571, 162.272727}, 
    { 36.428571,  46.363636}, 
    { 36.428571,  69.545455}, 
    { 63.750000,  69.545455}, 
    { 63.750000,  81.136364}, 
    { 63.750000,  69.545455}, 
    {127.500000,  69.545455}, 
    {227.678571, 243.409091}, 
    {118.392857, 173.863636}, 
    {200.357143, 173.863636}, 
    {118.392857, 127.500000}, 
    {173.035714, 173.863636}, 
    {173.035714, 139.090909}, 
    {173.035714,  69.545455}, 
    {118.392857, 197.045455}, 
    {191.250000, 139.090909}, 
    { 91.071429,  92.727273}, 
    { 72.857143, 127.500000}, 
    {154.821429, 139.090909}, 
    { 72.857143, 139.090909}, 
    {236.785714, 231.818182}, 
    {200.357143, 185.454545}, 
    {163.928571, 162.272727}, 
    {200.357143,  69.545455}, 
    {163.928571, 197.045455}, 
    {200.357143, 139.090909}, 
    {127.500000, 127.500000}, 
    {127.500000,  69.545455}, 
    {145.714286, 150.681818}, 
    {145.714286, 104.318182}, 
    {255.000000, 162.272727}, 
    {127.500000, 139.090909}, 
    {136.607143,  69.545455}, 
    {127.500000, 139.090909}, 
    {118.392857, 139.090909}, 
    {163.928571, 115.909091}, 
    {118.392857, 139.090909}, 
    { 45.535714,   0.000000}, 
    {  0.000000, 104.318182}, 
    { 27.321429,   0.000000}, 
    { 63.750000, 208.636364}, 
    {127.500000, 173.863636}, 
    { 63.750000, 115.909091}, 
    {127.500000, 173.863636}, 
    { 63.750000, 197.045455}, 
    {136.607143,  69.545455}, 
    { 81.964286, 255.000000}, 
    {127.500000, 139.090909}, 
    { 45.535714,  69.545455}, 
    { 45.535714, 115.909091}, 
    {100.178571, 150.681818}, 
    { 72.857143,  69.545455}, 
    {118.392857, 208.636364}, 
    { 81.964286, 139.090909}, 
    { 63.750000, 150.681818}, 
    { 81.964286, 185.454545}, 
    { 81.964286, 185.454545}, 
    { 63.750000,  69.545455}, 
    { 72.857143, 150.681818}, 
    {109.285714,  92.727273}, 
    { 54.642857, 173.863636}, 
    { 54.642857, 115.909091}, 
    { 81.964286, 208.636364}, 
    { 54.642857, 127.500000}, 
    { 54.642857, 139.090909}, 
    { 81.964286, 139.090909}, 
    {100.178571, 115.909091}, 
    { 81.964286, 104.318182}, 
    {100.178571, 115.909091}, 
    { 72.857143,   0.000000}, 
};

class Image {
protected:
    size_t                index;
    vector<unsigned char> intensities;
    size_t                label;
public:
    Image(const size_t &index) : 
        index(index), 
        intensities(IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH) {}
    size_t getIndex() 
        { return this->index; }
    vector<unsigned char> *getIntensities() 
        { return &this->intensities; }
    unsigned char getIntensity(const size_t &x, const size_t &y) 
        { return this->intensities[IMAGE_SIDE_LENGTH * y + x]; }
    size_t getLabel() 
        { return this->label; }
    void setLabel(const size_t &label) 
        { this->label = label; }
    
    void putTextArt(ostream &os) {
        auto putHorizontalBorder = [&os]() {
            os << '+';
            for (auto x = 0; x < IMAGE_SIDE_LENGTH; x++) 
                os << '-';
            os << '+' << endl;
        };
        
        putHorizontalBorder();
        for (auto y = 0; y < IMAGE_SIDE_LENGTH; y += 2) {
            os << '|';
            for (auto x = 0; x < IMAGE_SIDE_LENGTH; x++) {
                size_t minDiffLetterIndex = 0;
                double minDiff = 512.0;
                for (auto i = 0; i < PRINTABLE_LETTERS_NUMBER; i++) {
                    if (i == '\\' - FIRST_PRINTABLE_LETTER) 
                        continue;
                    double diff = 
                        fabs(getIntensity(x, y)     - 
                            PRINTABLE_LETTER_INTENSITIES[i].upperIntensity) + 
                        fabs(getIntensity(x, y + 1) - 
                            PRINTABLE_LETTER_INTENSITIES[i].lowerIntensity);
                    if (diff < minDiff) {
                        minDiffLetterIndex = i;
                        minDiff = diff;
                    }
                }
                os << (char)(FIRST_PRINTABLE_LETTER + minDiffLetterIndex);
            }
            os << '|' << endl;
        }
        putHorizontalBorder();
    }
};

using MNIST = vector<shared_ptr<Image>>;

inline shared_ptr<MNIST> readMNIST(istream &imagesIS, istream &labelsIS) {
    auto mnist = newInstance<MNIST>();
    
    imagesIS.seekg(4, ios_base::cur);
    unsigned long imagesNumber;
    imagesIS.read((char *)&imagesNumber, sizeof(unsigned long));
    imagesNumber = reverseByteOrder(imagesNumber);
    unsigned long rowsNumber;
    imagesIS.read((char *)&rowsNumber, sizeof(unsigned long));
    rowsNumber = reverseByteOrder(rowsNumber);
    if (rowsNumber != IMAGE_SIDE_LENGTH) 
        throw describe(__FILE__, "(", __LINE__, "): ", "‰æ‘œ‚Ì‚‚³‚Í", IMAGE_SIDE_LENGTH, "‚Å‚È‚¯‚ê‚Î‚È‚è‚Ü‚¹‚ñB");
    unsigned long columnsNumber;
    imagesIS.read((char *)&columnsNumber, sizeof(unsigned long));
    columnsNumber = reverseByteOrder(columnsNumber);
    if (columnsNumber != IMAGE_SIDE_LENGTH) 
        throw describe(__FILE__, "(", __LINE__, "): ", "‰æ‘œ‚Ì•‚Í", IMAGE_SIDE_LENGTH, "‚Å‚È‚¯‚ê‚Î‚È‚è‚Ü‚¹‚ñB");
    mnist->resize(imagesNumber);
    for (auto i = 0; i < imagesNumber; i++) 
        (*mnist)[i] = newInstance<Image>(i);
    for (auto i = 0; i < imagesNumber; i++) 
        imagesIS.read((char *)&(*(*mnist)[i]->getIntensities())[0], IMAGE_AREA);
    
    labelsIS.seekg(8, ios_base::cur);
    for (auto i = 0; i < imagesNumber; i++) {
        unsigned char label;
        labelsIS.read((char *)&label, sizeof(unsigned char));
        (*mnist)[i]->setLabel(label);
    }
    
    return mnist;
}

#endif
