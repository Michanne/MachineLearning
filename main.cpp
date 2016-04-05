#include "include/Algorithm.h"

int main(int argc, char** argv)
{
    LogisticRegressionModel k;
    k.parse("data/data1.txt", ',', true);
    k.train(1, 1000, false, Mat<double>({{0, 0, 0}}));
    std::cout << "Accuracy: " << k.test(k.Y) * 100 << "%" << std::endl;
    return 0;
}

