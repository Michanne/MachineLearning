#include "include/Algorithm.h"

int main(int argc, char** argv)
{
    LogisticRegressionModel k;
    k.parse("data/data2.txt", ',', {
            Option(Statistics::REGULARIZE, {"6", "0.1"})
            });
    k.train(1, 50);
    std::cout   << "Accuracy: "
                << k.test(k.data.Y) * 100
                << "%" << std::endl;
    return 0;
}

