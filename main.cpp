#include "include/Algorithm.h"

int main(int argc, char** argv)
{
    Data data1;
    data1.parse("data/data1.txt", ',', {
            Option(Statistics::SCALE)
            });

    LogisticRegressionModel k;
    k.load(data1);
    k.train(1, 500);
    std::cout   << "Accuracy: "
                << k.test(data1.Y) * 100
                << "%" << std::endl;
    return 0;
}

