#ifndef ALGORITHM_H
#define ALGORITHM_H
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <initializer_list>

template <typename T>
struct Vec;

template <typename T>
struct Mat
{
    int rows, columns;
    std::vector<T> matrix;

    ///Constructors
    Mat<T>();
    Mat<T>(int rows, int columns, T value);
    template <size_t r, size_t c>
    Mat<T>(T(&values)[r][c]);
    Mat<T>(std::initializer_list<std::initializer_list<T>> values);

    ///Overloaded Operators
    Mat<T> operator*(Mat<T> rhs);
    Mat<T> operator*(Vec<T> rhs);
    Mat<T> operator-(Mat<T> rhs);
    Mat<T> operator+(Mat<T> rhs);

    ///Utilities
    template <typename Functor>
    Mat<T> map(Functor f);
    Mat<T> flatten();
    Mat<T> transpose();
    void addColumns(int numColumns, T value);
    Mat<T> getColumns(int begin = 0, int end = 0);
    Vec<T> toVector();
    void print();
};

template <typename T>
struct Vec
{
    int rows, columns;
    std::vector<T> vector;

    ///Constructors
    Vec<T>();

    ///Overloaded Operators
    Vec<T> operator+(Vec<T> rhs);
    Vec<T> operator*(Vec<T> rhs);
    Vec<T> operator-(Vec<T> rhs);
    Vec<T> operator/(Vec<T> rhs);

    ///Utilities
    T sum();
    Mat<T> toMatrix();
    Mat<T> transpose();
    template <typename Functor>
    Vec<T> map(Functor f);
    void addColumns(int numColumns, T value);
};

class Statistics
{
public:
    const double e = 2.7182818284;

    double sigmoid(double x)
    {
        return (1.0 / (1 + exp(-x)));
    }

    template<typename T> double altsigmoid(T x)
    {
        return (1.0 / (1 + pow(e, -x)));
    }

    template <typename Functor1, typename Functor2>
    Mat<double> descend(Mat<double> initial_theta,
                        Mat<double> X_s, Mat<double> Y_s,
                        double learning_rate, int iterations, bool cvprint,
                        Functor1 minimization_function,
                        Functor2 cost_function)
    {
        std::cout << "Gradient descent of data using parameters:\n";
        std::cout << "alpha:\t" << learning_rate << "\niterations:\t" << iterations << "\ninitial theta:\t";
        initial_theta.print();
        std::cout << "Computing...\n";

        Mat<double> costVector(1, iterations, 0);

        double best_cost = 10000000;
        double min_learning_rate = (1.0/1000000000.0);

        // Converge to local minimum
        for(int i = 0; i < iterations; ++i)
        {
            Mat<double> gradient = minimization_function(initial_theta, learning_rate, X_s, Y_s);

            if(cost_function(initial_theta - gradient, X_s, Y_s) < best_cost)
            {
                best_cost = cost_function(initial_theta - gradient, X_s, Y_s);
                initial_theta = initial_theta - gradient;
                costVector.matrix.at(i) = cost_function(initial_theta, X_s, Y_s);
            }
            else
            {
                if(learning_rate < min_learning_rate)
                {
                    std::cout << "minimum learning rate achieved" << std::endl;
                    break;
                }
                learning_rate /= 10;
                --i;
                std::cout << "learning rate:\t" << learning_rate << std::endl;
            }
        }

        if(cvprint)
            costVector.transpose().print();

        return initial_theta;
    }
};

class LogisticRegressionModel : public Statistics
{
public:
    Mat<double> theta;
    Mat<double> X;
    Mat<double> Y;
    double alpha;
    double lambda;

    LogisticRegressionModel();

    ///Calculates the cost of using @X modified by @theta as hypotheses for @Y
    double cost(Mat<double> t, Mat<double> X_s, Mat<double> Y_s);

    ///Calculates the gradient of the hypotheses of @X for @theta
    Mat<double> logitGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s);

    ///Calculates the regularized gradient of the hypotheses of @X for @theta
    Mat<double> logitRegularizedGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s);

    ///Iterates through gradient descent to find optimal values for @theta
    void train(double learning_rate, int iterations, bool cvprint);
    void train(double learning_rate, int iterations, bool cvprint, Mat<double> initial_theta, bool regularized = false);

    ///Validates the accuracy of @theta using a validation set
    double test(Mat<double> validation_set);

    ///Parses a file
    bool parse(std::string filename, char delims = ' ', bool scale = false);
};

class LinearRegressionModel : public Statistics
{

};

#include "../src/Matrix.tcc"
#include "../src/Vector.tcc"

#endif // ALGORITHM_H
