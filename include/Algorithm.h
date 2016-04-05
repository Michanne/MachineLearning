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
    void addColumns(Mat<T> mat);
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

struct Data
{
    Mat<double> theta;
    Mat<double> X;
    Mat<double> Y;
    Mat<double> costVector;
    double alpha    = 0.0;
    double lambda   = 0.0;
    bool regularized= false;
    bool scaled     = false;
    bool printCosts = false;
};

class Statistics
{
public:
    const double e = 2.7182818284;

    /**
    Regularize  - performs L2-regularization on data features mapped to polynomial terms
    Scale       - performs feature scaling on data set
    Costs       - prints the cost vector for all iterations
    **/
    enum Options
    {
        REGULARIZE,
        SCALE,
        COSTS
    };

    double sigmoid(double x)
    {
        return (1.0 / (1 + exp(-x)));
    }

    template<typename T> double altsigmoid(T x)
    {
        return (1.0 / (1 + pow(e, -x)));
    }

    template <typename Functor1, typename Functor2>
    Mat<double> descend(Data& data,
                        int iterations, bool cvprint,
                        Functor1 minimization_function,
                        Functor2 cost_function)
    {
        Mat<double> initial_theta = data.theta;

        std::cout << "Gradient descent of data using parameters:";
        std::cout   << "\nalpha:\t\t" << data.alpha
                    << "\nlambda:\t\t" << data.lambda
                    << "\nregularization:\t" << (data.regularized ? "on" : "off")
                    << "\nscaling:\t" << (data.scaled ? "on" : "off")
                    << "\niterations:\t" << iterations
                    << "\ninitial theta:\t";
                    initial_theta.print();
        std::cout << "\nComputing...\n";

        Mat<double> costVector(1, iterations, 0);

        double best_cost = 1;
        double min_learning_rate = (1.0/1000000000.0);

        // Converge to local minimum
        for(int i = 0; i < iterations; ++i)
        {
            Mat<double> gradient = minimization_function(initial_theta, data.alpha, data.X, data.Y);

            if(cost_function(initial_theta - gradient, data.X, data.Y) < best_cost)
            {
                best_cost = cost_function(initial_theta - gradient, data.X, data.Y);
                initial_theta = initial_theta - gradient;
                costVector.matrix.at(i) = cost_function(initial_theta, data.X, data.Y);
            }
            else
            {
                if(data.alpha < min_learning_rate)
                {
                    std::cout << "minimum learning rate achieved" << std::endl;
                    break;
                }
                data.alpha /= 10;
                --i;
                std::cout << "learning rate:\t" << data.alpha << std::endl;
            }
        }

        if(cvprint)
            costVector.transpose().print();

        return initial_theta;
    }
};

struct Option
{
    std::pair<Statistics::Options, std::vector<std::string>> options;

    Option(){ options = std::pair<Statistics::Options, std::vector<std::string>>(); }
    Option(Statistics::Options k){ options = std::pair<Statistics::Options, std::vector<std::string>>(k, std::vector<std::string>());}
    Option(Statistics::Options k, std::vector<std::string> v){ options = std::pair<Statistics::Options, std::vector<std::string>>(k, v);}
};

class LogisticRegressionModel : public Statistics
{
public:
    Data data;

    LogisticRegressionModel();
    ~LogisticRegressionModel();

    ///Calculates the cost of using @X modified by @theta as hypotheses for @Y
    double cost(Mat<double> t, Mat<double> X_s, Mat<double> Y_s);

    ///Calculates the gradient of the hypotheses of @X for @theta
    Mat<double> logitGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s);

    ///Calculates the regularized gradient of the hypotheses of @X for @theta
    Mat<double> logitRegularizedGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s);

    ///Iterates through gradient descent to find optimal values for @theta
    void train(double learning_rate, int iterations);
    void train(double learning_rate, int iterations, Mat<double> initial_theta);

    ///Validates the accuracy of @theta using a validation set
    double test(Mat<double> validation_set);

    ///Parses a file
    bool parse(std::string filename, char delims = ' ', std::vector<Option> options = std::vector<Option>());

    ///Option functions
    void parseOption(Option, Data&);
    void scaleFeatures(Data&);
    void expressFeatures(Data&, int, double);
    int convertint(std::string);
    double convertdouble(std::string);
};

class LinearRegressionModel : public Statistics
{

};

#include "../src/Matrix.tcc"
#include "../src/Vector.tcc"

#endif // ALGORITHM_H
