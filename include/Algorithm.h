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

    Mat<T>()
    {
        rows = 0;
        columns = 0;
    }

    Mat<T>(int rows, int columns, T value)
    {
        this->rows = rows;
        this->columns = columns;
        for(int i = 0; i < rows*columns; ++i)
            matrix.push_back(value);
    }

    template <size_t r, size_t c>
    Mat<T>(T(&values)[r][c])
    {
        this->rows = r;
        this->columns = c;
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < columns; ++j)
                matrix.push_back(values[i][j]);
    }

    Mat<T>(std::initializer_list<std::initializer_list<T>> values)
    {
        this->rows = values.size();
        int c = 0;
        for(auto i : values)
        {
            for(auto j : i)
            {
                ++c;
                matrix.push_back(j);
            }

            this->columns = c;
        }
    }

    Mat<T> operator*(Mat<T> rhs)
    {
        Mat<T> temp;

        if(rhs.rows == columns)
        {
            temp.rows = rows;
            temp.columns = rhs.columns;
            for(int i = 0; i < rows; ++i)
            {
                for(int p = 0; p < rhs.columns; ++p)
                {
                    T sum = 0;

                    for(int j = 0; j < rhs.rows; ++j)
                    {
                        sum += matrix.at((i * columns) + j) * rhs.matrix.at((j * rhs.columns) + p);
                    }

                    temp.matrix.push_back(sum);
                }
            }
        }

        return temp;
    }

    Mat<T> operator*(Vec<T> rhs)
    {
        Mat<T> temp;

        if(rhs.rows == columns)
        {
            temp.rows = rows;
            temp.columns = rhs.columns;
            for(int i = 0; i < rows; ++i)
            {
                for(int p = 0; p < rhs.columns; ++p)
                {
                    T sum = 0;

                    for(int j = 0; j < rhs.rows; ++j)
                    {
                        sum += matrix.at((i * columns) + j) * rhs.vector.at((j * rhs.columns) + i);
                    }

                    temp.matrix.push_back(sum);
                }
            }
        }

        return temp;
    }

    Mat<T> operator-(Mat<T> rhs)
    {
        Mat<T> temp;
        if(rhs.rows  == rows && rhs.columns == columns)
        {
            temp.rows = rows;
            temp.columns = columns;
            for(int i = 0; i < rows*columns; ++i)
                temp.matrix.push_back(matrix.at(i) - rhs.matrix.at(i));
        }

        return temp;
    }

    Mat<T> operator+(Mat<T> rhs)
    {
        Mat<T> temp;
        if(rhs.rows  == rows && rhs.columns == columns)
        {
            temp.rows = rows;
            temp.columns = columns;
            for(int i = 0; i < rows*columns; ++i)
                temp.matrix.push_back(matrix.at(i) + rhs.matrix.at(i));
        }

        return temp;
    }

    template <typename Functor>
    Mat<T> perform(Functor f)
    {
        Mat<T> temp;
        temp.matrix = this->matrix;
        temp.rows = this->rows;
        temp.columns = this->columns;
        for(unsigned i = 0; i < matrix.size(); ++i)
            f(temp.matrix[i]);

        return temp;
    }

    Mat<T> flatten()
    {
        Mat<T> temp(1, columns, 0);
        for(int j = 0; j < columns; ++j)
        {
            T sum = 0;
            for(int i = 0; i < rows; ++i)
                sum += matrix.at((i * columns) + j);
            temp.matrix.at(j) = sum;
        }

        return temp;
    }

    Mat<T> transpose()
    {
        Mat<T> temp;
        temp.rows = columns;
        temp.columns = rows;
        for(int j = 0; j < columns; ++j)
            for(int i = 0; i < rows; ++i)
                temp.matrix.push_back(matrix.at((i*columns)+j));

        return temp;
    }

    void addColumns(int numColumns, T value)
    {
        for(int i = 0; i < numColumns; ++i)
        {
            int elem = 0;
            int j = 0;
            do
            {
                matrix.insert(matrix.begin()+(j * columns)+elem, value);
                ++elem;
                ++j;
            }while(j < rows);
            ++columns;
        }
    }

    Mat<T> getColumns(int begin = 0, int end = 0)
    {
        Mat<T> temp;
        temp.rows = rows;
        temp.columns = end - begin;
        for(int i = 0; i < rows; ++i)
            for(int j = begin; j < end; ++j)
                temp.matrix.push_back(matrix.at((i * columns) + j));

        return temp;
    }

    Vec<T> toVector()
    {
        Vec<T> temp;
        if(columns == 1)
        {
            temp.rows = rows;
            temp.columns = columns;
            temp.vector = matrix;
        }
        //else exit(-1);
        return temp;
    }

    void print()
    {
        for(int i = 0; i < rows; ++i)
        {
            std::cout << "[ ";
            for(int j = 0; j < columns; ++j)
                std::cout << matrix.at((i * columns) + j) << ", ";
            std::cout << "]" << std::endl;
        }
    }
};

template <typename T>
struct Vec
{
    int rows, columns;
    std::vector<T> vector;

    Vec<T>()
    {
        rows = 0;
        columns = 0;
    }

    Vec<T> operator+(Vec<T> rhs)
    {
        Vec<T> r;
        r.rows = rows;
        r.columns = columns;
        for(unsigned i = 0; i < vector.size(); ++i)
            r.vector.push_back(vector[i] + rhs.vector.at(i));
        return r;
    }

    Vec<T> operator*(Vec<T> rhs)
    {
        Vec<T> r;
        r.rows = rows;
        r.columns = columns;
        for(unsigned i = 0; i < vector.size(); ++i)
            r.vector.push_back(vector.at(i) * rhs.vector.at(i));
        return r;
    }

    Vec<T> operator-(Vec<T> rhs)
    {
        Vec<T> r;
        r.rows = rows;
        r.columns = columns;
        for(unsigned i = 0; i < vector.size(); ++i)
            r.vector.push_back(vector[i] - rhs.vector.at(i));
        return r;
    }

    Vec<T> operator/(Vec<T> rhs)
    {
        Vec<T> r;
        r.rows = rows;
        r.columns = columns;
        for(unsigned i = 0; i < vector.size(); ++i)
            r.vector.push_back(vector[i] / rhs.vector.at(i));
        return r;
    }

    T sum()
    {
        T s = 0;
        for(unsigned i = 0; i < vector.size(); ++i)
        {
            s += vector[i];
        }

        return s;
    }

    Mat<T> toMatrix()
    {
        Mat<T> temp(rows, 1, 0);
        temp.matrix = vector;
        return temp;
    }

    Mat<T> transpose()
    {
        Mat<T> temp;
        temp.rows = columns;
        temp.columns = rows;
        for(unsigned i = 0; i < vector.size(); ++i)
            temp.matrix.push_back(vector[i]);
        return temp;
    }

    template <typename Functor>
    Vec<T> perform(Functor f)
    {
        Vec<T> temp;
        temp.vector = this->vector;
        temp.rows = this->rows;
        temp.columns = this->columns;
        for(unsigned i = 0; i < vector.size(); ++i)
            f(vector[i]);

        return temp;
    }

    void addColumns(int numColumns, T value)
    {
        for(int i = 0; i < numColumns; ++i)
        {
            int elem = 0;
            int j = 0;
            do
            {
                vector.insert(vector.begin()+(j * columns)+elem, value);
                ++elem;
                ++j;
            }while(j < rows);
            ++columns;
        }
    }

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

    LogisticRegressionModel()
    {
        alpha = 1;
        lambda = 1;
    }

    ///Calculates the cost of using @X modified by @theta as hypotheses for @Y
    double cost(Mat<double> t, Mat<double> X_s, Mat<double> Y_s)
    {
        // J(t) = 1/m * sum( -y_i * log(h(x_i)) - (1-y_i) * log(1 - h(x_i)))
        // m = number of training examples
        // y = correct output
        // x = input
        // h = hypothesis function
        // i = index of training example
        // t = theta
        Mat<double> hypothesis  = (X_s * t.transpose()).perform([&](double& x){ x = sigmoid(x);});
        Vec<double> simplify    = (Y_s.perform([&](double& x){ x *= -1; }).toVector() * hypothesis.perform([&](double& h){ h = log(h); }).toVector())
                                - (Y_s.perform([&](double& x){ x = 1-x; }).toVector() * hypothesis.perform([&](double& h){ h = log(1 - h);}).toVector());
        return simplify.sum()/((double)X.rows);
    }

    ///Calculates the gradient of the hypotheses of @X for @theta
    Mat<double> logitGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s)
    {
        // dJ(t)/dt_j = a/m * sum( (h(x_i) - y_i) * x_i,j)
        // a = learning rate
        // m = number of training examples
        // x = input
        // y = output
        // i = index of training example
        // t = theta
        Vec<double> hypothesis = ((X_s * initial_theta.transpose()).perform([&](double& k){ k = sigmoid(k); })).toVector();
        Mat<double> loss = (hypothesis - Y_s.toVector()).toMatrix();
        Mat<double> gradient = (loss.transpose() * X_s).perform([&](double& k){ k = learning_rate * k / ((double)X_s.rows); });
        return gradient;
    }

    ///Calculates the regularized gradient of the hypotheses of @X for @theta
    Mat<double> logitRegularizedGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s)
    {
        // dJ(t)/dt_j = a/m * sum( (h(x_i) - y_i) * x_i,j + l/m * t_j)
        // a = learning rate
        // m = number of training examples
        // x = input
        // y = output
        // i = index of training example
        // t = theta
        // l = regularization term
        Vec<double> hypothesis = ((X_s * initial_theta.transpose()).perform([&](double& k){ k = sigmoid(k); })).toVector();
        Mat<double> loss = (hypothesis - Y_s.toVector()).toMatrix();
        Mat<double> regularization = initial_theta.perform([&](double& x){ x *= lambda/X_s.rows; });
        regularization.matrix[0] = 0;
        Mat<double> gradient = (loss.transpose() * X_s).perform([&](double& k){ k = learning_rate * k / ((double)X_s.rows); });
        gradient = gradient + regularization;
        return gradient;
    }

    ///Iterates through gradient descent to find optimal values for @theta
    void train(double learning_rate, int iterations, bool cvprint){ train(learning_rate, iterations, cvprint, theta); }
    void train(double learning_rate, int iterations, bool cvprint, Mat<double> initial_theta, bool regularized = false)
    {
        theta = initial_theta;
        alpha = learning_rate;
        std::cout << "Initial cost:\t" << cost(theta, X, Y) << std::endl;

        auto minimization = [&](Mat<double> t,
                                double a,
                                Mat<double> tX,
                                Mat<double> tY)->Mat<double>
                                {
                                    if(regularized)
                                        return logitRegularizedGradientDescent(t, a, tX, tY);
                                    else
                                        return logitGradientDescent(t, a, tX, tY);
                                };

        // Compute optimum theta parameters
        theta = descend(theta,
                X, Y,
                alpha, iterations, cvprint,
                // Minimization function
                minimization,
                // Cost function
                [&](Mat<double> t,
                    Mat<double> tX,
                    Mat<double> tY)->double
                {
                    return cost(t, tX, tY);
                });

        std::cout << "Cost for using: \t";
        theta.print();
        std::cout << "as theta:\t" << cost(theta, X, Y)  << "\n\n";
    }

    ///Validates the accuracy of @theta using a validation set
    double test(Mat<double> validation_set)
    {
        std::cout << "Testing on data set [ " << validation_set.rows << " x " << validation_set.columns << " ]...\n";

        Mat<double> probability = (X * theta.transpose()).perform([&](double& x){ x = sigmoid(x); x = (x >= 0.5 ? 1 : 0);});
        Vec<double> correct;

        for(int i = 0; i < probability.rows; ++i)
        {
            correct.vector.push_back( probability.matrix.at(i) == Y.matrix.at(i) );
            ++correct.rows;
        }

        correct.columns = 1;

        std::cout << "Test completed\n";
        return correct.sum()/correct.vector.size();
    }

    ///Parses a file
    bool parse(std::string filename, char delims = ' ', bool scale = false)
    {
        std::cout << "Parsing file [" << filename << "]...\n";
        std::ifstream data(filename);
        if(data.is_open())
        {
            // Split each line into X and Y values
            std::string line;

            while(std::getline(data, line))
            {
                // All columns before the last become X valued
                std::stringstream l(line);
                std::vector<std::string> tokens;
                std::string s;
                while(std::getline(l, s, delims))
                {
                    tokens.push_back(s);
                }
                for(unsigned i = 0; i < tokens.size()-1; ++i)
                {
                    X.matrix.push_back(std::stod(tokens[i]));
                }
                Y.matrix.push_back(std::stod(tokens[tokens.size()-1]));
                Y.columns = 1;
                X.columns = tokens.size()-1;
                ++X.rows;
                tokens.clear();
            }

            Y.rows = X.rows;

            if(scale)
            {
                std::cout << "Scaling features...\n";
                // Scale features in X data set
                std::vector<double> max_features(X.columns, 0);
                std::vector<double> min_features(X.columns, 0);

                // Get all extreme features
                for(int i = 0; i < X.columns; ++i)
                {
                    max_features[i] = X.getColumns(i, i+1).matrix[0];
                    min_features[i] = X.getColumns(i, i+1).matrix[0];
                    X.getColumns(i, i+1).perform([&](double& x){
                                                 if(x >= max_features[i]) { max_features[i] = x; }
                                                 if(x <= min_features[i]) { min_features[i] = x; }});
                }

                // Perform scaling
                for(int i = 0; i < X.columns; ++i)
                {
                    std::cout << "max: " <<  max_features[i] << "\tmin: " << min_features[i] << std::endl;
                    for(int j = 0; j < X.rows; ++j)
                    {
                        X.matrix.at((j * X.columns) + i) /= (max_features[i] - min_features[i]);
                    }
                }
            }

            // Initialize thetas and add 1's columns
            theta.addColumns(X.columns + 1, 0);
            X.addColumns(1, 1);
            theta.rows = 1;

            return true;
        }

        else return false;
    }
};

class LinearRegressionModel : public Statistics
{

};

#endif // ALGORITHM_H
