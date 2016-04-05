#include "../include/Algorithm.h"

LogisticRegressionModel::LogisticRegressionModel()
{
    alpha = 1;
    lambda = 1;
}

double LogisticRegressionModel::cost(Mat<double> t, Mat<double> X_s, Mat<double> Y_s)
{
    // J(t) = 1/m * sum( -y_i * log(h(x_i)) - (1-y_i) * log(1 - h(x_i)))
    // m = number of training examples
    // y = correct output
    // x = input
    // h = hypothesis function
    // i = index of training example
    // t = theta
    Mat<double> hypothesis  = (X_s * t.transpose()).map([&](double& x){ x = sigmoid(x);});
    Vec<double> simplify    = (Y_s.map([&](double& x){ x *= -1; }).toVector() * hypothesis.map([&](double& h){ h = log(h); }).toVector())
                            - (Y_s.map([&](double& x){ x = 1-x; }).toVector() * hypothesis.map([&](double& h){ h = log(1 - h);}).toVector());
    return simplify.sum()/((double)X.rows);
}

Mat<double> LogisticRegressionModel::logitGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s)
{
    // dJ(t)/dt_j = a/m * sum( (h(x_i) - y_i) * x_i,j)
    // a = learning rate
    // m = number of training examples
    // x = input
    // y = output
    // i = index of training example
    // t = theta
    Vec<double> hypothesis = ((X_s * initial_theta.transpose()).map([&](double& k){ k = sigmoid(k); })).toVector();
    Mat<double> loss = (hypothesis - Y_s.toVector()).toMatrix();
    Mat<double> gradient = (loss.transpose() * X_s).map([&](double& k){ k = learning_rate * k / ((double)X_s.rows); });
    return gradient;
}

///Calculates the regularized gradient of the hypotheses of @X for @theta
Mat<double> LogisticRegressionModel::logitRegularizedGradientDescent(Mat<double> initial_theta, double learning_rate, Mat<double> X_s, Mat<double> Y_s)
{
    // dJ(t)/dt_j = a/m * sum( (h(x_i) - y_i) * x_i,j + l/m * t_j)
    // a = learning rate
    // m = number of training examples
    // x = input
    // y = output
    // i = index of training example
    // t = theta
    // l = regularization term
    Vec<double> hypothesis = ((X_s * initial_theta.transpose()).map([&](double& k){ k = sigmoid(k); })).toVector();
    Mat<double> loss = (hypothesis - Y_s.toVector()).toMatrix();
    Mat<double> regularization = initial_theta.map([&](double& x){ x *= lambda/X_s.rows; });
    regularization.matrix[0] = 0;
    Mat<double> gradient = (loss.transpose() * X_s).map([&](double& k){ k = learning_rate * k / ((double)X_s.rows); });
    gradient = gradient + regularization;
    return gradient;
}

void LogisticRegressionModel::train(double learning_rate, int iterations, bool cvprint){ train(learning_rate, iterations, cvprint, theta); }
void LogisticRegressionModel::train(double learning_rate, int iterations, bool cvprint, Mat<double> initial_theta, bool regularized)
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

double LogisticRegressionModel::test(Mat<double> validation_set)
{
    std::cout << "Testing on data set [ " << validation_set.rows << " x " << validation_set.columns << " ]...\n";

    Mat<double> probability = (X * theta.transpose()).map([&](double& x){ x = sigmoid(x); x = (x >= 0.5 ? 1 : 0);});
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

bool LogisticRegressionModel::parse(std::string filename, char delims, bool scale)
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
                X.getColumns(i, i+1).map([&](double& x){
                                             if(x >= max_features[i]) { max_features[i] = x; }
                                             if(x <= min_features[i]) { min_features[i] = x; }});
            }

            // map scaling
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
