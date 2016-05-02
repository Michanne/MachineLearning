#include "../include/Algorithm.h"

bool Data::parse(std::string filename, char delims, std::vector<Option> options)
{
    std::cout << "Parsing file:\t[" << filename << "]\n";
    std::ifstream file(filename);
    if(file.is_open())
    {
        // Split each line into X and Y values
        std::string line;

        while(std::getline(file, line))
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

        this->options = options;

        return true;
    }

    else return false;
}

LogisticRegressionModel::LogisticRegressionModel()
{
    data.alpha = 1;
    data.lambda = 1;
}

LogisticRegressionModel::~LogisticRegressionModel(){}

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
    return simplify.sum()/((double)X_s.rows);
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
    Mat<double> regularization = initial_theta.map([&](double& x){ x *= data.lambda/X_s.rows; });
    regularization.matrix[0] = 0;
    Mat<double> gradient = (loss.transpose() * X_s).map([&](double& k){ k = learning_rate * k / ((double)X_s.rows); });
    gradient = gradient + regularization;
    return gradient;
}

void LogisticRegressionModel::train(double learning_rate, int iterations){ train(learning_rate, iterations, data.theta); }
void LogisticRegressionModel::train(double learning_rate, int iterations, Mat<double> initial_theta)
{
    data.theta = initial_theta;
    data.alpha = learning_rate;
    std::cout << "Initial cost:\t" << cost(data.theta, data.X, data.Y) << std::endl;

    auto minimization = [&](Mat<double> t,
                            double a,
                            Mat<double> tX,
                            Mat<double> tY)->Mat<double>
                            {
                                if(data.regularized)
                                    return logitRegularizedGradientDescent(t, a, tX, tY);
                                else
                                    return logitGradientDescent(t, a, tX, tY);
                            };

    // Compute optimum theta parameters
    data.theta = descend(
            data,
            iterations, data.printCosts,
            // Minimization function
            minimization,
            // Cost function
            [&](Mat<double> t,
                Mat<double> tX,
                Mat<double> tY)->double
            {
                return cost(t, tX, tY);
            });

    std::cout << "Cost for using: ";
    data.theta.print();
    std::cout << "as theta:\t" << cost(data.theta, data.X, data.Y)  << "\n\n";
}

double LogisticRegressionModel::test(Mat<double> validation_set)
{
    std::cout << "Testing on data set [ " << validation_set.rows << " x " << validation_set.columns << " ]...\n";

    Mat<double> probability = (data.X * data.theta.transpose()).map([&](double& x){ x = sigmoid(x); x = (x >= 0.5 ? 1 : 0);});
    Vec<double> correct;

    for(int i = 0; i < probability.rows; ++i)
    {
        correct.vector.push_back( probability.matrix.at(i) == data.Y.matrix.at(i) );
        ++correct.rows;
    }

    correct.columns = 1;

    std::cout << "Test completed\n";
    return correct.sum()/correct.vector.size();
}

void LogisticRegressionModel::scaleFeatures(Data& d)
{
    std::cout << "Scaling features...\n";
    // Scale features in X data set
    std::vector<double> max_features(d.X.columns, 0);
    std::vector<double> min_features(d.X.columns, 0);

    // Get all extreme features
    for(int i = 0; i < d.X.columns; ++i)
    {
        max_features[i] = d.X.getColumns(i, i+1).matrix[0];
        min_features[i] = d.X.getColumns(i, i+1).matrix[0];
        d.X.getColumns(i, i+1).map([&](double& x){
                                     if(x >= max_features[i]) { max_features[i] = x; }
                                     if(x <= min_features[i]) { min_features[i] = x; }});
    }

    // map scaling
    for(int i = 0; i < d.X.columns; ++i)
    {
        std::cout << "max: " <<  max_features[i] << "\tmin: " << min_features[i] << std::endl;
        for(int j = 0; j < d.X.rows; ++j)
        {
            d.X.matrix.at((j * d.X.columns) + i) /= (max_features[i] - min_features[i]);
        }
    }

    d.scaled = true;
}

void LogisticRegressionModel::expressFeatures(Data& d, int degree, double regularization_term)
{
    // for each column j of X, multiply j to the nth power with each other column to the nth power
    Mat<double> temp;
    temp.rows = d.X.rows;
    for(int i = 0; i < d.X.columns; ++i)
    {
        for(int j = 0; j <= degree; ++j)
        {
            temp.addColumns(d.X.getColumns(i,i+1).map([&](double& x){ x = pow(x, j); }));
        }
        for(int k = 0; k < d.X.columns; ++k)
        {
            if(k == i)
                continue;
            temp.addColumns((d.X.getColumns(k, k+1).toVector() * d.X.getColumns(i, i+1).toVector()).toMatrix());
        }
    }
    d.X = temp;
    d.lambda = regularization_term;
    d.regularized = true;
}

void LogisticRegressionModel::load(Data& data)
{
    for(auto i : data.options)
    {
        parseOption(i, data);
    }
    // Initialize thetas and add 1's columns
    data.theta.addColumns(data.X.columns + 1, 0);
    data.X.addColumns(1, 1);
    data.theta.rows = 1;

    this->data = data;
}

void LogisticRegressionModel::parseOption(Option opt, Data& d)
{
    switch(opt.options.first)
    {
    case Statistics::Options::SCALE:
        scaleFeatures(d);
        break;
    case Statistics::Options::REGULARIZE:
        if(opt.options.second.size() != 2)
            std::cout << "Regularization failed with invalid number of parameters!\n";
        else expressFeatures(d, convertint(opt.options.second.at(0)), convertdouble(opt.options.second.at(1)));
        break;
    case Statistics::Options::COSTS:
        d.printCosts = true;
        break;
    }
}

int LogisticRegressionModel::convertint(std::string s)
{
    return std::strtol(s.c_str(), NULL, 10);
}

double LogisticRegressionModel::convertdouble(std::string s)
{
    return std::strtod(s.c_str(), NULL);
}
