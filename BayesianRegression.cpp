#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <regex>
#include <random>
#include <utility>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <assert.h>
#include <sstream>
using namespace std;

class CSVReader {
private:
	string fileName;
public:
	CSVReader(string fileName) : fileName(fileName) { }
	vector<vector<double>> getData();
};

vector<vector<double>>CSVReader::getData() {
        ifstream file_object;
		file_object.open(fileName, ifstream::in);
        if (!file_object.is_open()) {
            // Handle file opening error
            std::cerr << "Error: Unable to open file." << std::endl;
            return std::vector<std::vector<double>>();
        }

        std::vector<std::vector<double>> data_double;
        std::string line;
        std::string delimiter = ",";

        while (getline(file_object, line)) {
            std::vector<std::string> line_of_file;
            std::istringstream line_stream(line);

            // Split the string using the specified delimiter
            std::string token;
            while (getline(line_stream, token, delimiter[0])) {
                line_of_file.push_back(token);
            }

            // Convert the string values to doubles
            std::vector<double> line_of_file_double;
            for (const std::string& s : line_of_file) {
                try {
                    line_of_file_double.push_back(std::stod(s));
                } catch (const std::invalid_argument& e) {
                    // Handle conversion errors
                    std::cerr << "Error: Invalid double conversion in line." << std::endl;
                    line_of_file_double.push_back(0.0); // Default value
                }
            }

            data_double.push_back(line_of_file_double);
        }

        file_object.close();
		//cout<<"Exited GetData"<<endl;
        return data_double;
    }


class Regression {
private:
	vector<double> m_x, m_y;
	pair<double, double> gradient(double& slope, double& intercept);
	static double loss(const vector<double>& y_hat, const vector<double>& y_true);
public:
	Regression(vector<double>& x, vector<double>& y) : m_x(x), m_y(y) {}
	pair<double, double> train(int n_iter, double lr);
	vector<double> predict(const double& s, const double& i, const vector<double>& inp_x);
};


pair<double, double> Regression::train(int n_iter, double lr) {
	/* Train the regression with gradient Descent*/
	mt19937 rng;
	rng.seed(random_device()());
	uniform_real_distribution<double> gen(0, 1);	
	double slope_ = gen(rng);
	double intercept_ = -1.0 * gen(rng);
	double dslope_, dintercept_;
	pair<double, double> final_slope_intercept;
	for (int epoch = 0; epoch <= n_iter; epoch++) {
		pair<double, double> dslope_intercept = gradient(slope_, intercept_);
		// gradients of the slope and intercept
		dslope_ = dslope_intercept.first;
		dintercept_ = dslope_intercept.second;
		//cout << "Slope: " << dslope_ << " Intercept grad" << dintercept_ << "\n";
		// update the slope and intercept
		slope_ -= lr * dslope_;
		intercept_ -= lr * dintercept_;
		// display the slope and intercept
		if (epoch % 1000 == 0) {
			vector<double> y_hat = predict(slope_, intercept_, m_x);
			double loss_value = loss(y_hat, m_y);
			//cout << "Losss= " << loss_value << endl;
			//D(loss_value);
		}
	}
	// assign the value to slope and intercept
	final_slope_intercept.first = slope_;
	final_slope_intercept.second = intercept_;
	//cout<<"Exited Train"<<endl;
	return final_slope_intercept;
}

vector<double> Regression::predict(const double& s, const double& i, const vector<double>& inp_x) {
	/* s: (double) slope
	i : (double) intercept
	inp_x: the vector<double> of input variable to be predicted
	*/
	//cout<<"Enterd Regression Predict"<<endl;
	vector<double> y(inp_x.size());
	for(size_t ind = 0; ind < y.size(); ind++){
		double temp = i + s * inp_x[ind];
		y[ind] = temp;
	}
	//cout<<"Exited Regression Predict"<<endl;
	return y;
}

double Regression::loss(const vector<double>& y_hat, const vector<double>& y_true) {
	/* Estimate the mean-squared loss function
	y_hat : vector<double> of the predicted y
	y_true: vector<double> of the actual y-value
	*/
	//cout<<"Entered Loss Value"<<endl;
	vector<double> temp_diff;
	// take the difference of two vectors, add the difference to temp_diff vector
	transform(y_hat.begin(), y_hat.end(), y_true.begin(), inserter(temp_diff, temp_diff.begin()), minus<double>());
	double loss_value = inner_product(temp_diff.begin(), temp_diff.end(), temp_diff.begin(), 0);
	loss_value /= temp_diff.size();
	//cout<<"Exited Loass Value"<<endl;
	return loss_value;
}

pair<double, double> Regression::gradient(double& s, double& i) {
	/* Estimates the grdients of slope and intercept parameters
	Argumets
	s : double slope parameters
	i : double intercept parameter
	Returns:
	pair<double, double>, returns a pair of gradients for slope and intercept
	*/
	//cout<<"Entered Regression Gradient"<<endl;
	auto m = m_x.size();
	vector<double> y_hat = predict(s, i, m_x);
	vector<double> err;
	transform(y_hat.begin(), y_hat.end(), m_y.begin(), inserter(err, err.begin()), minus<double>());
	// gradient of slope
	double dslope =  1.0 / m * inner_product(err.begin(), err.end(), m_x.begin(), 0.0);
	//gradient of intercept
	double dintercept = 1.0 / m * accumulate(err.begin(), err.end(), 0.0);
	pair<double, double> dslope_intercept(dslope, dintercept);
	//cout<<"Exited Regression Gradient"<<endl;
	return dslope_intercept;
}


void log_transform(vector<double>& y) {
	/* tranforms the y coordinated if the provided y vector*/
	//cout<<"Entered Transform"<<endl;
	transform(y.begin(), y.end(), y.begin(), [](const double& s){return log(s);});
	//cout<<"Exited Transform"<<endl;
	
}


int writeFile(const pair<double, double>& coefficient) {
	//cout<<"Entered Writefile"<<endl;
	ofstream write_my_file;
	write_my_file.open("output.txt");
	write_my_file << coefficient.first << "\n";
	write_my_file << coefficient.second << "\n";
	write_my_file.close();
	//cout<<"Exited writeFile"<<endl;
	return 0;
}

class Gibbs_Sampler {
private:
	vector<double> m_x;
	vector<double> m_y;

	double sample_beta0(double beta1, double tau, double mu_0, double tau_0, mt19937& generator);
	double sample_beta1(double beta0, double tau, double mu_1, double tau_1, mt19937& generator);
	double sample_tau(double beta_0, double beta_1, double alpha, double beta, mt19937& generator);
public:
	Gibbs_Sampler(vector<double> x, vector<double> y) {
		assert(x.size() == y.size());
		m_x = x;
		m_y = y;
	}
	unordered_map<string, vector<double>> sample(int n_iter, unordered_map<string, double>& init, unordered_map<string, double>& hyper);
	static vector<double> predict(double beta0, double beta1, const vector<double>& x);
	static double predict(double beta0, double beta1, double x);
	static unordered_map<string, unordered_map<string, double>> summary(unordered_map<string, vector<double>> trace, int n_burn);
};


unordered_map<string, unordered_map<string, double>> Gibbs_Sampler::summary(unordered_map<string, vector<double>> trace, int n_burn) {
	/* Arguement:
	unordered_map<string, vector<double>: trace
	trace["beta0"]:  vector<double> of length n, trace of intercept
	trace["beta1":	vector<double> of length n, trace of slope+
	trace["tau"]:	vector<double> of length n, trace of tau
	n_burn: integer type , Burning sample  size 
	*/
	// +++++++++++++++++++++++++++++++++++++++++++++++
	//+++++++++++++++++++++++++++++++++++++++++
    // mean for beta0
	//cout<<"Entered Unorderd Summary"<<endl;
	double sum_beta0 = accumulate(trace["beta0"].begin() + n_burn, trace["beta0"].end(), 0.0);
	double mu_beta0 = sum_beta0 / (trace["beta0"].size() - n_burn);
	// standard deviation for beta0
	vector<double> difference_beta0(trace["beta0"].size(),0.0);
	transform(trace["beta0"].begin() + n_burn, trace["beta0"].end(), difference_beta0.begin(), bind(minus<double>(), placeholders::_1, mu_beta0));
	double sq_sum_beta0 = inner_product(difference_beta0.begin(), difference_beta0.end(), difference_beta0.begin(), 0.0);
	double std_beta0 = sqrt(sq_sum_beta0 / (trace["beta0"].size() - n_burn));	
	
	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// mean for beta1
	double sum_beta1 = accumulate(trace["beta1"].begin() + n_burn, trace["beta1"].end(), 0.0);
	double mu_beta1 = sum_beta1 / (trace["beta1"].size() - n_burn);
	// standard deviation for beta1
	vector<double> difference_beta1(trace["beta1"].size(), 0.0);
	transform(trace["beta1"].begin() + n_burn, trace["beta1"].end(), difference_beta1.begin(), bind(minus<double>(), placeholders::_1, mu_beta1));
	double sq_sum_beta1 = inner_product(difference_beta1.begin(), difference_beta1.end(), difference_beta1.begin(), 0.0);
	double std_beta1 = sqrt(sq_sum_beta1 / (trace["beta1"].size() - n_burn));	
	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// mean for tau
	double sum_tau = accumulate(trace["tau"].begin() + n_burn, trace["tau"].end(), 0.0);
	double mu_tau = sum_tau / (trace["tau"].size() - n_burn);
	// standard deviation for beta1
	vector<double> difference_tau(trace["tau"].size(), 0.0);
	transform(trace["tau"].begin() + n_burn, trace["tau"].end(), difference_tau.begin(), bind(minus<double>(), placeholders::_1, mu_tau));
	double sq_sum_tau = inner_product(difference_tau.begin(), difference_tau.end(), difference_tau.begin(), 0.0);
	double std_tau = sqrt(sq_sum_tau / (trace["tau"].size() - n_burn));
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     
	//#ifdef _DEBUG
		cout << "Mean Beta0 >> " << mu_beta0 << " Standard deviation beta0 " << std_beta0 << "\n";
		cout << "Mean Beta1 >> " << mu_beta1 << " Standard deviation beta1 " << std_beta1 << "\n";
		cout << "Mean Tau >> " << mu_tau << " Standard deviation Tau " << std_tau << "\n";
	//#endif
	// save the summary in nested_pair
	unordered_map<string, unordered_map<string, double>> result
	{  {"beta0", unordered_map<string, double>{{"mean", mu_beta0}, {"standard deviation", std_beta0}}}
	  ,{ "beta1", unordered_map<string, double>{{"mean", mu_beta1}, {"standard deviation", std_beta1 }}}
	  ,{ "tau", unordered_map<string, double>{{"mean", mu_tau}, {"standard deviation", std_tau }}}
	};

	//cout<<"Exited Unordered Summary "<<endl;
	return result;

}


unordered_map<string, vector<double> > Gibbs_Sampler::sample(int n_iter, unordered_map<string, double>& init, unordered_map<string, double>& hyper) {
	// load the initial values
	//cout<<"Entered Sample"<<endl;
	double beta0 = init["beta0"];
	double beta1 = init["beta1"];
	double tau = init["tau"];
	// initalize unordered with vectors of length n_iter
	unordered_map<string, vector<double>> trace
	{    
		{"beta0", vector<double>(n_iter, 0.0)}
		,{"beta1", vector<double>(n_iter, 0.0)}
		,{"tau", vector<double>(n_iter, 0.0) }
	};
	
	//Random seed
	random_device rng;
	// Initialize Mersenne Twister psuedo-random number generator
	mt19937 gen(rng());
	

	for (int i = 0; i < n_iter; i++) {
		// sample intercept and update value from previous iteration
		 beta0 = sample_beta0(beta1, tau, hyper["mu_0"], hyper["tau_0"], gen);
		 trace["beta0"][i] = beta0;
		//sample slope and update
		 beta1 = sample_beta1(beta0, tau, hyper["mu_1"], hyper["tau_1"], gen);
		 trace["beta1"][i] = beta1;
		// sample precision and update
		 tau = sample_tau(beta0, beta1, hyper["alpha"], hyper["beta"], gen);
		 trace["tau"][i] = tau;

		//  if (i % 100 == 0) {
		// 	 cout << "Intercept>>  " << beta0 << "  Slope>> " << beta1 << "  Precision>> " << tau << "\n";
		//  }
		
		
	}
	//cout<<"Exited Sample"<<endl;
	return trace;

}

vector<double> Gibbs_Sampler::predict(double beta0, double beta1, const vector<double>& x) {
	/* prediction fuunction for vector*/
	//cout<<"Entered V predict"<<endl;
	vector<double> y_hat;
	for (const auto& val : x) {
		y_hat.emplace_back(val * beta1 + beta0);
	}
	//cout<<"Exited V predict"<<endl;
	return y_hat;
}

double Gibbs_Sampler::predict(double beta0, double beta1, double x) {
	/* overloadded prediction function*/
	//cout<<"Exited predict"<<endl;
	return beta0 + beta1 * x;
}



double Gibbs_Sampler::sample_beta0(double beta1, double tau, double mu_0, double tau_0, mt19937& de) {
	//cout<<"Entered sample_beta0"<<endl;
	auto n{ m_x.size() };
	vector<double> mx_y;	
	vector<double> beta_x;
	transform(m_x.begin(), m_x.end(), inserter(beta_x, beta_x.begin()), bind(multiplies<double>(), placeholders::_1, beta1));
	transform(m_y.begin(), m_y.end(), beta_x.begin(), beta_x.begin(), minus <double>());
	double sum_y_x_beta = accumulate(beta_x.begin(), beta_x.end(), 0.0);	
	double precision = tau_0 + tau * n;
	double mean = (tau_0  * mu_0 + tau * sum_y_x_beta) / precision;
	normal_distribution<double> nd(mean, 1 / sqrt(precision)); // normal random generator, mean followed by std
	//cout<<"Exited sample_beta0"<<endl;
	return nd(de);

}

double Gibbs_Sampler::sample_beta1(double beta0, double tau, double mu_1, double tau_1, mt19937& de) {
	//cout<<"Entered sample_beta1"<<endl;
	auto n{m_x.size() };
	vector<double> temp_y_beta;
	transform(m_y.begin(), m_y.end(), inserter(temp_y_beta, temp_y_beta.begin()), bind(minus<double>(), placeholders::_1, beta0));
	double sum = inner_product(temp_y_beta.begin(), temp_y_beta.end(), m_x.begin(), 0.0);
	double mean = tau_1 * mu_1 + tau * sum;
	double precision = tau_1 + tau * inner_product(m_x.begin(), m_x.end(), m_x.begin(), 0.0);
	mean /= precision;
	// generate random number
	normal_distribution<double> nd(mean, 1 / sqrt(precision));
	//cout<<"Exited sample_beta1"<<endl;
	return nd(de);


}

double Gibbs_Sampler::sample_tau(double beta_0, double beta_1, double alpha, double beta, mt19937& de) {
	/* samples the precision parameters*/
	//cout<<"Entered sample_tau"<<endl;
	auto n{ m_y.size() };
	double shape = alpha + (double) n / 2;
	auto pred_y = Gibbs_Sampler::predict(beta_0, beta_1, m_x);
	transform(m_y.begin(), m_y.end(), pred_y.begin(), pred_y.begin(), minus<double>());
	double sum_squared_error = inner_product(pred_y.begin(), pred_y.end(), pred_y.begin(), 0.0);
	double rate = beta + sum_squared_error / 2.0;
	gamma_distribution<double> gde(shape, 1/rate);
	//cout<<"Exited sample_tau"<<endl;
	return gde(de);
}



vector<vector<double>> sample_predict_posterior(unordered_map<string, unordered_map<string, double>>& summary, int number_of_sample, const vector<double>& x) {
	//cout<<"Entered sample predict posterior"<<endl;
	auto mu_beta0 = summary["beta0"]["mean"];
	auto std_beta0 = summary["beta0"]["standard deviation"];

	auto mu_beta1 = summary["beta1"]["mean"];
	auto std_beta1 = summary["beta1"]["standard deviation"];

	auto mu_tau = summary["tau"]["mean"];
	auto std_tau = summary["tau"]["standard deviation"];
	
	vector<vector<double>> y_hat;
	// reserve space
	y_hat.reserve(x.size());
	random_device ringer;
	// Initialize Mersenne Twister psuedo-random number generator
	mt19937 generator(ringer());
	for (auto& x_value : x) {
		vector<double> temp_pred(number_of_sample);
		for (int i = 0; i < number_of_sample; i++) {
			normal_distribution<double> beta0_nrand(mu_beta0, std_beta0);
			normal_distribution<double> beta1_nrand(mu_beta1, std_beta1);
			normal_distribution<double> tau_nrand(mu_tau, std_tau);			
			double beta0 = beta0_nrand(generator);
			double beta1 = beta1_nrand(generator);
			double tau = tau_nrand(generator);

			double z_mean = beta0 + beta1 * x_value;
			normal_distribution<double> z_dist(z_mean, 1 / sqrt(tau));
			//===========================================
			double z = z_dist(generator);
			//============================================
			temp_pred[i] = exp(z);			
		}
		y_hat.emplace_back(temp_pred);
	}

	//cout<<"Exited sample predict posterior"<<endl;
	return y_hat;		
 }

int main()
{ 
	CSVReader reader("house_prices_dataset.csv");
	const auto output = reader.getData();
	
	vector<double> x;
	vector<double> y;
	for (const auto& val : output) {
		x.emplace_back(val[0]);
		y.emplace_back(val[1]);
	}
	//cout <<"size of x"<< x.size() << "size of y"<< y.size()<<"\n";	
	log_transform(y);

	// initialize the hyper-priors for intercept N(0,1), slope N(0,1), and standard-deviation Gamma(alpha, beta)
	unordered_map<string, double> hyper_parameters{
		 {"mu_0", 0.0}
		,{"tau_0",1.0}
		,{"mu_1", 0.0}
		,{"tau_1", 1.0}
		,{"alpha", 2.0}
		,{"beta", 1.0}
	};
	// seed for random number generation
	// initialize intercept, slope and standard deviation with random number
	unordered_map<string, double> initial_values{
		{"beta0", 0.0}
		,{"beta1", -0.05}
		,{"tau",2.0}
	};
	// parameters for mcmc sampling
	int number_of_iteration = pow(10,4);
	// define the object
	Gibbs_Sampler mcmc_object(x, y);
	// call the sample method which outputs trace
	auto trace = mcmc_object.sample(number_of_iteration, initial_values, hyper_parameters);
	auto summary_result = Gibbs_Sampler::summary(trace,number_of_iteration/2);
	
	// get the prediction



	vector<double> xx{2231.88, 3455.78,4567.89};
	auto y_prediction = sample_predict_posterior(summary_result, 500,xx);


	//cout<<"Entered output csv file"<<endl;
	ofstream output_yhat;
	output_yhat.open("Results.csv");
	for (const auto& temp : y_prediction) {
		for (const auto& value : temp) {
			// write each prediction for all beta comma seperate in a single line
			output_yhat << value << ",";
		}
		//
		output_yhat << "\n";
	}
	output_yhat.close();
	//cout<<"Exited output csv File"<<endl;

	return 0;
}

