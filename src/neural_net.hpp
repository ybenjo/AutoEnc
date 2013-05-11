// Neural network
// All notations : http://www.stanford.edu/class/cs294a/sparseAutoencoder.pdf

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <boost/random.hpp>

struct myeq : std::binary_function<std::pair<int, int>, std::pair<int, int>, bool>{
  bool operator() (const std::pair<int, int> & x, const std::pair<int, int> & y) const{
    return x.first == y.first && x.second == y.second;
  }
};

struct myhash : std::unary_function<std::pair<int, int>, size_t>{
private:
  const std::hash<int> h_int;
public:
  myhash() : h_int() {}
  size_t operator()(const std::pair<int, int> & p) const{
    size_t seed = h_int(p.first);
    return h_int(p.second) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }
};

// Wrapper of Boost::random
// Ref : http://d.hatena.ne.jp/n_shuyo/20100407/random
template<class D, class G = boost::mt19937>
class Rand {
  G gen_;
  D dst_;
  boost::variate_generator<G, D> rand_;
public:
  Rand() : gen_(static_cast<unsigned long>(time(0))), rand_(gen_, dst_) {}
  template<typename T1>
  Rand(T1 a1) : gen_(static_cast<unsigned long>(time(0))), dst_(a1), rand_(gen_, dst_) {}
  template<typename T1, typename T2>
  Rand(T1 a1, T2 a2) : gen_(static_cast<unsigned long>(time(0))), dst_(a1, a2), rand_(gen_, dst_) {}

  typename D::result_type operator()() { return rand_(); }
};

class NeuralNet{
public:
  NeuralNet(int size, double alpha, double beta, double lambda, double rho){
    // Initialize parameters
    // Unit size of hidden layer
    _size = size;

    // Learning rate parameter
    _alpha = alpha;
    // KL-regularizer weight parameter
    _beta = beta;
    // Weight decay parameter(L2-regularizer weight)
    _lambda = lambda;
    // Sparsity parameter
    _rho = rho;
  }

  double sigmoid(double x){
    return 1 / (1 + exp(-x));
  };

  // x : <<feature_id, val>, ..., <feature_id, val>, >
  // y : <<answer_id, val>, ..., <answer_id, val>, >
  double set_data(const std::vector<std::pair<int, double> >& x, const std::vector<std::pair<int, double> >& y){
    _all_x.push_back(x);
    _all_y.push_back(y);
  };

  // Train oneself(x = f(x))
  // for autoencoding
  double set_data(const std::vector<std::pair<int, double> >& x){
    set_data(x, x);
  };

  // Initialize _w
  void initialize_w(){
    std::vector<std::vector<std::pair<int, double> > >::iterator iter_1;
    std::vector<std::pair<int, double> >::iterator iter_2;

    // Search x's dimension
    for(iter_1 = _all_x.begin(); iter_1 != _all_x.end(); ++iter_1){
      std::vector<std::pair<int, double> > vec = (*iter_1);
      for(iter_2 = vec.begin(); iter_2 != vec.end(); ++iter_2){
        int feature_id = (*iter_2).first;
	if(feature_id > _size_x) _size_x = feature_id;
      }
    }

    // Search y's dimension
    for(iter_1 = _all_y.begin(); iter_1 != _all_y.end(); ++iter_1){
      for(iter_2 = (*iter_1).begin(); iter_2 != (*iter_1).end(); ++iter_2){
        int answer_id = (*iter_2).first;
	if(answer_id > _size_y) _size_y = answer_id;
      }
    }

    // inclement for loop's condition
    ++_size_x;
    ++_size_y;
    
    // Initialize _w
    // w[layer][from][to]

    // Initialize N(0, 0.01 ^ 2)
    Rand<boost::normal_distribution<> > rnorm(0, 0.0001);

    // Weights of [x] => [hidden layer]
    std::vector<std::vector<double> > first_layer_weight;
    for(int from = 0; from < _size_x; ++from){
      std::vector<double> w;
      for(int to = 0; to < _size; ++to){
	double val = rnorm();
	w.push_back(val);
      }
      first_layer_weight.push_back(w);
    }

    // Weights of [hidden layer] => [y]
    std::vector<std::vector<double> > second_layer_weight;
    for(int from = 0; from < _size; ++from){
      std::vector<double> w;
      for(int to = 0; to < _size_y; ++to){
	double val = rnorm();
	w.push_back(val);
      }
      second_layer_weight.push_back(w);
    }

    _w.push_back(first_layer_weight);
    _w.push_back(second_layer_weight);
  };

  // initialize _b
  // _b[<layer, node_id>]
  void initialize_b(){
    // initialize N(0, 0.01 ^ 2)
    Rand<boost::normal_distribution<> > rnorm(0, 0.0001);

    // first layer
    // [x] => [hidden layer]
    for(int node_id_to = 0; node_id_to < _size; ++node_id_to){
      double val = rnorm();
      _b[std::make_pair(0, node_id_to)] = val;
    }

    // second layer
    // [hidden layer] => [y]
    for(int node_id_to = 0; node_id_to < _size_y; ++node_id_to){
      double val = rnorm();
      _b[std::make_pair(1, node_id_to)] = val;
    }
  };

  // Set min/max of y and Rewrite _all_y
  void initialize_y(){
    std::vector<std::vector<std::pair<int, double> > >::iterator iter_1;
    std::vector<std::pair<int, double> >::iterator iter_2;

    // set min/max
    for(iter_1 = _all_y.begin(); iter_1 != _all_y.end(); ++iter_1){
      std::vector<std::pair<int, double> > vec = (*iter_1);
      for(iter_2 = vec.begin(); iter_2 != vec.end(); ++iter_2){
	int answer_id = (*iter_2).first;
        double val = (*iter_2).second;
	if(_min_y[answer_id] > val) _min_y[answer_id] = val;
	if(_max_y[answer_id] < val) _max_y[answer_id] = val;
      }
    }

    // Rewrite _all_y
    for(int i = 0; i < _all_y.size(); ++i){
      for(int j = 0; j < (_all_y.at(i)).size(); ++j){
	std::pair<int, double> prev_elem = (_all_y.at(i)).at(j);
	int answer_id = prev_elem.first;
	double prev_val = prev_elem.second;

	double min = _min_y[answer_id];
	double max = _max_y[answer_id];
	// Scaling prev_val to [0, 1]
	double new_val = (prev_val - min) / (max - min);

	std::pair<int, double> new_elem = std::make_pair(answer_id, new_val);
	(_all_y.at(i)).at(j) = new_elem;
      }
    }
  };

  void initialize(){
    initialize_w();
    initialize_b();
    initialize_y();
  };

  // forward
  void forward(){
    // clear _a
    _a.clear();

    for(int data_id = 0; data_id < _all_x.size(); ++data_id){

      // a[<layer, node_id>]
      std::unordered_map<std::pair<int, int>, double, myhash, myeq> a;

      // z[<layer, node_id>]
      // a[<layer, node_id>] = sigmoid(z[<layer, node_id>]);
      std::unordered_map<std::pair<int, int>, double, myhash, myeq> z;

      // First, add bias factor
      // [b] => [hidden layer]
      for(int node_id_to = 0; node_id_to < _size; ++ node_id_to){
	z[std::make_pair(0, node_id_to)] = _b[std::make_pair(0, node_id_to)];
      }
      // [b] => [y]
      for(int node_id_to = 0; node_id_to < _size_y; ++ node_id_to){
	z[std::make_pair(1, node_id_to)] = _b[std::make_pair(1, node_id_to)];
      }

      // [x] => [hidden layer]
      // z += w * x
      std::vector<std::pair<int, double> >::iterator iter;
      std::vector<std::pair<int, double> > vec = _all_x.at(data_id);
      for(iter = vec.begin(); iter != vec.end(); ++iter){
	std::pair<int, double> elem = (*iter);
	int node_id_from = elem.first;
	double val = elem.second;

	for(int node_id_to = 0; node_id_to < _size; ++ node_id_to){
	  z[std::make_pair(0, node_id_to)] += _w.at(0).at(node_id_from).at(node_id_to) * val;
	}
      }

      // update a = sigmoid(z)
      for(int node_id = 0; node_id < _size; ++ node_id){
	std::pair<int, int> key = std::make_pair(0, node_id);
	a[key] = sigmoid(z[key]);
      }

      // [hidden layer] => [y]
      for(int node_id_from = 0; node_id_from < _size; ++ node_id_from){
	for(int node_id_to = 0; node_id_to < _size_y; ++node_id_to){
	  double val = a[std::make_pair(0, node_id_from)];
	  z[std::make_pair(1, node_id_to)] += _w.at(1).at(node_id_from).at(node_id_to) * val;
	}
      }

      // update a = sigmoid(z)
      for(int node_id = 0; node_id < _size_y; ++ node_id){
	std::pair<int, int> key = std::make_pair(1, node_id);
	a[key] = sigmoid(z[key]);
      }

      // update _a
      _a.push_back(a);
    }
  };

  // calc cost function J(W, b)
  double cost_function(){
    double sum_of_squares = 0.0, l2_norm = 0.0, sparse_error = 0.0;
    // calc || hat{y} - y || ^ 2
    for(int data_id = 0; data_id < _a.size(); ++data_id){
      // a[<layer, node_id>]
      std::unordered_map<std::pair<int, int>, double, myhash, myeq> a = _a.at(data_id);

      std::unordered_map<int, int> checked;
      double err = 0.0;
      std::vector<std::pair<int, double> > y = _all_y.at(data_id);

      std::vector<std::pair<int, double> >::iterator i;
      for(i = y.begin(); i != y.end(); ++i){
	std::pair<int, double> elem = (*i);
	int answer_id = elem.first;
	double answer_val = elem.second;
	double predict_val = a[std::make_pair(1, answer_id)];

	// update flag
	checked[answer_id] = 1;

	// update err
	err += pow((answer_val - predict_val), 2);
      }

      // check other answer_id
      for(int answer_id = 0; answer_id < _size_y; ++answer_id){
	// calc dont check answer_id
	if(checked.find(answer_id) == checked.end()){
	  double predict_val = a[std::make_pair(1, answer_id)];
	  err += pow(predict_val, 2);
	}
      }
    }
    sum_of_squares /= (_a.size() * 2);

    // calc regularizer
    std::vector<std::vector<std::vector<double> > >::iterator i_1;
    for(i_1 = _w.begin(); i_1 != _w.end(); ++i_1){
      std::vector<std::vector<double> >::iterator i_2;
      std::vector<std::vector<double> > w_1 = *i_1;
      for(i_2 = w_1.begin(); i_2 != w_1.end(); ++i_2){
	std::vector<double>::iterator i_3;
	std::vector<double> w_2 = *i_2;
	for(i_3 = w_2.begin(); i_3 != w_2.end(); ++i_3){
	  double w = *i_3;
	  l2_norm += pow(w, 2);
	}
      }
    }
    l2_norm = l2_norm * _lambda / 2;

    // calc sparse term
    // calc KL divergence
    std::unordered_map<int, double> rho;
    int m = _a.size();
    for(int data_id = 0; data_id < m; ++data_id){
      std::unordered_map<std::pair<int, int>, double, myhash, myeq> a = _a.at(data_id);
      for(int node_id = 0; node_id < _size; ++node_id){
	rho[node_id] += a[std::make_pair(0, node_id)] / m;
      }
    }

    for(int node_id = 0; node_id < _size; ++node_id){
      double kl = _rho * log(_rho / rho[node_id]) + (1 - _rho) * log((1 - _rho) / (1 - rho[node_id]));
      sparse_error += kl;
    }

    sparse_error *= _beta;

    double error = sum_of_squares + l2_norm + sparse_error;
    return error;
  }
  
  void back_propagation(){
    
  };

private:
  // Training data
  // <
  //   <<feature_id, val>, ..., <feature_id, val>, >,
  //   ...,
  //   <<feature_id, val>, ..., <feature_id, val>, >
  // >
  std::vector<std::vector<std::pair<int, double> > > _all_x;

  // Answer data
  // <
  //   <<answer_id, val>, ..., <answer_id, val>, >,
  //   ...,
  //   <<answer_id, val>, ..., <answer_id, val>, >
  // >
  std::vector< std::vector<std::pair<int, double> > > _all_y;

  // x/y dimensions
  int _size_x, _size_y;

  // min/max of y(for train) of each answer_id
  std::unordered_map<int, double> _min_y, _max_y;

  // Activation values
  // _a.at(data_id)[<layer, node_id>]
  std::vector<std::unordered_map<std::pair<int, int>, double, myhash, myeq> > _a;

  // Weights
  // _w[layer][from_id][to_id]
  std::vector<std::vector<std::vector<double> > > _w;

  // Bias factor
  // _b[<layer, node_id>]
  std::unordered_map<std::pair<int, int>, double, myhash, myeq> _b;

  // Parameters
  int _size;
  double _alpha, _beta, _lambda, _rho;
};

