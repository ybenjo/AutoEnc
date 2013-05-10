// neural network
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

// boost random wrapper
// ref : http://d.hatena.ne.jp/n_shuyo/20100407/random
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

  // train oneself(x = f(x))
  // for autoencoding
  double set_data(const std::vector<std::pair<int, double> >& x){
    set_data(x, x);
  };

  // initialize _w
  void initialize(){
    std::vector<std::vector<std::pair<int, double> > >::iterator iter_1;
    std::vector<std::pair<int, double> >::iterator iter_2;

    // search x's dimension
    for(iter_1 = _all_x.begin(); iter_1 != _all_x.end(); ++iter_1){
      for(iter_2 = (*iter_1).begin(); iter_2 != (*iter_1).end(); ++iter_2){
        int feature_id = (*iter_2).first;
	if(feature_id > _max_x) _max_x = feature_id;
      }
    }

    // search y's dimension
    for(iter_1 = _all_y.begin(); iter_1 != _all_y.end(); ++iter_1){
      for(iter_2 = (*iter_1).begin(); iter_2 != (*iter_1).end(); ++iter_2){
        int answer_id = (*iter_2).first;
	if(answer_id > _max_y) _max_y = answer_id;
      }
    }
    
    // initialize _w
    // w[layer][from][to]

    // initialize N(0, 0.05)
    Rand<boost::normal_distribution<> > rnorm(0, 0.05);

    // weights of [x] => [hidden layer]
    std::vector<std::vector<double> > first_layer_weight;
    for(int from = 0; from <= _max_x; ++from){
      std::vector<double> w;
      for(int to = 0; to < _size; ++to){
	double val = rnorm();
	w.push_back(val);
      }
      first_layer_weight.push_back(w);
    }

    // weights of [hidden layer] => [y]
    std::vector<std::vector<double> > second_layer_weight;
    for(int from = 0; from < _size; ++from){
      std::vector<double> w;
      for(int to = 0; to <= _max_y; ++to){
	double val = rnorm();
	w.push_back(val);
      }
      second_layer_weight.push_back(w);
    }

    _w.push_back(first_layer_weight);
    _w.push_back(second_layer_weight);
  };

  void forward(){
  };

  void backward(){
  };

private:
  // training data
  // <
  //   <<feature_id, val>, ..., <feature_id, val>, >,
  //   ...,
  //   <<feature_id, val>, ..., <feature_id, val>, >
  // >
  std::vector<std::vector<std::pair<int, double> > > _all_x;

  // answer data
  // <
  //   <<answer_id, val>, ..., <answer_id, val>, >,
  //   ...,
  //   <<answer_id, val>, ..., <answer_id, val>, >
  // >
  std::vector< std::vector<std::pair<int, double> > > _all_y;

  // x/y dimensions
  int _max_x, _max_y;
    
  // activation values
  std::unordered_map<std::pair<int, int>, double, myhash, myeq> _a;

  // weights
  // _w[layer][from_id][to_id]
  std::vector<std::vector<std::vector<double> > > _w;

  // bias factor
  std::unordered_map<int, double> _b;

  // parameters
  int _size;
  double _alpha, _beta, _lambda, _rho;
};

