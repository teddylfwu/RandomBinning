#include <random>
using namespace std;

class Gamma{
	
	public:
	default_random_engine gen;
	
	double gamma;
	gamma_distribution<double>* dist;

	Gamma(double g){
		gamma=g;
		dist = new gamma_distribution<double>(2.0, 1.0/gamma);
	}

	double generate(){
		return (*dist)(gen);
	}
};
