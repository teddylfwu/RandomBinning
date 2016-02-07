#ifndef LOSS
#define LOSS

#include<cmath>
using namespace std;

class LossFunc{
	
	public:
	virtual double fval(double v, double y)=0;
	virtual double deriv(double v, double y)=0;
	virtual double sec_deriv_ubound()=0;
};

class SquareLoss: public LossFunc {
	
	public:
	double fval(double v, double y){
		
		double val = (y-v);
		return val*val/2.0;
	}

	double deriv(double v, double y){
		
		double val = -(y-v);
		return val;
	}

	double sec_deriv_ubound(){
		return 1.0;
	}
};

class L2hingeLoss: public LossFunc {
	
	public:
	double fval(double v, double y){
		
		double val = (1-y*v);
		if( val > 0.0 )
			return val*val/2.0;
		else
			return 0.0;
	}

	double deriv(double v, double y){
		
		double val = (1-y*v);
		if( val > 0.0 )
			return -y*val;
		else
			return 0.0;
	}

	double sec_deriv_ubound(){
		return 1.0;
	}
};

class LogisticLoss: public LossFunc {
	
	public:
	double fval(double v, double y){
		
		double val = -y*v;
		if( val < 0.0 )
			return 	log( 1.0 + exp(val) );
		else
			return  log( exp(-val) + 1.0 ) + val;
	}

	double deriv(double v, double y){
		
		double val = -y*v;
		if( val < 0.0 )
			return (-y)*exp(val)/(1.0 + exp(val));
		else
			return (-y)/(exp(-val)+1.0);
	}

	double sec_deriv_ubound(){
		return 0.25;
	}
};

#endif
