#ifndef _TRON_H
#define _TRON_H

#include "util.h"

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual Int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, double eps_cg = 0.1, Int max_iter = 1000);
	~TRON();

	void tron(double *w);
	void set_prInt_string(void (*i_prInt) (const char *buf));

private:
	Int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(Int n, double *x);

	double eps;
	double eps_cg;
	Int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_prInt_string)(const char *buf);
};
#endif
