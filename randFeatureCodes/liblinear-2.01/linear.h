#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "util.h" 

struct feature_node
{
	Int index;
	double value;
};

struct problem
{
	Int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

struct parameter
{
	Int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	Int nr_weight;
	Int *weight_label;
	double* weight;
	double p;
	double *init_sol;
};

struct model
{
	struct parameter param;
	Int nr_class;		/* number of classes */
	Int nr_feature;
	double* w;
	Int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, Int nr_fold, double *target);
void find_parameter_C(const struct problem *prob, const struct parameter *param, Int nr_fold, double start_C, double max_C, double *best_C, double *best_rate);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

Int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

Int get_nr_feature(const struct model *model_);
Int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, Int* label);
double get_decfun_coef(const struct model *model_, Int feat_idx, Int label_idx);
double get_decfun_bias(const struct model *model_, Int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
Int check_probability_model(const struct model *model);
Int check_regression_model(const struct model *model);
void set_prInt_string_function(void (*prInt_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

