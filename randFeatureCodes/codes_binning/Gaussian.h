#include<stdlib.h>
#include<cmath>

double randn(double mu=0.0, double sigma=1.0) {

	static bool deviateAvailable=false;		//flag
	static float storedDeviate;		//		deviate from previous calculation
	double dist, angle;

	if (!deviateAvailable) {

		dist=sqrt( -2.0 * log(double(rand()) / double(RAND_MAX)) );
		angle=2.0 * M_PI * (double(rand()) / double(RAND_MAX));

		storedDeviate=dist*cos(angle);
		deviateAvailable=true;

		return dist * sin(angle) * sigma + mu;
	}
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}
