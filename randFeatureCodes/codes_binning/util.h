#include <set>
#include <map>
#include <vector>

using namespace std;


class VectComp{ //assume vector of the length
	public:
	bool operator()(const vector<int>* a, const vector<int>* b){
		
		for(int i=0; i<a->size(); i++){
			
			int v1 = a->at(i);
			int v2 = b->at(i);
			if( v1 < v2 )
				return 1;
			else if( v1 > v2 )
				return 0;
		}
		
		return 0; //equal means not <
	}
};

typedef map<vector<int>*, int, VectComp> CodeIndexMap;
typedef map<vector<int>*, int, VectComp>::iterator CodeIndexMapIter;
