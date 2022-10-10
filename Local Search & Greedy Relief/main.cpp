#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <stdlib.h>  
#include <random> 
#include <chrono> 
using namespace std;


const float alpha = 0.5;
const int K = 5;
const float sigma = 0.3;
const int iteraciones = 15000;
const int cotavecinos = 20;

class Instance{
	public:
		vector<double> x;
		string y;
		Instance(){}
		Instance(vector<double> _x, double _y){
			x=_x;
			y=to_string(_y);
		} 
};

default_random_engine gen;

void read_arff(vector<vector<double> > & x, vector<double> & y, string file) {
	x.clear();
	y.clear();
	ifstream i;
	string leyendo, caracteristica;
	vector<double> objeto;
	string ini = "";

	i.open(file, ifstream::in);
	if (i.fail()){
		cerr << "Error leyendo el archivo arff: " << file << endl;
		exit(-1);
	}
	else{
	    while (!i.eof()){
	    	while(ini!="@data"){
	    		getline(i, ini, '\n');
	    	}

			getline(i, leyendo, '\n');

	      	if (!i.eof()){
		        istringstream iss(leyendo);
		        objeto.clear();
		        while(getline(iss, caracteristica, ',')){
		        	if(caracteristica=="b")
		        		caracteristica = "1";
		        	if(caracteristica=="g")
		        		caracteristica = "2";
			        objeto.push_back(stod(caracteristica));
		        }

		        x.push_back(objeto);
	  		}
	    }
	}
	i.close();

  for (int i=0; i < x.size(); ++i){
    y.push_back(x[i][x[i].size()-1]);
    x[i].pop_back();
  }

}

void normalize(vector<Instance> & data) {
	double min, max;
	for(int i=0; i<data[0].x.size(); i++){
		min = data[0].x[i];
		max = data[0].x[i];

		for(int j=0; j<data.size(); j++){
			if(min>data[j].x[i])
				min = data[j].x[i];
			if(max<data[j].x[i])
				max = data[j].x[i];
		}

		if(max-min != 0){
			for(int j=0; j<data.size(); j++)
				data[j].x[i] = (data[j].x[i] - min) / (max - min);			
		}


	}
}

double weightedDistance(vector<double> o1, vector<double> o2, vector<double> w){
	double d = 0;
	for(int i=0; i<o1.size(); i++){
		d += w[i] * (o1[i]-o2[i]) * (o1[i]-o2[i]);
	}

	d = sqrt(d);

	return d;
}

//Medidas

double score(vector<string> prediccion, vector<Instance> test){
	int aciertos = 0;

	for(int i=0; i<prediccion.size(); i++){
		if(prediccion[i]==test[i].y)
			aciertos++;
	}
	return ((float)aciertos / prediccion.size())* 100;
}

double reduction(vector<double> & w){
	int irrelevantes=0;

	for(int i=0; i<w.size(); i++)
		if(w[i]<0.1)
			irrelevantes++;

	return  ((float) irrelevantes / w.size()) * 100.0;
}

double funcionObjetivo(double score, double reduction){
	return alpha * score + (1.0 - alpha) * reduction;
}


string class_1nn(Instance instancia, vector<Instance> train, bool weighted, vector<double>  w){
	int pos = 0;
	double d=0;
	if(!weighted)
		w = vector<double> (instancia.x.size(), 1.0);

	double min_distance = numeric_limits<double>::max();

	for(int i=0; i<train.size(); i++){
		if(train[i].x!=instancia.x){
			d = weightedDistance(instancia.x, train[i].x, w); 
			if(d<min_distance){
				pos = i;
				min_distance=d;
			}
		}			
	}
	
	return train[pos].y;
}


void nearestEnemyandFriend(Instance instancia, vector<Instance> train, int & posEnemy, int & posFriend){
	double d, min_distanceEnemy=numeric_limits<double>::max(), min_distanceFriend=numeric_limits<double>::max();
	vector<double> w(instancia.x.size(), 1.0);
	for(int i=0; i<train.size(); i++){
		if(train[i].x!=instancia.x){
			d = weightedDistance(instancia.x, train[i].x, w);
			if(train[i].y!=instancia.y && d<min_distanceEnemy){
				min_distanceEnemy=d;
				posEnemy=i;
			}
			if(train[i].y==instancia.y && d<min_distanceFriend){
				min_distanceFriend=d;
				posFriend=i;
			}			
		}

	}
}


void computeReliefWeights(vector<Instance> train, vector<double> & w){
	for(int i=0; i<train[0].x.size(); i++)
		w[i]=0;

	for(int i=0; i<train.size(); i++){
		int posFriend=0, posEnemy=0;

		nearestEnemyandFriend(train[i], train, posEnemy, posFriend);

		for(int j=0; j<w.size(); j++)
			w[j] = w[j] + fabs(train[i].x[j] - train[posEnemy].x[j]) - fabs(train[i].x[j] - train[posFriend].x[j]);


	}

	double maxWeight = *max_element(w.begin(), w.end());

	for(int i=0; i<w.size(); i++){
		if(w[i]<0)
			w[i]=0.0;
		else
			w[i] /= maxWeight; 

		//cout << w[i] << " ";
	}
}


int local_search(vector<Instance> & training, vector<double> & w) {

  normal_distribution<double> normal(0.0, sigma);
  uniform_real_distribution<double> uniform_real(0.0, 1.0);

  const int n = training[0].x.size();
  vector<string> clasificados;
  vector<int> ind;
  double bestf;
  int mut = 0;

  for (int i = 0; i < n; i++) {
    ind.push_back(i);
    w[i] = uniform_real(gen);
  }

  shuffle(ind.begin(), ind.end(), gen);
  for (int i = 0; i < training.size(); i++)
    clasificados.push_back(class_1nn(training[i], training, true, w));
  bestf = funcionObjetivo(score(clasificados, training), reduction(w));
  clasificados.clear();

  int iter = 0, vecino = 0;
  bool mejora = false;
  while (iter < iteraciones && vecino < n * cotavecinos) {
    int comp = ind[iter % n];
    vector<double> w_mut = w;
    w_mut[comp] += normal(gen);
    if (w_mut[comp] > 1)
      w_mut[comp] = 1;
    else 
    	if (w_mut[comp] < 0)
    		w_mut[comp] = 0;

    for (int i = 0; i < training.size(); i++)
      clasificados.push_back(class_1nn(training[i], training, true, w_mut));
    double f = funcionObjetivo(score(clasificados, training), reduction(w_mut));
    iter++;

    if (f > bestf) {
      mut++;
      vecino = 0;
      w = w_mut;
      bestf = f;
      mejora = true;
    }
    else {
      vecino++;
    }

    clasificados.clear();
    if (iter % n == 0 || mejora) {
      shuffle(ind.begin(), ind.end(), gen);
      mejora = false;
    }
  }
  return 0;
 }




void makeKFolds(vector<Instance> &data, vector<vector<Instance>> &training, vector<vector<Instance>> &test){
  string y = data[0].y;

  vector<Instance> c1, c2;
  int n = data.size();

  for(int i=0; i<n; i++){
    if(data[i].y == y)
      c1.push_back(data[i]);
    else
      c2.push_back(data[i]);
  }

  random_shuffle(c1.begin(), c1.end());
  random_shuffle(c2.begin(), c2.end());

  double p = ceil((double) n/K);
  int sc1 = c1.size(), sc2 = c2.size();
  int mc1 = ceil((double) sc1/K), mc2 = ceil((double) sc2/K);

  vector<Instance> aux1, aux2;
  vector<vector<Instance>> pc1, pc2;

  for(int i=0; i<K; i++){
    for(int j=i*mc1; j<min((i+1)*mc1, sc1); j++){
      aux1.push_back(c1[j]);
    }
    pc1.push_back(aux1);
    aux1.clear();
  }

  for(int i=0; i<K; i++){
    for(int j=i*mc2; j<min((i+1)*mc2, sc2); j++){
      aux1.push_back(c2[j]);
    }
    pc2.push_back(aux1);
    aux1.clear();
  }

  for(int i=0; i<K; i++){
    for(int j=0; j<K; j++){
      if(i!=j){
        for(int k=0; k<pc1[j].size(); k++)
          aux1.push_back(pc1[j][k]);
        for(int k=0; k<pc2[j].size(); k++)
          aux1.push_back(pc2[j][k]);
      }
      else{
        for(int k=0; k<pc1[j].size(); k++)
          aux2.push_back(pc1[j][k]);
        for(int k=0; k<pc2[j].size(); k++)
          aux2.push_back(pc2[j][k]);
      }
    }
    training.push_back(aux1);
    test.push_back(aux2);
    aux1.clear();
    aux2.clear();
  }

}


int main(int argc, char *argv[]){


	string datasets[3] =  {"Instancias_APC/ionosphere.arff", "Instancias_APC/parkinsons.arff", "Instancias_APC/spectf-heart.arff"};

	for(auto ds : datasets){
		cout << ds << endl;

		vector<vector<double> > x;
		vector<double> y;
		vector<vector<Instance>> training;
		vector<vector<Instance>> test;
		vector<string> clasificados;
		vector<double> pesos;

		vector<Instance> data;


		read_arff(x, y, ds);

		for(int i=0; i<x.size(); i++){
			data.push_back(Instance(x[i], y[i]));
		}

	  int m = 5;
	  normalize(data);
	  gen.seed(stoi(argv[1]));



	  makeKFolds(data, training, test);


	  vector<double> w (data[0].x.size(), 0);
	  double s, r;
	  chrono::high_resolution_clock::time_point ini, fin;
    chrono::duration<double> time;


	  cout << "1-NN " << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    for(int j=0; j<test[i].size(); j++){
	    	clasificados.push_back(class_1nn(test[i][j], training[i], false, w));
	    }
	    s = score(clasificados, test[i]);
	    r=0;
	    fin = chrono::high_resolution_clock::now();
	    time = chrono::duration_cast<chrono::milliseconds>(fin - ini);
	    cout << s << " " << r << " " << funcionObjetivo(s, r) << " " << time.count() << endl;
	    clasificados.clear();
	  }

	  cout << "RELIEF" << endl;
	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    computeReliefWeights(training[i], w);
	    for(int j=0; j<test[i].size(); j++){
	    	clasificados.push_back(class_1nn(test[i][j], training[i], true, w));
	    }
	    s = score(clasificados, test[i]);
	    r = reduction(w);
	    fin = chrono::high_resolution_clock::now();
	    time = chrono::duration_cast<chrono::milliseconds>(fin - ini);
	    cout << s << " " << r << " " << funcionObjetivo(s, r) << " " << time.count() << endl;
	    clasificados.clear();
	  }


	  cout << "BL" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    local_search(training[i], w);
	    for(int j=0; j<test[i].size(); j++){
	    	clasificados.push_back(class_1nn(test[i][j], training[i], true, w));
	    }
	    s = score(clasificados, test[i]);
	    r = reduction(w);
	    fin = chrono::high_resolution_clock::now();
	    time = chrono::duration_cast<chrono::milliseconds>(fin - ini);
	    cout << s << " " << r << " " << funcionObjetivo(s, r) << " " << time.count() << endl;
	    clasificados.clear();
	  }

}


}