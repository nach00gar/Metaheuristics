#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <functional>
#include <iterator>
#include <string>
#include <algorithm>
#include <math.h>
#include <stdlib.h>  
#include <random> 
#include <chrono> 
using namespace std;


//CONSTANTES GENERALES Y P1
const float alpha = 0.5;
const int K = 5;
const float sigma = 0.3;
const int iteraciones = 1000;
const int cotavecinos = 20;



//CONSTANTES P3 TRAYECTORIAS
const int max_total_eval =15000;
const int numero_arranques = 15;

//ENFRIAMIENTO
const float phi = 0.3;
const float mu = 0.3;
float final_temp = 1e-3;
const int VECINOSMAX = 10;
const int VECINOSMAXAUX = 1;
const float MAX_SUCCESS_PER_NEIGHBOUR = 0.1;
const int MAX_ITER_ES = 15000;
const int MAX_ITER_ES_AUX = 1000;
const int INICIOS_BMB = 15;


//REITERADA
const float sigma_ils = 0.4;
const int MAX_ITER_LS = 1000;

const int MAX_NEIGHBOUR_PER_TRAIT_LS = 20;
const int ITER_ILS = 15;
const float MUTATION_FACTOR_ILS = 0.1;

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


struct Solution{
	vector<double> pesos;
	float fitness;
};

struct SolutionComparer{
	bool operator()(Solution uno, Solution otro){
		return uno.fitness < otro.fitness;
	}
};

void computeFitness(vector<Instance> training, Solution& c){
  vector<string> prediccion;

  for(int i=0; i<training.size(); i++){
    prediccion.push_back(class_1nn(training[i], training, true, c.pesos));
  }

  c.fitness = funcionObjetivo(score(prediccion, training), reduction(c.pesos));
}



Solution init_solution(const vector<Instance> training, int n) {
  Solution sol;
  uniform_real_distribution<double> random_real(0.0, 1.0);

  sol.pesos.resize(n);
  for (int i = 0; i < n; i++)
    sol.pesos[i] = random_real(gen);
  computeFitness(training, sol);

  return sol;
}

// Mutate a component of a weight vector
void mutate(vector<double>& w, int comp, float sigma) {
  normal_distribution<double> normal(0.0, sigma);
  w[comp] += normal(gen);

  if (w[comp] < 0.0) w[comp] = 0.0;
  if (w[comp] > 1.0) w[comp] = 1.0;
}




//P3



void enfriamiento_simulado(const vector<Instance>& training, vector<double>& w) {
  Solution sol, best_sol;


  //Inicializaciones
  float temp;
  float initial_temp;

  int iter = 0;

  int successful;
  int neighbour;
  int n = w.size();

  uniform_int_distribution<int> random_int(0, n - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  // Inicializamos la solución
  sol = init_solution(training, n);
  iter++;
  best_sol = sol;
  //Cálculo de la temperatura inicial
  initial_temp = (mu * (1.0 - best_sol.fitness / 100.0)) / (- 1.0 * log(phi));
  temp = initial_temp;

  while (final_temp >= temp)
    final_temp = temp / 100.0;

  const int MAX_NEIGHBOUR = VECINOSMAX * n;
  const int MAX_SUCCESS = MAX_SUCCESS_PER_NEIGHBOUR * MAX_NEIGHBOUR;
  const int M = MAX_ITER_ES / MAX_NEIGHBOUR;


  const float beta = (float) (initial_temp - final_temp) / (M * initial_temp * final_temp);

  successful = MAX_SUCCESS;
  iter = 1;
  while(iter < MAX_ITER_ES && successful != 0) {
    neighbour = 0;
    successful = 0;

    while(iter < MAX_ITER_ES && neighbour < MAX_NEIGHBOUR && successful < MAX_SUCCESS) {
      int comp = random_int(gen);
      Solution sol_mut = sol;
      mutate(sol_mut.pesos, comp, sigma);
      computeFitness(training, sol_mut);
      iter++;
      neighbour++;

      float diff = sol.fitness - sol_mut.fitness;

      if (diff == 0)
        diff = 0.001;

      if (diff < 0 || random_real(gen) <= exp(-1.0 * diff / temp)) {
        successful++;
        sol = sol_mut;
        if (sol.fitness > best_sol.fitness)
          best_sol = sol;
      }

    }

    temp = temp / (1.0 + beta * temp);

  }


  w = best_sol.pesos;
}



//ILS


void local_search(const vector<Instance>& training, Solution& s) {
  const int n = s.pesos.size();
  vector<int> index;
  double best_fitness = s.fitness;
  int iter = 0;
  int neighbour = 0;
  bool improvement = false;

  for (int i = 0; i < n; i++)
    index.push_back(i);
  shuffle(index.begin(), index.end(), gen);

  while (iter < MAX_ITER_LS && neighbour < n * MAX_NEIGHBOUR_PER_TRAIT_LS) {
    int comp = index[iter % n];

    Solution s_mut = s;
    mutate(s_mut.pesos, comp, sigma);
    computeFitness(training, s_mut);
    iter++;

    if (s_mut.fitness > best_fitness) {
      neighbour = 0;
      s = s_mut;
      best_fitness = s_mut.fitness;
      improvement = true;
    }

    else {
      neighbour++;
    }

    if (iter % n == 0 || improvement) {
      shuffle(index.begin(), index.end(), gen);
      improvement = false;
    }
  }
}

void ils(const vector<Instance>& training, vector<double>& w) {
  int n = w.size();
  uniform_int_distribution<int> random_int(0, n - 1);
  Solution s = init_solution(training, n);

  local_search(training, s);

  for (int i = 1; i < ITER_ILS; i++) {
    Solution s_mut = s;

    set<int> mutated;
    for (int j = 0; j < (int) MUTATION_FACTOR_ILS * n; j++) {
      int comp;

      while(mutated.size() == j) { 
        comp = random_int(gen);
        mutated.insert(comp);
      }

      mutate(s_mut.pesos, comp, sigma_ils);
    }

    computeFitness(training, s_mut);
    local_search(training, s_mut);

    if (s_mut.fitness > s.fitness)
      s = s_mut;
  }

  w = s.pesos;
}


void multiarranque_ls(const vector<Instance>& training, vector<double>& w) {
  int n = w.size();

  vector<Solution> inicios;

  for (int i=0; i<INICIOS_BMB; i++){
    Solution s = init_solution(training, n);
    inicios.push_back(s);
  }

  for (int i = 0; i < INICIOS_BMB; i++)
    local_search(training, inicios[i]);


  Solution best_sol = inicios[0];

  for (int i = 1; i < INICIOS_BMB; i++){
    if (best_sol.fitness < inicios[i].fitness)
      best_sol = inicios[i];
  }
    
  w = best_sol.pesos;
  
}




//Hibridación

void enfriamiento_simulado_auxiliar(const vector<Instance>& training, Solution& ils) {
  Solution sol, best_sol;
  int n = ils.pesos.size();

  //Inicializaciones
  float temp;
  float initial_temp;

  int iter = 0;

  int successful;
  int neighbour;

  uniform_int_distribution<int> random_int(0, n - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  // Inicializamos la solución
  sol = ils;
  iter++;
  best_sol = sol;
  //Cálculo de la temperatura inicial
  initial_temp = (mu * (1.0 - best_sol.fitness / 100.0)) / (- 1.0 * log(phi));
  temp = initial_temp;

  while (final_temp >= temp)
    final_temp = temp / 100.0;

  const int MAX_NEIGHBOUR = VECINOSMAXAUX * n;
  const int MAX_SUCCESS = MAX_SUCCESS_PER_NEIGHBOUR * MAX_NEIGHBOUR;
  const int M = MAX_ITER_ES / MAX_NEIGHBOUR;


  const float beta = (float) (initial_temp - final_temp) / (M * initial_temp * final_temp);

  successful = MAX_SUCCESS;
  iter = 1;
  while(iter < MAX_ITER_ES_AUX && successful != 0) {
    neighbour = 0;
    successful = 0;


    while(iter < MAX_ITER_ES_AUX && neighbour < MAX_NEIGHBOUR && successful < MAX_SUCCESS) {
      int comp = random_int(gen);
      Solution sol_mut = sol;
      mutate(sol_mut.pesos, comp, sigma);
      computeFitness(training, sol_mut);
      iter++;
      neighbour++;

      float diff = sol.fitness - sol_mut.fitness;

      if (diff == 0)
        diff = 0.001;

      if (diff < 0 || random_real(gen) <= exp(-1.0 * diff / temp)) {
        successful++;
        sol = sol_mut;
        if (sol.fitness > best_sol.fitness)
          best_sol = sol;
      }

    }

    temp = temp / (1.0 + beta * temp);

  }


  ils = best_sol;
}


void hibrido(const vector<Instance>& training, vector<double>& w) {
  int n = w.size();
  uniform_int_distribution<int> random_int(0, n - 1);
  Solution s = init_solution(training, n);

  enfriamiento_simulado_auxiliar(training, s);

  for (int i = 1; i < ITER_ILS; i++) {
    Solution s_mut = s;

    set<int> mutated;
    for (int j = 0; j < (int) MUTATION_FACTOR_ILS * n; j++) {
      int comp;

      while(mutated.size() == j) { 
        comp = random_int(gen);
        mutated.insert(comp);
      }

      mutate(s_mut.pesos, comp, sigma_ils);
    }

    computeFitness(training, s_mut);
    enfriamiento_simulado_auxiliar(training, s_mut);

    if (s_mut.fitness > s.fitness)
      s = s_mut;
  }

  w = s.pesos;
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

    cout << "BMB" << endl;

    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      multiarranque_ls(training[i], w);
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

	  cout << "ES" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    enfriamiento_simulado(training[i], w);
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


    cout << "ILS" << endl;

    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      ils(training[i], w);
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


    cout << "Hibridación ILS-ES" << endl;

    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      hibrido(training[i], w);
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