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
const int iteraciones = 15000;
const int cotavecinos = 20;


//CONSTANTES P2 GENÉTICOS
const int tamanioagg = 30;
const int tamanioage = 2;
const int tamaniomemeticos = 10;
const int max_eval = 15000;
const int vecinosporaplicacion = 2;
const int frecuenciabusqueda = 10;

const float pcruce = 0.7;
const float ablx = 0.3;
const float pmutacion = 0.7;



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


struct Cromosoma{
	vector<double> pesos;
	float fitness;
};

struct FitnessBasedComparer{
	bool operator()(Cromosoma uno, Cromosoma otro){
		return uno.fitness < otro.fitness;
	}
};

typedef multiset<Cromosoma, FitnessBasedComparer> Poblacion;


void computeFitness(vector<Instance> training, Cromosoma& c){
	vector<string> prediccion;

	for(int i=0; i<training.size(); i++){
		prediccion.push_back(class_1nn(training[i], training, true, c.pesos));
	}

	c.fitness = funcionObjetivo(score(prediccion, training), reduction(c.pesos));
}


void launchPopulation(Poblacion& p, int numc, int tamanio, vector<Instance>& train){
	uniform_real_distribution<double> random_real(0.0, 1.0);

  for(int i=0; i<numc; i++) {
    Cromosoma c;
    c.pesos.resize(tamanio);

    for (int j=0; j<tamanio; j++){
      c.pesos[j] = random_real(gen);
    }

    computeFitness(train, c);
    p.insert(c);
  }
}


Cromosoma selectByTournament(Poblacion &p){
	uniform_int_distribution<int> random_int(0, p.size()-1);

	int aleatorio = random_int(gen);
  auto it = p.begin();
  advance(it, aleatorio);
  Cromosoma c1 = *it;
  it = p.begin();
	aleatorio = random_int(gen);
  advance(it, aleatorio);
  Cromosoma c2 = *it;	  

  if(c1.fitness > c2.fitness)
  	return c1;
  else
  	return c2;
}


pair<Cromosoma, Cromosoma> alphaBLX(Cromosoma& c1, Cromosoma& c2) {
  Cromosoma h1, h2;
  h1.pesos.resize(c1.pesos.size());
  h2.pesos.resize(c2.pesos.size());

  for (int i=0; i<c1.pesos.size(); i++) {
    float cmin = min(c1.pesos[i], c2.pesos[i]);
    float cmax = max(c1.pesos[i], c2.pesos[i]);
    float distancia = cmax - cmin;

    uniform_real_distribution<float>
      random_real(cmin - (distancia * ablx), cmax + (distancia * ablx));

    h1.pesos[i] = random_real(gen);
    h2.pesos[i] = random_real(gen);

    if (h1.pesos[i] < 0) 
    	h1.pesos[i] = 0;
    if (h1.pesos[i] > 1) 
    	h1.pesos[i] = 1;
    if (h2.pesos[i] < 0) 
    	h2.pesos[i] = 0;
    if (h2.pesos[i] > 1) 
    	h2.pesos[i] = 1;
  }

  h1.fitness = -1.0; // Se acaban de generar luego no estan evaluados
  h2.fitness = -1.0;

  return make_pair(h1, h2);
}


Cromosoma arithmeticCross(Cromosoma& c1, Cromosoma& c2) {
  Cromosoma hijo;

  hijo.pesos.resize(c1.pesos.size());

  for (int i = 0; i < c1.pesos.size(); i++)
    hijo.pesos[i] = (c1.pesos[i] + c2.pesos[i]) / 2.0;

  hijo.fitness = -1.0; // Se acaban de generar luego no estan evaluados

  return hijo;
}


void mutateWeight(Cromosoma& c, int i) {
  normal_distribution<double> normal(0.0, sigma);
  c.pesos[i] += normal(gen);
  c.fitness = -1.0; 

  if (c.pesos[i] < 0) 
  	c.pesos[i] = 0;
  if (c.pesos[i] > 1) 
  	c.pesos[i] = 1;
}

int expected_mutations(int total_genes) {
  float expected_mut = pmutacion * total_genes;
  if (expected_mut <= 1.0)
    return 1;  // Garantizaremos al menos una mutación

  float resto = modf(expected_mut, &expected_mut);
  uniform_real_distribution<double> random_real(0.0, 1.0);
  double u = random_real(gen);
  if (u <= resto)
    expected_mut++;

  return expected_mut;
}


int agg_blx(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;
  Poblacion::reverse_iterator best_parent;  // El multiset guarda el mejor padre ahí para conservar elitismo
  
  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size() * tamanioagg;
  int ncruces = pcruce * (tamanioagg / 2);


  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamanioagg, pesos.size(), training);
  evaluaciones += tamanioagg;

  while (evaluaciones < max_eval) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    best_parent = p.rbegin();  // Guardamos la posición del mejor padre


    intermedia.resize(tamanioagg);
    for (int i=0; i<tamanioagg; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * ncruces; i+=2) {
      pair<Cromosoma, Cromosoma> par = alphaBLX(intermedia[i], intermedia[i+1]);     // Cruce BLX
      intermedia[i] = par.first;
      intermedia[i+1] = par.second;
    }

    set<int> posmutadas;
    int nmut = expected_mutations(ngenes);
    //Mutar
    for (int i=0; i<nmut; i++) {
      int pos;

      while(posmutadas.size() == i) {
        pos = random_int(gen);
        posmutadas.insert(pos);
      }

      int cromamutar = pos / pesos.size();
      int genamutar = pos % pesos.size();

      mutateWeight(intermedia[cromamutar], genamutar);
    }

    //Recalcular función objetivo
    for (int i=0; i<tamanioagg; i++) {
      if (intermedia[i].fitness == -1.0) {
        computeFitness(training, intermedia[i]);
        evaluaciones++;
      }
      nueva.insert(intermedia[i]);
    }

    auto current_best = nueva.rbegin(); // Guardamos la posición del mejor padre

    if (current_best->fitness < best_parent->fitness) { //Garantizar la conservación del mejor padre
      nueva.erase(nueva.begin());
      nueva.insert(*best_parent);
    }

    //Cambio de generación
    p = nueva;
    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}


int agg_ca(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;
  Poblacion::reverse_iterator best_parent;  // El multiset guarda el mejor padre ahí, lo usaremos para conservar elitismo
  
  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size() * tamanioagg;
  int ncruces = pcruce * (tamanioagg / 2);


  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamanioagg, pesos.size(), training);
  evaluaciones += tamanioagg;

  while (evaluaciones < max_eval) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    best_parent = p.rbegin();  // Guardamos la posición del mejor padre

    intermedia.resize(2*tamanioagg);
    for (int i=0; i<2*tamanioagg; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * ncruces; i++) {
      Cromosoma hijo = arithmeticCross(intermedia[i], intermedia[2*tamanioagg - i - 1]); //Cruce aritmético
      intermedia[i] = hijo;
    }

    //Mutar
    set<int> posmutadas;
    int nmut = expected_mutations(ngenes);

    for (int i=0; i<nmut; i++) {
      int pos;

      while(posmutadas.size() == i) {
        pos = random_int(gen);
        posmutadas.insert(pos);
      }

      int cromamutar = pos / pesos.size();
      int genamutar = pos % pesos.size();

      mutateWeight(intermedia[cromamutar], genamutar);
    }

    //Recalcular
    for (int i=0; i<tamanioagg; i++) {
      if (intermedia[i].fitness == -1.0) {
        computeFitness(training, intermedia[i]);
        evaluaciones++;
      }
      nueva.insert(intermedia[i]);
    }

    auto current_best = nueva.rbegin();

    if (current_best->fitness < best_parent->fitness) {
      nueva.erase(nueva.begin());
      nueva.insert(*best_parent);
    }

    p = nueva;
    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}



int age_blx(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;

  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size();
  int ncruces = (tamanioage / 2);
  float pmut = pmutacion * tamanioage * ngenes;


  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamanioagg, pesos.size(), training); //No es un error, se inicializa con 30 que es el valor de tamanioagg, tamanioage hace referencia a la poblacíón intermedia
  evaluaciones += tamanioagg;

  while (evaluaciones < max_eval) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    intermedia.resize(tamanioage);
    for (int i=0; i<tamanioage; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * ncruces; i+=2) {
      pair<Cromosoma, Cromosoma> par = alphaBLX(intermedia[i], intermedia[i+1]); //Cruce BLX
      intermedia[i] = par.first;
      intermedia[i+1] = par.second;
    }

    //Mutar
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for(int i=0; i<tamanioage; i++) {
      if (random_real(gen) <= pmut) {
        int gene = random_int(gen);
        mutateWeight(intermedia[i], gene);
      }
    }

    for (int i=0; i<tamanioage; i++) {
      computeFitness(training, intermedia[i]);
      evaluaciones++;
      nueva.insert(intermedia[i]);
    }


//Almaceno la posición de los mejores hijos y peores padres que compiten por pasar de generación
    auto peor = p.begin();
    auto segundopeor = ++p.begin();
    auto mejor = nueva.rbegin();
    auto segundomejor = ++nueva.rbegin();

    if (segundomejor->fitness > segundopeor->fitness) {
      // perviven ambos hijos

      p.erase(segundopeor);
      p.erase(p.begin());
      p.insert(*segundomejor);
      p.insert(*mejor);
    }

    else{ //Sólo pervive uno
    	if(mejor->fitness > peor->fitness){
    		p.erase(peor);
    		p.insert(*mejor);
    	}
    }

    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}



int age_ca(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;

  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size();
  int ncruces = (tamanioage / 2);
  float pmut = pmutacion * tamanioage * ngenes;


  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamanioagg, pesos.size(), training);
  evaluaciones += tamanioagg;

  while (evaluaciones < max_eval) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    intermedia.resize(2*tamanioage);
    for (int i=0; i<2*tamanioage; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * ncruces; i+=2) {
			intermedia[i] = arithmeticCross(intermedia[i], intermedia[2*tamanioage - i - 1]); //Cruce aritmético
    }

    //Mutar
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for(int i=0; i<tamanioage; i++) {
      if (random_real(gen) <= pmut) {
        int gene = random_int(gen);
        mutateWeight(intermedia[i], gene);
      }
    }

    //Recalcular
    for (int i=0; i<tamanioage; i++) {
      computeFitness(training, intermedia[i]);
      evaluaciones++;
      nueva.insert(intermedia[i]);
    }

    auto peor = p.begin();
    auto segundopeor = ++p.begin();
    auto mejor = nueva.rbegin();
    auto segundomejor = ++nueva.rbegin();

    if (segundomejor->fitness > segundopeor->fitness) {
      // perviven ambos hijos

      p.erase(segundopeor);
      p.erase(p.begin());
      p.insert(*segundomejor);
      p.insert(*mejor);
    }

    else{ //solo pervive uno
    	if(mejor->fitness > peor->fitness){
    		p.erase(peor);
    		p.insert(*mejor);
    	}
    }

    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}


int lightLocalSearch(vector<Instance> & training, Cromosoma & c) {

  normal_distribution<double> normal(0.0, sigma);

  const int n = c.pesos.size();
  vector<int> ind;
  double bestf;
  int evaluaciones = 0;

  for (int i = 0; i < n; i++) {
    ind.push_back(i);
  }

  shuffle(ind.begin(), ind.end(), gen);

  bestf = c.fitness;

  while (evaluaciones < n * vecinosporaplicacion ) {
    
    int comp = ind[evaluaciones % n];
    Cromosoma cmut = c;
    cmut.pesos[comp] += normal(gen);

    if (cmut.pesos[comp] > 1)
      cmut.pesos[comp] = 1;
    else 
    	if (cmut.pesos[comp] < 0)
    		cmut.pesos[comp] = 0;

    computeFitness(training, cmut);
    evaluaciones++;

    if (cmut.fitness > bestf) {
			c = cmut;
			bestf = cmut.fitness;
    }

    if (evaluaciones % n == 0) {
      shuffle(ind.begin(), ind.end(), gen);
    }
  }
  return evaluaciones;
 }


int memetic1(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;
  Poblacion::reverse_iterator mejorpadre;  // Elitism
  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size() * tamaniomemeticos;
  int num_cross = pcruce * (tamaniomemeticos / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamaniomemeticos, pesos.size(), training);
  evaluaciones += tamaniomemeticos;

  while (evaluaciones < iteraciones) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    mejorpadre = p.rbegin(); 

    intermedia.resize(tamaniomemeticos);
    for (int i=0; i<tamaniomemeticos; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * num_cross; i+=2) {
      pair<Cromosoma, Cromosoma> par = alphaBLX(intermedia[i], intermedia[i+1]);
      intermedia[i] = par.first;
      intermedia[i+1] = par.second;
    }

    set<int> posmutadas;
    int nmut = expected_mutations(ngenes);

    for (int i=0; i<nmut; i++) {
      int pos;

      while(posmutadas.size() == i) {
        pos = random_int(gen);
        posmutadas.insert(pos);
      }

      int cromamutar = pos / pesos.size();
      int genamutar = pos % pesos.size();

      mutateWeight(intermedia[cromamutar], genamutar);
    }

    for (int i=0; i<tamaniomemeticos; i++) {
      if (intermedia[i].fitness == -1.0) {
        computeFitness(training, intermedia[i]);
        evaluaciones++;
      }
      nueva.insert(intermedia[i]);
    }

    auto current_best = nueva.rbegin();

    if (current_best->fitness < mejorpadre->fitness) {
      nueva.erase(nueva.begin());
      nueva.insert(*mejorpadre);
    }

    p = nueva;
    
    //Aplicar búsqueda local sobre toda la población
    if (generacion % frecuenciabusqueda == 0) {

      nueva.clear();
      for (auto it = p.begin(); it != p.end(); ++it) {
        Cromosoma c = *it;
        evaluaciones += lightLocalSearch(training, c);
        nueva.insert(c);
      }

      p = nueva;
    }

    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}



int memetic2(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;
  Poblacion::reverse_iterator mejorpadre;
  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size() * tamaniomemeticos;
  int num_cross = pcruce * (tamaniomemeticos / 2);
  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamaniomemeticos, pesos.size(), training);
  evaluaciones += tamaniomemeticos;

  while (evaluaciones < iteraciones) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    mejorpadre = p.rbegin();

    intermedia.resize(tamaniomemeticos);
    for (int i=0; i<tamaniomemeticos; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * num_cross; i+=2) {
      pair<Cromosoma, Cromosoma> par = alphaBLX(intermedia[i], intermedia[i+1]);
      intermedia[i] = par.first;
      intermedia[i+1] = par.second;
    }

    set<int> posmutadas;
    int nmut = expected_mutations(ngenes);

    for (int i=0; i<nmut; i++) {
      int pos;

      while(posmutadas.size() == i) {
        pos = random_int(gen);
        posmutadas.insert(pos);
      }

      int cromamutar = pos / pesos.size();
      int genamutar = pos % pesos.size();

      mutateWeight(intermedia[cromamutar], genamutar);
    }

    for (int i=0; i<tamaniomemeticos; i++) {
      if (intermedia[i].fitness == -1.0) {
        computeFitness(training, intermedia[i]);
        evaluaciones++;
      }
      nueva.insert(intermedia[i]);
    }

    auto current_best = nueva.rbegin();

    if (current_best->fitness < mejorpadre->fitness) {
      nueva.erase(nueva.begin());
      nueva.insert(*mejorpadre);
    }

    p = nueva;
    
    //Aplicar búsqueda local sobre un 10 por ciento aleatorio de la población
    if (generacion % frecuenciabusqueda == 0) {
      uniform_int_distribution<int> random_int(0, tamaniomemeticos - 1);
      auto it = p.begin();
      advance(it, random_int(gen));
      Cromosoma c = *it;
      evaluaciones += lightLocalSearch(training, c);
      p.erase(it);
      p.insert(c);
    }

    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
}



 int memetic3(vector<Instance> training, vector<double>& pesos) {
  Poblacion p;
  Poblacion::reverse_iterator mejorpadre;
  int evaluaciones = 0;
  int generacion = 1;
  int ngenes = pesos.size() * tamaniomemeticos;
  int num_cross = pcruce * (tamaniomemeticos / 2);
  uniform_int_distribution<int> random_int(0, ngenes - 1);

  launchPopulation(p, tamaniomemeticos, pesos.size(), training);
  evaluaciones += tamaniomemeticos;

  while (evaluaciones < iteraciones) {
    vector<Cromosoma> intermedia;
    Poblacion nueva;

    mejorpadre = p.rbegin();

    intermedia.resize(tamaniomemeticos);
    for (int i=0; i<tamaniomemeticos; i++) {
      intermedia[i] = selectByTournament(p);
    }

    for (int i=0; i<2 * num_cross; i+=2) {
      pair<Cromosoma, Cromosoma> par = alphaBLX(intermedia[i], intermedia[i+1]);
      intermedia[i] = par.first;
      intermedia[i+1] = par.second;
    }

    set<int> posmutadas;
    int nmut = expected_mutations(ngenes);

    for (int i=0; i<nmut; i++) {
      int pos;

      while(posmutadas.size() == i) {
        pos = random_int(gen);
        posmutadas.insert(pos);
      }

      int cromamutar = pos / pesos.size();
      int genamutar = pos % pesos.size();

      mutateWeight(intermedia[cromamutar], genamutar);
    }

    for (int i=0; i<tamaniomemeticos; i++) {
      if (intermedia[i].fitness == -1.0) {
        computeFitness(training, intermedia[i]);
        evaluaciones++;
      }
      nueva.insert(intermedia[i]);
    }

    auto current_best = nueva.rbegin();

    if (current_best->fitness < mejorpadre->fitness) {
      nueva.erase(nueva.begin());
      nueva.insert(*mejorpadre);
    }

    p = nueva;
    
		//Aplicar búsqueda local sobre el 10 por ciento mejor de la población, sobre el mejor, la población tiene 10
    if (generacion % frecuenciabusqueda == 0) {
      auto it = --p.end();
      Cromosoma c = *it;
      evaluaciones += lightLocalSearch(training, c);
      p.erase(it);
      p.insert(c);
    }

    generacion++;
  }

  pesos = p.rbegin()->pesos;

  return generacion;
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

	  cout << "AGG - BLX" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    agg_blx(training[i], w);
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

	  cout << "AGG - CA" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    agg_ca(training[i], w);
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

	  cout << "AGE - BLX" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    age_blx(training[i], w);
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

	  cout << "AGE - CA" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    age_ca(training[i], w);
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

	  cout << "AM1 - BLX" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    memetic1(training[i], w);
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

	  cout << "AM2 - BLX" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    memetic2(training[i], w);
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

	  cout << "AM3 - BLX" << endl;

	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    memetic3(training[i], w);
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