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


//CONSTANTES POBLACIONES
const int max_eval = 15000;
const int NPACKS = 5;
const int NCOYOTESPERPACK = 5;
const int frecuenciameme =30;
const int vecinosporaplicacion = 2;



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

//Función para generar números caóticos

double valor = 0.4;

double logistic(){
  valor = 4*valor*(1-valor);
  return valor;
}


//Funciones generales de prácticas anteriores

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


//Implementaciones específicas para el COA


struct Coyote{
	vector<double> pesos;
	float fitness;
  int edad;

  bool operator==(Coyote & p) const{
    return this->fitness == p.fitness and this->pesos==p.pesos and this->edad==p.edad;
  }
};

struct FitnessBasedComparer{
	bool operator()(Coyote uno, Coyote otro){
		return uno.fitness < otro.fitness;
	}
};

//typedef multiset<Coyote, FitnessBasedComparer> Pack; Se trato de utilizar para mantener los coyotes ordenados, finalmente no se utiliza por complicaciones de memoria en tiempo de ejecución.


void computeFitness(vector<Instance> training, Coyote& c){
	vector<string> prediccion;

	for(int i=0; i<training.size(); i++){
		prediccion.push_back(class_1nn(training[i], training, true, c.pesos));
	}

	c.fitness = funcionObjetivo(score(prediccion, training), reduction(c.pesos));
}


void launchPopulation(vector<vector<Coyote> >& packs, int np, int nc, int tamanio, vector<Instance>& train){
	uniform_real_distribution<double> random_real(0.0, 1.0);
  vector<vector<Coyote> > init;
  for(int i=0; i<np; i++) {
    vector<Coyote> p;
    for(int j=0; j<nc; j++){
      Coyote c;
      c.pesos.resize(tamanio);
      c.edad=0;
      for (int k=0; k<tamanio; k++){
        c.pesos[k] = random_real(gen);
      }
      computeFitness(train, c);
      p.push_back(c);
    }
    init.push_back(p);
  }

  packs = init;

}

void chaoticlaunchPopulation(vector<vector<Coyote> >& packs, int np, int nc, int tamanio, vector<Instance>& train){
  vector<vector<Coyote> > init;
  for(int i=0; i<np; i++) {
    vector<Coyote> p;
    for(int j=0; j<nc; j++){
      Coyote c;
      c.pesos.resize(tamanio);
      c.edad=0;
      for (int k=0; k<tamanio; k++){
        c.pesos[k] = logistic();
      }
      computeFitness(train, c);
      p.push_back(c);
    }
    init.push_back(p);
  }

  packs = init;

}




vector<double> computeCulturalTendency(vector<Coyote> &pack){
  int tamanio = pack[0].pesos.size();
  vector<double> median(tamanio);
  vector<vector<double> > social_conditions(tamanio);
  for(int i=0; i<tamanio; i++){
    vector<double> condition;
    for(int j=0; j<NCOYOTESPERPACK; j++){
      condition.push_back(pack[j].pesos[i]);
    }
    sort(condition.begin(), condition.end());
    social_conditions[i]=condition;
  }

  for(int i=0; i<tamanio; i++){
    median[i]=social_conditions[i][social_conditions[i].size()/2];
  }

  return median;
}

Coyote birth(vector<Coyote> & pack, vector<Instance> & training, int tamanio, double pa){
  Coyote puppy;
  puppy.edad=0;
  puppy.pesos.resize(tamanio);

  uniform_int_distribution<int> random_int(0, tamanio - 1);
  uniform_int_distribution<int> random_coyote(0, NCOYOTESPERPACK - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  int d1 = random_int(gen);
  int d2 = random_int(gen);


  Coyote rc1 = pack[random_coyote(gen)];
  Coyote rc2 = rc1;



  while(rc1==rc2){
    rc2 = pack[random_coyote(gen)];
  }

  puppy.pesos[d1] = rc1.pesos[d1];
  puppy.pesos[d2] = rc2.pesos[d2];



  for(int i=0; i<tamanio; i++){
    if(i!=d1 and i!=d2){
      double aleatorio = random_real(gen);

      if(aleatorio<pa)
        puppy.pesos[i]=rc1.pesos[i];
      else{
        if(aleatorio>(1-pa))
          puppy.pesos[i]=rc2.pesos[i];
        else
          puppy.pesos[i]=random_real(gen);
      } 
    }
  }
  
  computeFitness(training, puppy);

  return puppy;
}

Coyote chaoticbirth(vector<Coyote> & pack, vector<Instance> & training, int tamanio, double pa){
  Coyote puppy;
  puppy.edad=0;
  puppy.pesos.resize(tamanio);

  uniform_int_distribution<int> random_int(0, tamanio - 1);
  uniform_int_distribution<int> random_coyote(0, NCOYOTESPERPACK - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  int d1 = random_int(gen);
  int d2 = random_int(gen);


  Coyote rc1 = pack[random_coyote(gen)];
  Coyote rc2 = rc1;



  while(rc1==rc2){
    rc2 = pack[random_coyote(gen)];
  }

  puppy.pesos[d1] = rc1.pesos[d1];
  puppy.pesos[d2] = rc2.pesos[d2];



  for(int i=0; i<tamanio; i++){
    if(i!=d1 and i!=d2){
      double aleatorio = random_real(gen);

      if(aleatorio<pa)
        puppy.pesos[i]=rc1.pesos[i];
      else{
        if(aleatorio>(1-pa))
          puppy.pesos[i]=rc2.pesos[i];
        else
          puppy.pesos[i]=logistic();
      } 
    }
  }
  
  computeFitness(training, puppy);

  return puppy;
}

//Búsqueda ligera implementada en la P2 para los meméticos

 int lightLocalSearch(vector<Instance> & training, Coyote & c) {

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
    Coyote cmut = c;
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


int coa(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  launchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  while (evaluaciones < max_eval) {

    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double r1 = random_real(gen);
        double r2 = random_real(gen);

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = birth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }
    
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
}

int coamejorado(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  launchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  while (evaluaciones < max_eval) {

    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double sum = poblacion[i][0].fitness, best = poblacion[i][0].fitness, worst = poblacion[i][0].fitness;

        for(int k=1; k<NCOYOTESPERPACK; k++){
          sum+=poblacion[i][k].fitness;
          if(best < poblacion[i][k].fitness)
            best = poblacion[i][k].fitness;
          if(worst > poblacion[i][k].fitness)
            worst = poblacion[i][k].fitness;
        }

        sum /= ( (double)NCOYOTESPERPACK);        

        double r1 = sum / best;
        double r2 = worst / sum;

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = birth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }
    
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
}

int coamem1(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  launchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  int generacion=0;
  while (evaluaciones < max_eval) {
    generacion++;
    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double r1 = random_real(gen);
        double r2 = random_real(gen);

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = birth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }

    if(generacion%frecuenciameme == 0){

      vector<Coyote> alfas;
      int positions;

      for(int i=0; i<NPACKS; i++){
        Coyote alfa = poblacion[i][0];
        positions=0;
        for(int j=1; j<NCOYOTESPERPACK; j++){
          if(alfa.fitness < poblacion[i][j].fitness){
            positions=j;
            alfa = poblacion[i][j];
          }
        }
        evaluaciones+= lightLocalSearch(training, poblacion[i][positions]);
      }

    }

    
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
}


int coamem2(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  launchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  int generacion=0;
  while (evaluaciones < max_eval) {
    generacion++;
    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double r1 = random_real(gen);
        double r2 = random_real(gen);

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = birth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }

    if(generacion%frecuenciameme == 0){

      vector<Coyote> alfas;
      int positions;

      for(int i=0; i<NPACKS; i++){
        Coyote alfa = poblacion[i][0];
        positions=0;
        for(int j=1; j<NCOYOTESPERPACK; j++){
          if(alfa.fitness < poblacion[i][j].fitness){
            positions=j;
            alfa = poblacion[i][j];
          }
        }
        evaluaciones+= lightLocalSearch(training, poblacion[i][positions]);
      }

      vector<Coyote> omegas;
      int positions2;

      for(int i=0; i<NPACKS; i++){
        Coyote omega = poblacion[i][0];
        positions2=0;
        for(int j=1; j<NCOYOTESPERPACK; j++){
          if(omega.fitness > poblacion[i][j].fitness){
            positions2=j;
            omega = poblacion[i][j];
          }
        }
        evaluaciones+= lightLocalSearch(training, poblacion[i][positions2]);
      }

    }

    
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
}


int coamem3(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  launchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  int generacion=0;
  while (evaluaciones < max_eval) {
    generacion++;
    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double r1 = random_real(gen);
        double r2 = random_real(gen);

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = birth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
    lightLocalSearch(training, alfas[i]);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
}


int chaoticcoa(vector<Instance> & training, vector<double>& pesos) {
  vector<vector<Coyote> > poblacion;
  
  int evaluaciones = 0;
  double intercambio = 0.005 * NCOYOTESPERPACK * NCOYOTESPERPACK;

  chaoticlaunchPopulation(poblacion, NPACKS, NCOYOTESPERPACK, pesos.size(), training);
  evaluaciones += (NPACKS * NCOYOTESPERPACK);

  uniform_int_distribution<int> random_int(0, NCOYOTESPERPACK - 1);
  uniform_int_distribution<int> random_pack(0, NPACKS - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  double ps = 1.0/pesos.size();
  double pa = (1-ps)/2;
  while (evaluaciones < max_eval) {

    for(int i=0; i<NPACKS; i++){

      Coyote alfa = poblacion[i][0];
      for(int j=1; j<NCOYOTESPERPACK; j++){
        if(alfa.fitness < poblacion[i][j].fitness)
          alfa = poblacion[i][j];
      }


           
      vector<double> pack_cultural_tendency = computeCulturalTendency(poblacion[i]);




      for(int j=0; j<NCOYOTESPERPACK; j++){
        Coyote actual = poblacion[i][j];
        Coyote rc1 = actual;

        while(rc1==actual){
          rc1 = poblacion[i][random_int(gen)];
        }
        Coyote rc2 = actual;


        while(rc2==actual or rc2==rc1){
          rc2 = poblacion[i][random_int(gen)];
        }
              

        vector<double> delta1(pesos.size()), delta2(pesos.size());

        double r1 = logistic();
        double r2 = logistic();

        for(int k=0; k<pesos.size(); k++){
          delta1[k]=alfa.pesos[k]-rc1.pesos[k];
          delta2[k]=pack_cultural_tendency[k]-rc2.pesos[k];

          actual.pesos[k]+= (r1*delta1[k] + r2*delta2[k]);

          if(actual.pesos[k]<0)
            actual.pesos[k]=0;
          if(actual.pesos[k]>1)
            actual.pesos[k]=1;
        }

        computeFitness(training, actual);
        evaluaciones++;
        
        if(actual.fitness>poblacion[i][j].fitness){
          poblacion[i][j]=actual;
        }
        
      }


      Coyote puppy = chaoticbirth(poblacion[i], training, pesos.size(), pa);
      evaluaciones++;


      vector<Coyote> worsethanpuppy;
      vector<int> worseindex;

      for(int j=0; j<NCOYOTESPERPACK; j++){
        if(poblacion[i][j].fitness < puppy.fitness){
          worsethanpuppy.push_back(poblacion[i][j]);
          worseindex.push_back(j);
        }
      }

      if(worsethanpuppy.size()==1){
        poblacion[i][worseindex[0]]=puppy;
      }
      else{
        if(worsethanpuppy.size()>1){
          Coyote mayor = worsethanpuppy[0];
          int pos = 0;
          for(int l=1; l<worsethanpuppy.size(); l++){
            if(mayor.edad<worsethanpuppy[l].edad){
              mayor = worsethanpuppy[l];
              pos = l;
            }
          }
          poblacion[i][worseindex[pos]]=puppy;
        }
      }
      
    }


    if(random_real(gen)<intercambio){
      int p1 = random_pack(gen);
      int p2 = random_pack(gen);

      int c1 = random_int(gen);
      int c2 = random_int(gen);


      Coyote aux = poblacion[p1][c1];
      poblacion[p1][c1] = poblacion[p2][c2];
      poblacion[p2][c2] = aux;
    }
    
    for(int h=0; h<NPACKS; h++){
      for(int m=0; m<NCOYOTESPERPACK; m++){
        poblacion[h][m].edad+=1;
      }
    }
    
    
  }

  vector<Coyote> alfas;

  for(int i=0; i<NPACKS; i++){
    Coyote alfa = poblacion[i][0];
    for(int j=1; j<NCOYOTESPERPACK; j++){
      if(alfa.fitness < poblacion[i][j].fitness)
        alfa = poblacion[i][j];
    }
    alfas.push_back(alfa);
  }

  Coyote best_coyote = alfas[0];
  for(int j=1; j<NPACKS; j++){
    if(best_coyote.fitness < alfas[j].fitness)
      best_coyote = alfas[j];
  }

  pesos = best_coyote.pesos;

  return 0;
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


	  cout << "COA" << endl;


	  for(int i=0; i<training.size(); i++){
	  	ini = chrono::high_resolution_clock::now();
	    coa(training[i], w);
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

    cout << "ChaoticCOA" << endl;


    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      chaoticcoa(training[i], w);
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

    cout << "COAMejorado" << endl;


    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      coamejorado(training[i], w);
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

    cout << "COAMemetic1" << endl;


    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      coamem1(training[i], w);
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


    cout << "COAMemetic2" << endl;


    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      coamem2(training[i], w);
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


    cout << "COAMemetic3" << endl;


    for(int i=0; i<training.size(); i++){
      ini = chrono::high_resolution_clock::now();
      coamem3(training[i], w);
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