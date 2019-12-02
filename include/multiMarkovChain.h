#ifndef MULTI_MARKOV_CHAIN_H
#define MULTI_MARKOV_CHAIN_H

class multi_mhmc
{
    public:
        multi_mhmc(StatisticalDistribution& in_distr, StatisticalDistribution& in_prop,int in_Nmax, VectorXd in_initVal, VectorXd in_delta)
            : 
            distr(&in_distr), 
            propDistr(&in_prop) 
        {
            // ADD Check propDistr has the draw member
            Nmax = in_Nmax;
            initVal = in_initVal;
            delta = in_delta;
        };
        ~multi_mhmc(){};

        StatisticalDistribution* distr;
        StatisticalDistribution* propDistr;
	int Nmax;
	VectorXd initVal;
	VectorXd delta;
	MatrixXd chain;
	double acceptanceRatio = 0;
	MatrixXd sampleMean;
	double intAutocorrTime;
        
	//void get_intAutocorrTime();
	void createChain()
        {
            chain.resize(initVal.size(), 1);
            chain.col(0) = initVal;
            VectorXd new_x;
            double ratio;
            sampleMean.resize(chain.rows(), Nmax);
            
            for (int i = 0; i < Nmax; i++)
            {
                chain.conservativeResize(chain.rows(), chain.cols() + 1);
        	new_x = propDistr->draw(chain.col(i), delta);
        
                ratio = distr->pdf(new_x) / distr->pdf(chain.col(i));
        
                if (ratio >= 1.0)
                {
                    chain.col(i + 1) = new_x;
        	    acceptanceRatio++;
                }
                else if (stochastic::set_uniform_random(0.5, 0.5)  < ratio)
                {
                    chain.col(i + 1) = new_x;
        	    acceptanceRatio++;
                }
                else
                {
                    chain.col(i + 1) = chain.col(i);
                }
        	sampleMean.col(i) = stochastic::mean(chain);
            }
        
            acceptanceRatio /= Nmax;
            Info << "Acceptance ratio = " << acceptanceRatio << endl;
            std::cout << "Mean = " << sampleMean.rightCols(1) << std::endl; 
        };
};

#endif // MULTI_MARKOV_CHAIN_H

