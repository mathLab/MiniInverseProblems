#ifndef MARKOV_CHAIN_H
#define MARKOV_CHAIN_H

class mhmc
{
    public:
        mhmc(StatisticalDistribution& in_distr, int in_Nmax, double in_initVal, double in_delta, string in_propDist = "uniform")
            : 
            distr(&in_distr) 
        {
            Nmax = in_Nmax;
            initVal = in_initVal;
            delta = in_delta;
            propDist = in_propDist;
        };
        ~mhmc(){};

        StatisticalDistribution* distr;
	int Nmax;
	double initVal;
	double delta;
	string propDist;
	VectorXd chain;
	double acceptanceRatio = 0;
	MatrixXd sampleMean;
	double intAutocorrTime;
        
	void get_intAutocorrTime()
        {
            intAutocorrTime = 0.5;
            MatrixXd autocovariance = stochastic::autocovariance(chain);
            cout << autocovariance.rows() << " x " << autocovariance.cols() << endl;
            for(int i = 0; i < Nmax; i++)
            { 
                intAutocorrTime += (1 - i / Nmax) * autocovariance(0,i);
            }
        };

	void createChain()
        {
            chain.resize(1);
            chain(0) = initVal;
            double new_x;
            double ratio;
            sampleMean.resize(1, Nmax);
        
            for (int i = 0; i < Nmax; i++)
            {
                chain.conservativeResize(chain.size() + 1);
                if(propDist.compare("uniform"))
        	{
        	    new_x = stochastic::set_uniform_random(chain(i), delta);
                }
                else if(propDist.compare("normal"))
        	{
        	    new_x = stochastic::set_normal_random(chain(i), delta);
                }
        	else
        	{
        	    cout << "Unknown proposal distribution, exiting" << endl;
        	    exit(10);
        	}
        
                if (distr->pdf(chain(i)) > 1e-14)
                {
                    ratio = distr->pdf(new_x) / distr->pdf(chain(i));
        
                    if (ratio >= 1.0)
                    {
                        chain(i + 1) = new_x;
        		acceptanceRatio++;
                    }
                    else if (stochastic::set_uniform_random(0.5, 0.5)  < ratio)
                    {
                        chain(i + 1) = new_x;
        		acceptanceRatio++;
                    }
                    else
                    {
                        chain(i + 1) = chain(i);
                    }
                }
                else
                {
                    chain(i + 1) = new_x;
        	    acceptanceRatio++;
                }
        	sampleMean.col(i) = stochastic::mean(chain);
            }
        
            acceptanceRatio /= Nmax;
            cout << "Acceptance ratio = " << acceptanceRatio << endl;
        };
};

#endif // MARKOV_CHAIN_H

