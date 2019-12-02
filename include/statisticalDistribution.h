
#ifndef STATISTICAL_DISTRIBUTION_H
#define STATISTICAL_DISTRIBUTION_H

class StatisticalDistribution
{
    public:
        
	virtual ~StatisticalDistribution(){};
        double mean;
        double var;
        double stdev;
	double acceptanceRatio;
	VectorXd sampleMean;

        // Distribution functions
        virtual double pdf(double& x) = 0;
        virtual double pdf(Ref<ArrayXd> x){exit(10);};
        virtual VectorXd draw(VectorXd x, VectorXd delta){exit(10);};
};

class StandardNormalDistribution : public StatisticalDistribution
{
    public:

        StandardNormalDistribution(double in_mean, double in_stdev)
        {
            mean = in_mean;
            stdev = in_stdev;
        };

        // Distribution functions
        virtual double pdf(double& x)
        {
            return (1.0 / std::sqrt(2.0 * M_PI * stdev * stdev)) * std::exp(- (x - mean) *
                    (x - mean) / (2 * stdev * stdev));
        };

        // Descriptive stats
        double get_mean()
        {
            return mean;   // equal to 0
        }
        double get_var()
        {
            return stdev * stdev;   // equal to 1
        }
        double get_stdev()
        {
            return stdev;   // equal to 1
        }

        // Obtain a sequence of random draws from the standard normal distribution
        //virtual void random_draws(const std::vector<double>& uniform_draws,
        //                          std::vector<double>& dist_draws);
};

class UniformDistribution : public StatisticalDistribution
{
    public:
        
	UniformDistribution(double in_mean, double in_delta)
        {
            mean = in_mean;
            delta = in_delta;
        };
        
	double delta;

        // Distribution functions
        virtual double pdf(double& x)
        {
            double prob = 0;
            if(x >= mean - delta && x <= mean + delta)
            {
                prob = 1 / (2 * delta);
            }
            return ( prob ); 
        };

        // Return a sample from the pdf
	double sample()
        {
            return ( (VectorXd::Random(1)(0) * delta) + mean ); 
        };

};

#endif // STATISTICAL_DISTRIBUTION_H 

