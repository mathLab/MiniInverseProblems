
// * * * * * * * * * * * * * * * stochastic  * * * * * * * * * * * * * //

#ifndef STOCHASTIC_H
#define STOCHASTIC_H

namespace stochastic
{
    inline double set_uniform_random(double mean, double delta)
    {
        return ( (VectorXd::Random(1)(0) * delta) + mean );
    }
    
    inline double set_normal_random(double mean, double stddev)
    {
        std::random_device rd{};
        std::mt19937 gen{rd()};
 
        std::normal_distribution<> d{mean,stddev};
	return d(gen);
    }
    
    inline double set_gamma_random(double mean, double stddev)
    {
        std::default_random_engine generator;
        std::gamma_distribution<double> distribution(mean,stddev);
        return distribution(generator);
    }

    inline MatrixXd set_normal_random_matrix(int rows, int cols, double mean, double stddev)
    {
        MatrixXd M(rows, cols);
	for(int i = 0; i < rows; i++)
	{
	    for(int j = 0; j < cols; j++)
	    {
	        M(i,j) = set_normal_random(mean, stddev);
	    }
	}
        return M;
    }

    inline VectorXd mean(MatrixXd Matrix)
    {
        if(Matrix.cols() == 1)
	{
	    if(Matrix.rows() == 1)
	    {
	        cout << "WARNING: computing mean of a single scalar" << endl;
	    }
	    cout << "WARNING: only one sample. The samples should be on the columns" << endl;
	}
	return Matrix.transpose().colwise().mean();
    }

    inline MatrixXd center(MatrixXd sample)
    {
        sample.transposeInPlace();
        return sample.rowwise() - sample.colwise().mean();
    }

    inline VectorXd stddev(MatrixXd sample)
    {
        int N = sample.cols();
	VectorXd mean = stochastic::mean(sample);
	VectorXd sigma = VectorXd::Zero(sample.rows());

        for(int i = 0; i < N; i++)
	{
	    VectorXd sS = sample.col(i) - mean;
	    sigma = sigma.array() + sS.array().square();
	}
	sigma = (sigma.array() / (N - 1)).cwiseSqrt();
	return sigma;
    }

    inline MatrixXd covariance(MatrixXd sample)
    {
        MatrixXd centered = stochastic::center(sample);
        MatrixXd cov = (centered.adjoint() * centered) / double(sample.rows() - 1);
	return cov;
    }


    // Calculates credibility bounds
    // Returns a matrix with 2 columns. In each column there is one bound
    inline MatrixXd credibilityBounds(MatrixXd sample, double frac = 0.95)
    {
        // Compute the credible region on the mean
        double Nsigma = sqrt(2) * erfinv(frac);

	VectorXd mean = stochastic::mean(sample);
	VectorXd sigma_mu = stochastic::stddev(sample).array();
	MatrixXd ret(sample.rows(),2);
	ret.col(0) = mean - Nsigma * sigma_mu; 
	ret.col(1) = mean + Nsigma * sigma_mu; 
	return ret;
    }

    // Normalized autocovariance function
    // Autocorrelation function
    inline MatrixXd autocovariance(MatrixXd chain)
    {
        if(chain.cols() == 1)
	{
	    cout << "WARNING: In computing ACF, check if the chain is a row vector" << endl; 
	    chain.transposeInPlace();
	}

	M_Assert(chain.cols() >= 1, "Only one sample, samples should be on the columns of the matrix");
        MatrixXd autoCov(chain.rows(), chain.cols() - 1);
        double autocov0 = 1.0;
	for(int row = 0; row < autoCov.rows(); row ++)
	{
            for(int col = 0; col < autoCov.cols(); col++)
            {
                int h = chain.cols() - col;
                MatrixXd tail = chain.block(row, col, 1, h) - MatrixXd::Ones(1,h) * stochastic::mean(chain.row(row))(0);
                MatrixXd head = chain.block(row, 0, 1, h) - MatrixXd::Ones(1,h) * stochastic::mean(chain.row(row))(0);
                autoCov(row,col) = 1.0/(h) * (tail * head.transpose())(0,0);
                if(col == 0)
                {
                    autocov0 = autoCov(row,col);
                }
                autoCov(row,col) /= autocov0;
            }
	}
        return autoCov;
    }

    inline MatrixXd intAutocorrTime(MatrixXd chain)
    {
        if(chain.cols() == 1)
	{
	    cout << "WARNING: In computing ACF, check if the chain is a row vector" << endl; 
	    chain.transposeInPlace();
	}
        MatrixXd intAutocorrTime = MatrixXd::Ones(chain.rows(),1) * 0.5;
        MatrixXd autocovariance = stochastic::autocovariance(chain);
	for(int i = 0; i < intAutocorrTime.rows(); i++)
	{
            for(int j = 0; j < autocovariance.cols(); j++)
            {
                intAutocorrTime(i,0) += (1 - j / chain.cols()) * autocovariance(i,j);
		if(intAutocorrTime(i,0) < j/6)
		{
		    j = autocovariance.cols();
		    
		}
            }
	}
	return intAutocorrTime;
    }


}

#endif // STOCHASTIC_H

