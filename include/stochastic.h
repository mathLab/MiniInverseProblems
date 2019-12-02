
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
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution (mean, stddev);

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
	    Matrix.transposeInPlace();
	}
        VectorXd mu(Matrix.rows());
	for(int i = 0; i < Matrix.rows(); i++)
	{
	    mu(i) = Matrix.row(i).sum() / Matrix.cols();
	}
	return mu;
    }

    inline MatrixXd covariance(MatrixXd sample)
    {
        VectorXd mu = stochastic::mean(sample);
	MatrixXd cov(sample.rows(), sample.rows());
	int Nobserv = sample.cols();
	cov = 1 / (Nobserv -1) * (sample - mu * MatrixXd::Ones(1, Nobserv)) * (sample - mu * MatrixXd::Ones(1, Nobserv)).transpose();
	return cov;
    }

    //// Normalized autocovariance function
    //inline VectorXd autocovariance(VectorXd chain)
    //{
    //    VectorXd autoCov;
    //    autoCov.resize(chain.size() - 1);
    //    double autocov0 = 1.0;
    //    for(int i = 0; i < autoCov.size(); i++)
    //    {
    //        VectorXd tail = chain.tail(chain.size() - i) - VectorXd::Ones(chain.size() - i) * stochastic::mean(chain);
    //        VectorXd head = chain.head(chain.size() - i) - VectorXd::Ones(chain.size() - i) * stochastic::mean(chain);
    //        autoCov(i) = 1.0/(chain.size() - 1) * tail.dot(head);
    //        if(i == 0)
    //        {
    //            autocov0 = autoCov(i);
    //        }
    //        autoCov(i) /= autocov0;
    //    }
    //    return autoCov;
    //}

    // Normalized autocovariance function
    inline MatrixXd autocovariance(MatrixXd chain)
    {
        MatrixXd autoCov;
        if(chain.cols() == 1)
	{
	    chain.transposeInPlace();
	}
        autoCov.resize(chain.rows(), chain.cols() - 1);
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
        MatrixXd intAutocorrTime = MatrixXd::Ones(chain.rows(),1) * 0.5;
        MatrixXd autocovariance = stochastic::autocovariance(chain);
	for(int i = 0; i < intAutocorrTime.rows(); i++)
	{
            for(int j = 0; j < autocovariance.cols(); j++)
            {
                intAutocorrTime(i,0) += (1 - j / chain.cols()) * autocovariance(i,j);
            }
	}
	return intAutocorrTime;
    }


}

#endif // STOCHASTIC_H

