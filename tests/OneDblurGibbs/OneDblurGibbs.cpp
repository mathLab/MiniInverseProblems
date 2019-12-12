#include "MiniInverseProblems.h"
//  
//  Hierarchical Gibbs sampler for 1d image deblurring with zero BCs.
//
//  Once the samples are computed, the sample mean is used as an estimator 
//  of the unknown image and empirical quantiles are used to compute 95%
//  credibility intervals for each unknown. 
//
//  The Geweke test is used to determine whether the chain, after burnin, 
//  is in equilibrium, and the integrated auto correlated time and essential 
//  sample size are estimated as described in Chapter 5. 
// 




int main()
{
    string folderRes = "results";
    // Create a Toeplitz matrix A using zero BCs and a Gaussian kernel.
    int n = 128; //No. of grid points
    double h = 1.0 / n;
    VectorXd t = VectorXd::LinSpaced(n, h/2, 1-h/2);
    double sig = .05;  
    VectorXd kernel = (1 / sqrt(M_PI) / sig) * exp(-(t.array() - h/2) * (t.array() - h/2) / (sig * sig));

    // Create a Toeplitz matrix A.
    MatrixXd A(n,n);
    for(int i = 0; i < n; i++)
    {	
        VectorXd v(n);
        v << (kernel.segment(1,i)).reverse(), kernel.head(n - i);
        A.row(i) = v.array();
    }
    A = A.array() * h;
    
    // Set up true solution x_true and data b = A*x_true + error.
    VectorXd x_true(n);
    for(int i = 0; i < n; i++)
    {
        if(t(i) > .1 && t(i) < .25)
        {
            x_true(i) = .75;
        }
        else if(t(i) > .3 && t(i) < .32)
        {
            x_true(i) = .25;
        }
        else if(t(i) > .5 && t(i) < 1)
        {
            x_true(i) = std::sin(2 * M_PI * t(i));
            x_true(i) = x_true(i) * x_true(i) * x_true(i) * x_true(i);
        }
        else
        {
            x_true(i) = 0;
        }
    }
    x_true  = x_true.array() * 50;// / x_true.norm();
    EigenStream::exportVector(t, "t", "eigen", folderRes);
    EigenStream::exportVector(x_true, "x_true", "eigen", folderRes);

    VectorXd Ax = A * x_true;
    double err_lev = 2; // Percent error in data
    double sigma = err_lev/100 * Ax.norm() / sqrt(n);
    VectorXd eta = sigma * stochastic::set_normal_random_matrix(n, 1, 0.0, 1.0);
    VectorXd b = Ax + eta;
    int m = b.size();
    VectorXd Atb = A.transpose() * b;
    MatrixXd AtA = A.transpose() * A;
    EigenStream::exportVector(b, "b", "eigen", folderRes);

    
    // second derivative precision matrix, with zero BCs, for prior
    MatrixXd L(n,n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
	{
	    if(i == j)
	    {
	        L(i,j) = 2;
	    }
	    else if(j == i -1 || j == i + 1)
	    {
	        L(i,j) = -1;
	    }
	}
    }
    
    // MCMC sampling
    int nsamps  = 10000;
    int nChol   = 0;
    MatrixXd xsamp = MatrixXd::Zero(n,nsamps);
    VectorXd delsamp = VectorXd::Zero(nsamps); 
    delsamp(0) = .1;
    VectorXd lamsamp = delsamp;
    lamsamp(0) = 1;
    MatrixXd R = ((lamsamp(0) * AtA + delsamp(0) * L).llt().matrixL()).transpose();
    nChol   = 1;
    xsamp.col(0) = R.inverse() * ((R.transpose()).inverse() * (lamsamp(0) * Atb));

    // hyperpriors: lambda~Gamma(a,1/t0), delta~Gamma(a1,1/t1)
    double a0=1; 
    double t0=0.0001; 
    double a1=1; 
    double t1=0.0001;
    for (int i = 0; i < nsamps - 1; i++)
    {
        //------------------------------------------------------------------
        // 1a. Using conjugacy, sample the noise precision lam=1/sigma^2,
        // conjugate prior: lam~Gamma(a0,1/t0)
	VectorXd Axsamp = A * xsamp.col(i) - b;
        lamsamp(i+1) = stochastic::set_gamma_random(a0 + m/2, 1.0 / (t0 + Axsamp.squaredNorm() / 2));
        
	//------------------------------------------------------------------
        // 1b. Using conjugacy, sample regularization precisions delta,
        // conjugate prior: delta~Gamma(a1,1/t1);
	double xtLx = xsamp.col(i).transpose() * (L * xsamp.col(i));
        delsamp(i+1) = stochastic::set_gamma_random(a1 + n/2, 1.0 / (t1 + xtLx / 2));
        
	//------------------------------------------------------------------
        // 2. Using conjugacy relationships, sample the image.
        R = ((AtA * lamsamp(i+1) + delsamp(i+1) * L).llt().matrixL()).transpose();
        nChol = nChol + 1;
        xsamp.col(i+1) = R.fullPivLu().solve((R.transpose()).fullPivLu().solve( Atb * lamsamp(i+1)) + stochastic::set_normal_random_matrix(n, 1, 0.0, 1.0));
    }
    
    // Visualize the MCMC chain
    // Plot the sample mean and 95% credibility intervals for x.
    int nburnin = nsamps/10;
    MatrixXd xsamp_steady = xsamp.rightCols(nsamps - nburnin);
    MatrixXd credibilityBounds = stochastic::credibilityBounds(xsamp, 0.95);
    VectorXd x_mean = stochastic::mean(xsamp_steady);
    double relative_error = (x_true - x_mean).norm() / x_true.norm();
    cout << "Relative error = " << relative_error << endl;
    EigenStream::exportVector(x_mean, "x_mean", "eigen", folderRes);
    EigenStream::exportMatrix(credibilityBounds, "credibilityBounds", "eigen", folderRes);

    VectorXd lambda_steady = lamsamp.tail(nsamps - nburnin);
    VectorXd delta_steady = delsamp.tail(nsamps - nburnin);
    MatrixXd lambdaIACT = stochastic::intAutocorrTime(lambda_steady);
    MatrixXd deltaIACT = stochastic::intAutocorrTime(delta_steady);
    cout << "lambda IACF = " << lambdaIACT <<endl;
    cout << "delta = " << deltaIACT <<endl;

    EigenStream::exportVector(lambda_steady, "lambda_steady", "eigen", folderRes);
    EigenStream::exportVector(delta_steady, "delta_steady", "eigen", folderRes);

    MatrixXd acfLambda = stochastic::autocovariance(lambda_steady);
    MatrixXd acfDelta = stochastic::autocovariance(delta_steady);
    EigenStream::exportMatrix(acfLambda, "acfLambda", "eigen", folderRes);
    EigenStream::exportMatrix(acfDelta, "acfDelta", "eigen", folderRes);
    return 0;
}
