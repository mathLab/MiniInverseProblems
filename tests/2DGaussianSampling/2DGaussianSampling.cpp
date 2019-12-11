#include "MiniInverseProblems.h"
//  
//  Sampling from 2D gaussian distribution
//

int main()
{
    string folderRes = "results";
    
    int N = 2;
    int Nsamples = 10000;
    double rho = 0.5;
    VectorXd mean(N);
    mean << 0 , 0;
    MatrixXd cov(N,N);
    cov << 1, rho,
           rho, 1;
    std::cout << "cov = " << R.transpose() * R  <<std::endl;
    
    // Generate 10000 realization from the random vector using Cholesky factor of the covariance
    MatrixXd R = (cov.llt().matrixL()).transpose();
    
    MatrixXd v = stochastic::set_normal_random_matrix(N, Nsamples, 0.0, 1.0);
    MatrixXd samples = R.inverse() * v;
    
    VectorXd testMean = stochastic::mean(samples);
    MatrixXd testCov = stochastic::covariance(samples);
    std::cout << "Direct sampling Mean = " << testMean << std::endl;
    std::cout << "Direct sampling Cov = " << testCov << std::endl;
    
    // Hierarchical Gibbs for sampling
    int Ngibbs = 100000;
    VectorXd y0(N);
    y0 << 0, 0;
    MatrixXd y(N,Ngibbs);
    y.col(0) = y0; 
    for(int k = 1; k < Ngibbs; k++)
    {
        y(0,k) = stochastic::set_normal_random(rho * y(1, k-1), 1 - (rho * rho));
	y(1,k) = stochastic::set_normal_random(rho * y(0, k), 1 - (rho * rho)); 
    }
    
    testMean = stochastic::mean(y);
    testCov = stochastic::covariance(y);
    std::cout << "Gibbs Mean = " << testMean << std::endl;
    std::cout << "Gibbs Cov = " << testCov << std::endl;
    
    
    
    
    //// Generate the numerical integral matrix 
    //int n = 80; //No. of grid points
    //double h = 1.0 / n;
    //int N = 2 * n  - 1;
    //MatrixXd A = MatrixXd::Ones(N,N).triangularView<Lower>();
    //A *= h;
    //
    ////Set up true solution x_true, which will be a non-symmetric kernel. 
    ////First, we create the left- and right-halves of the kernel, and then we 
    ////combine them and normalize the resulting kernel to obtain x_true.
    //VectorXd t = VectorXd::LinSpaced(N, -1+h, 1-h);
    //double sig1 = .1;  
    //cout << "t.size() = " << t.size() << endl;
    //VectorXd kernelleft = exp(-(t.head(n).array() * t.head(n).array()) / (2 * sig1 * sig1));
    //double sig2 = .2; 
    //VectorXd kernelright = exp(-(t.tail(n - 1).array() * t.tail(n - 1).array()) / (2 * sig2 * sig2));
    //VectorXd kernel(kernelleft.size() + kernelright.size());
    //kernel << kernelleft, kernelright;
    //kernel = kernel.array() / (kernel.sum() * h);
    //VectorXd x_true = kernel;
    //cout << "x_true.size() = " << x_true.size() << endl;
    //cout << A.rows() << " , " << A.cols() << endl;
    //
    //// Now, we generate the data b = A*x_true + error.
    //VectorXd Ax = A * x_true;
    //double err_lev = 2; // Percent error in data
    //double sigma = err_lev/100 * Ax.norm() / sqrt(N);
    //VectorXd eta = sigma * stochastic::set_normal_random_matrix(x_true.size(), 1, 0.0, 1.0);
    //VectorXd b = Ax + eta;
    //EigenStream::exportVector(t, "t", "eigen", folderRes);
    //EigenStream::exportVector(x_true, "x_true", "eigen", folderRes);
    //EigenStream::exportVector(b, "b", "eigen", folderRes);
    //
    //// Compute the TSVD solution, choosing k using one of UPRE, GCV, or DP.
    //linearSystem linSys(A, b);
    //linSys.TSVD("UPRE", sigma);

    //// Now compute the regularized solution.
//ph//i         = zeros(N,1); phi(1:k)=1;
//id//x         = (dS>0);
//dS//filt      = zeros(size(dS));
//dS//filt(idx) = phi(idx)./dS(idx);
//xf//ilt       = V*(dSfilt.*(U'*b));
//re//l_error   = norm(xfilt-x_true)/norm(x_true)
//fi//gure(2)
//  //plot(t,x_true,'b-',t,xfilt,'k-')
    //
    //VectorXd phi = VectorXd::Zero(N);
    //phi.segment(0,linSys.truncSingVal) = VectorXd::Ones(linSys.truncSingVal); 
    //JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    //VectorXd dS = svd.singularValues();
    //MatrixXd U = svd.matrixU();
    //MatrixXd V = svd.matrixV();
    //VectorXd dSfilt = VectorXd::Zero(dS.size());
    //for(int i = 0; i < dS.size(); i++)
    //{
    //    if(dS(i) > 0)
    //    {
    //        dSfilt(i) = phi(i) / dS(i);
    //    }
    //}
    //VectorXd xfilt = V * (dSfilt.cwiseProduct(U.transpose() * b));
    //double rel_error = (xfilt - x_true).norm() / x_true.norm();
    //cout << "Relative error = " << rel_error << endl;
    //EigenStream::exportVector(xfilt, "xfilt", "eigen", folderRes);
    return 0;
}
