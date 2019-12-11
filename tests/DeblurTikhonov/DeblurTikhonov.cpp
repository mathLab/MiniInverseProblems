#include "MiniInverseProblems.h"
//  1d image deblurring with zero boundary conditions.
//
//  Tikhonov regularization is implemented with the regularization parameter 
//  alpha chosen using one of UPRE, GCV, DP, or L-curve. 
//
//  This test is taken from the reference https://archive.siam.org/books/cs19/

int main()
{
    string folderRes = "results";
    // Create a Toeplitz matrix A using zero BCs and a Gaussian kernel.
    int n = 80; //No. of grid points
    double h = 1.0 / n;
    VectorXd t = VectorXd::LinSpaced(n, h/2, 1-h/2);
    double sig = .03;  
    VectorXd kernel = (1 / sqrt(2 *M_PI) / sig) * exp(-(t.array() - h/2) * (t.array() - h/2) / (2 *sig * sig));

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
    x_true  = x_true.array() / x_true.norm();
    EigenStream::exportVector(t, "t", "eigen", folderRes);
    EigenStream::exportVector(x_true, "x_true", "eigen", folderRes);

    VectorXd Ax = A * x_true;
    double err_lev = 2; // Percent error in data
    double sigma = err_lev/100 * Ax.norm() / sqrt(n);
    VectorXd eta = sigma * stochastic::set_normal_random_matrix(x_true.size(), 1, 0.0, 1.0);
    VectorXd b = Ax + eta;
    EigenStream::exportVector(b, "b", "eigen", folderRes);

    // Compute the Tikhonov solution with alpha chosen using one of UPRE, GCV,
    // DP, or L-curve.
    linearSystem linSys(A, b);
    linSys.tikhonov("UPRE", sigma);
    std::cout << "Regularizing parameter = " << linSys.alpha << std::endl;
    
    VectorXd xfilt = linSys.x;
    double rel_error = (xfilt-x_true).norm() / x_true.norm();
    cout << "Relative error = " << rel_error << endl;
    EigenStream::exportVector(xfilt, "xfilt", "eigen", folderRes);

    return 0;
}
