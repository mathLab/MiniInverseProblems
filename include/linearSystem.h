#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H
// * * * * * * * * * * * * * * * Linear System  * * * * * * * * * * * * * //

class linearSystem
{
    public:
        linearSystem(){};
        linearSystem(MatrixXd& _A, MatrixXd& _b)
        {
            A = _A;
            b = _b;
            M_Assert(A.rows() == b.rows(), "A and b must have same number of rows.");
            M_Assert(b.cols() == 1, "b must have only one column");
        };
        MatrixXd A;
	MatrixXd b;
        MatrixXd x;

	void TSVD(string regPar, double sigma = 0.0, label truncatedSV = 0)
        {
            Info << "Using truncated SVD for regularization" << endl;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
                                                  Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::MatrixXd U = svd.matrixU();
            Eigen::MatrixXd V = svd.matrixV();
        
            if (truncatedSV == 0)
            {
                Info << "Regularization parameter selection method: " << endl;
        	Info << regPar << endl;
        	if(regPar == "UPRE")
        	{
        	    truncatedSV = 0;
        	    VectorXd Utb = U.transpose() * b;
        	    double min = Utb.squaredNorm();
        	    Info << "Unbiased predictive risk estimator" << endl;
        	    for(int k = 1; k < svd.singularValues().size(); k++)
        	    {
        	         int s = Utb.size() - k + 1;
        	         if(min > Utb.tail(s).squaredNorm() + 2 * sigma * sigma * k)
        		 {
        		     truncatedSV = k;
        		     min = Utb.tail(s).squaredNorm() + 2 * sigma * sigma * k;
        		 }
        	    }
        	    Info << "Truncating after " << truncatedSV << " eigenvalues" << endl;
        
        	}
        	else
        	{
        	    Info << "I don't know the regularization parameter selection method, exiting" << endl;
        	    exit(9);
        	}
        
        
            }
            else
            {
                Info << "Using user-defined truncating value, k = " << truncatedSV << endl;
            }
        
            for (label i = 0; i < truncatedSV; i++)
            {
                scalar coeff = (U.col(i).transpose() * b)(0, 0);
        
                if (i == 0)
                {
                    x = coeff / svd.singularValues()(i) * V.col(i);
                }
                else
                {
                    x += coeff / svd.singularValues()(i) * V.col(i);
                }
            }
        };

	void tikhonov(string regPar, double sigma)
        {
            Info << "Using Tikhonov regularization" << endl;
            JacobiSVD<Eigen::MatrixXd> svd(A,
                                           Eigen::ComputeThinU | Eigen::ComputeThinV);
            MatrixXd U = svd.matrixU();
            MatrixXd V = svd.matrixV();
            VectorXd S = svd.singularValues();
            VectorXd S2 = S.array() * S.array();
                
            Info << "Regularization parameter selection method: " << endl;
            Info << regPar << endl;
            double alpha;
            VectorXd a = VectorXd::LinSpaced(10000, 0 ,S2(0));
            VectorXd Utb = U.transpose() * b;
            VectorXd Utb2 = Utb.array() * Utb.array();
            double min;
            if(regPar == "UPRE")
            {
                Info << "Unbiased predictive risk estimator" << endl;
                for(int i = 0; i < a.size(); i++)
                {
        	     double U1 = ((a(i) * a(i) * Utb2).array() / ((S2.array() + a(i)) * (S2.array() + a(i)))).sum();
        	     double U2 = 2 * sigma * sigma * (S2.array() / (S2.array() + a(i))).sum();
        	     double U = U1 + U2;
        
        	     if(i == 0)
        	     {
        	         min = U;
            	         alpha = a(i);
        	     }
                     if(min > U)
            	     {
            	         alpha = a(i);
        		 min = U;
            	     }
                }
            }
            else if(regPar == "GCV")
            {
                Info << "Generalized Cross Validation" << endl;
                for(int i = 1; i < a.size(); i++)
                {
        	     double U1 = ((a(i) * a(i) * Utb2).array() / ((S2.array() + a(i)) * (S2.array() + a(i)))).sum();
        	     double U2 = A.rows() - (S2.array() / (S2.array() + a(i))).sum();
        	     U2 *= U2;
        	     double U = U1 / U2;
        
        	     if(i == 0)
        	     {
        	         min = U;
            	         alpha = a(i);
        	     }
                     if(min > U)
            	     {
            	         alpha = a(i);
        		 min = U;
            	     }
                }
            }
            else
            {
                Info << "I don't know the regularization parameter selection method, exiting" << endl;
                exit(9);
            }
            Info << "alpha = " << alpha << endl;
        
            //VectorXd Sfilt = S.array() / (S2.array() + alpha);
            //x = Sfilt.array() * (U.transpose() * b).array();
            //x = V * x;
            
            MatrixXd Aalpha = A.transpose() * A + alpha * MatrixXd::Identity(A.cols(), A.cols());
            x = Aalpha.fullPivLu().solve(A.transpose() * b);
        };
};

#endif // LINEAR_SYSTEM_H

