/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Example of a heat transfer Reduction Problem
SourceFiles
    18bayesian.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "ITHACAutilities.H"
#include "ITHACAstream.H"
#include "ITHACAbayesian.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Foam2Eigen.H"


//class testDistribution: public StatisticalDistribution
//{
//    public:
//        ~testDistribution(){};
//        
//        double pdf(double& x)
//        {
//	    Info << "PDF only defined for vector entry" << endl;
//            return 0;
//        }
//        
//	double pdf(Ref<ArrayXd> x)
//        {
//	    double retVal = 1 / std::sqrt(2 * M_PI) * std::exp(- (x(0) * x(0)) / 2 - (x(1) * x(1)));
//	    return retVal;
//        }
//	
//};

class testDistribution: public StatisticalDistribution
{
    public:
        ~testDistribution(){};
        
        double pdf(double& x)
        {
	    Info << "PDF only defined for vector entry" << endl;
            return 0;
        }
        
	double pdf(Ref<ArrayXd> x)
        {
	    double retVal = 0;
	    M_Assert(x.size() == 2, "The vector must have size 2");
	    if(std::abs(x(0)) <= 2 && std::abs(x(1)) <= 2)
	    {
	        retVal = std::exp(-10 * (x(0) * x(0) - x(1)) * (x(0) * x(0) - x(1)) - (x(1) - 0.25) * (x(1) - 0.25) * (x(1) - 0.25) * (x(1) - 0.25));
	    }
	    return retVal;
        }
	
};

class propDistribution: public StatisticalDistribution
{
    public:
       
        propDistribution(){};
	propDistribution(double in_gamma)
	{
	    gamma = in_gamma;
	};
        
	double gamma = 0.4;

        double pdf(double& x)
        {
	    Info << "PDF only defined for vector entry" << endl;
            return 0;
        }
        
	VectorXd draw(VectorXd x, VectorXd delta)
	{
	    VectorXd retVect(x.size());
	    for(int i =0; i < x.size(); i++)
	    {
	        retVect(i) = x(i) + gamma * stochastic::set_normal_random(0.0, 1.0); 
	    }
	    return retVect;
	}
};

int main(int argc, char* argv[])
{
    word folder = "./ITHACAoutput/test/";
    testDistribution dist;
    double gamma = 0.4;
    propDistribution propDist(gamma);
    VectorXd seeds_x0 = VectorXd::LinSpaced(200, -2, 2);
    VectorXd seeds_x1 = VectorXd::LinSpaced(200, -2, 2);
    MatrixXd pdf(seeds_x0.size(), seeds_x1.size());
    int Niter = 10000;
    VectorXd initVal(2);
    initVal << 0, 0;
    VectorXd delta = VectorXd::Zero(2);

    VectorXd x(2);
    for(int i = 0; i < seeds_x0.size(); i++)
    {
        for(int j = 0; j < seeds_x1.size(); j++)
	{
	    x << seeds_x0(i), seeds_x1(j);
	    pdf(i,j) = dist.pdf(x);
	}
    }

    //multi_mhmc markovChain(dist
    ITHACAstream::exportMatrix(pdf, "pdf", "eigen",
                               folder);
    ITHACAstream::exportVector(seeds_x0, "seedsX0", "eigen",
                               folder);
    ITHACAstream::exportVector(seeds_x1, "seedsX1", "eigen",
                               folder);
    Info << "pdf.sum = " << pdf.sum() << endl;
    
    multi_mhmc markovChain(dist, propDist, Niter, initVal, delta);
    markovChain.createChain();
    Info << "Chain constructed"<< endl;
    //markovChain.get_intAutocorrTime();
    //Info << "intAutocorrTime = " << markovChain.intAutocorrTime << endl;
    
    MatrixXd autocovariance = stochastic::autocovariance(markovChain.chain);
    MatrixXd intAutocorrTime = stochastic::intAutocorrTime(markovChain.chain);

    std::cout << "Integrated autocorrelation time = " << intAutocorrTime << endl;
    ITHACAstream::exportMatrix(markovChain.chain, "output", "eigen",
                               folder);
    ITHACAstream::exportMatrix(markovChain.sampleMean, "sampleMean", "eigen",
                               folder);
    ITHACAstream::exportMatrix(autocovariance, "autocovariance", "eigen",
                               folder);
    
    return 0;
}


