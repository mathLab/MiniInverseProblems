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



class testDistribution: public StatisticalDistribution
{
    public:
        testDistribution(double in_mean, double in_stdev)
        {
            mean = in_mean;
            stdev = in_stdev;
        }

        double pdf(double& x)
        {
            M_Assert(x > 0, "This pdf is only defined for positive reals");
            return (1.0 / (x * std::sqrt(2.0 * M_PI) * stdev) * std::exp(- (std::log(
                        x) - mean) * (std::log(x) - mean) / (2 * stdev * stdev)));
        }

        //// Descriptive stats
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

};

int main(int argc, char* argv[])
{
    word folder = "./ITHACAoutput/";
    // Create the Standard Normal Distribution and random draw vectors
    StandardNormalDistribution test(2.0, 1);
    VectorXd seeds = VectorXd::LinSpaced(20000, 0, 4);
    VectorXd pdf = seeds;

    // Output the values of the standard normal random draws
    for (int i = 0; i < seeds.size(); i++)
    {
        pdf(i) = test.pdf(seeds(i));
    }

    ITHACAstream::exportVector(pdf, "pdf", "eigen",
                               folder);
    ITHACAstream::exportVector(seeds, "seeds", "eigen",
                               folder);
    Info << "pdf.sum = " << pdf.sum() * seeds(1) << endl;
    
    mhmc markovChain(test, 10000, 1.0, 1.0, "normal");
    markovChain.createChain();
    Info << "Chain constructed"<< endl;
    markovChain.get_intAutocorrTime();
    Info << "intAutocorrTime = " << markovChain.intAutocorrTime << endl;
    
    VectorXd output = markovChain.chain;
    MatrixXd autocovariance = stochastic::autocovariance(markovChain.chain);

    ITHACAstream::exportVector(markovChain.chain, "output", "eigen",
                               folder);
    ITHACAstream::exportMatrix(markovChain.sampleMean, "sampleMean", "eigen",
                               folder);
    ITHACAstream::exportMatrix(autocovariance, "autocovariance", "eigen",
                               folder);
    
    return 0;
}


