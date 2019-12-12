#ifndef MINI_INVERSE_PROBLEMS_H
#define MINI_INVERSE_PROBLEMS_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <sys/stat.h>
#include <unistd.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic pop
#include <functional>
#include <chrono>
#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;


#include "M_assert.h"
#include "erfinv.h"
#include "EigenStream.h"
#include "func_min.h"
#include "stochastic.h"
#include "statisticalDistribution.h"
#include "linearSystem.h"
#include "markovChain.h"
#include "multiMarkovChain.h"

#endif // MINI_INVERSE_PROBLEMS_H
