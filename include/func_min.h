inline double lagrange_poly_min_extrap (
    double p1, 
    double p2,
    double p3,
    double f1,
    double f2,
    double f3
)
{
    M_Assert(p1 < p2 && p2 < p3 && f1 >= f2 && f2 <= f3, "Invalid input parameters");

    // This formula is out of the book Nonlinear Optimization by Andrzej Ruszczynski.  See section 5.2.
    double temp1 =    f1*(p3*p3 - p2*p2) + f2*(p1*p1 - p3*p3) + f3*(p2*p2 - p1*p1);
    double temp2 = 2*(f1*(p3 - p2)       + f2*(p1 - p3)       + f3*(p2 - p1) );

    if (temp2 == 0)
    {
        return p2;
    }

    const double result = temp1/temp2;

    // do a final sanity check to make sure the result is in the right range
    if (p1 <= result && result <= p3)
    {
        return result;
    }
    else
    {
        return std::min(std::max(p1,result),p3);
    }
}

// ----------------------------------------------------------------------------------------


template <typename object, typename funct>
double find_min_single_variable (
    const object& obj,
    const funct& f,
    double& starting_point,
    const double begin = -1e200,
    const double end = 1e200,
    const double eps = 1e-3,
    const long max_iter = 100,
    const double initial_search_radius = 1
)
{
    double search_radius = initial_search_radius;

    double p1=0, p2=0, p3=0, f1=0, f2=0, f3=0;
    long f_evals = 1;

    if (begin == end)
    {
        return (obj->*f)(starting_point);
    }
    using std::abs;
    using std::min;
    using std::max;

    // find three bracketing points such that f1 > f2 < f3.   Do this by generating a sequence
    // of points expanding away from 0.   Also note that, in the following code, it is always the
    // case that p1 < p2 < p3.

    // The first thing we do is get a starting set of 3 points that are inside the [begin,end] bounds
    p1 = max(starting_point-search_radius, begin);
    p3 = min(starting_point+search_radius, end);
    f1 = (obj->*f)(p1);
    f3 = (obj->*f)(p3);

    if (starting_point == p1 || starting_point == p3)
    {
        p2 = (p1+p3)/2;
        f2 = (obj->*f)(p2);
    }
    else
    {
        p2 = starting_point;
        f2 = (obj->*f)(starting_point);
    }

    f_evals += 2;

    // Now we have 3 points on the function.  Start looking for a bracketing set such that
    // f1 > f2 < f3 is the case.
    while ( !(f1 > f2 && f2 < f3))
    {
        // check for hitting max_iter or if the interval is now too small
        if (f_evals >= max_iter)
        {
            std::cout << "The max number of iterations of single variable optimization have been reached\n"
               << "without converging." << std::endl;;
        }
        if (p3-p1 < eps)
        {
            if (f1 < min(f2,f3)) 
            {
                starting_point = p1;
                return f1;
            }

            if (f2 < min(f1,f3)) 
            {
                starting_point = p2;
                return f2;
            }

            starting_point = p3;
            return f3;
        }
        
        // If the left most points are identical in function value then expand out the
        // left a bit, unless it's already at bound or we would drop that left most
        // point anyway because it's bad.
        if (f1==f2 && f1<f3 && p1!=begin)
        {
            p1 = max(p1 - search_radius, begin);
            f1 = (obj->*f)(p1);
            ++f_evals;
            search_radius *= 2;
            continue;
        }
        if (f2==f3 && f3<f1 && p3!=end)
        {
            p3 = min(p3 + search_radius, end);
            f3 = (obj->*f)(p3);
            ++f_evals;
            search_radius *= 2;
            continue;
        }


        // if f1 is small then take a step to the left
        if (f1 <= f3)
        { 
            // check if the minimum is butting up against the bounds and if so then pick
            // a point between p1 and p2 in the hopes that shrinking the interval will
            // be a good thing to do.  Or if p1 and p2 aren't differentiated then try and
            // get them to obtain different values.
            if (p1 == begin || (f1 == f2 && (end-begin) < search_radius ))
            {
                p3 = p2;
                f3 = f2;

                p2 = (p1+p2)/2.0;
                f2 = (obj->*f)(p2);
            }
            else
            {
                // pick a new point to the left of our current bracket
                p3 = p2;
                f3 = f2;

                p2 = p1;
                f2 = f1;

                p1 = max(p1 - search_radius, begin);
                f1 = (obj->*f)(p1);

                search_radius *= 2;
            }

        }
        // otherwise f3 is small and we should take a step to the right
        else 
        {
            // check if the minimum is butting up against the bounds and if so then pick
            // a point between p2 and p3 in the hopes that shrinking the interval will
            // be a good thing to do.  Or if p2 and p3 aren't differentiated then try and
            // get them to obtain different values.
            if (p3 == end || (f2 == f3 && (end-begin) < search_radius))
            {
                p1 = p2;
                f1 = f2;

                p2 = (p3+p2)/2.0;
                f2 = (obj->*f)(p2);
            }
            else
            {
                // pick a new point to the right of our current bracket
                p1 = p2;
                f1 = f2;

                p2 = p3;
                f2 = f3;

                p3 = min(p3 + search_radius, end);
                f3 = (obj->*f)(p3);

                search_radius *= 2;
            }
        }

        ++f_evals;
    }


    // Loop until we have done the max allowable number of iterations or
    // the bracketing window is smaller than eps.
    // Within this loop we maintain the invariant that: f1 > f2 < f3 and p1 < p2 < p3
    const double tau = 0.1;
    while( f_evals < max_iter && p3-p1 > eps)
    {
        double p_min = lagrange_poly_min_extrap(p1,p2,p3, f1,f2,f3);

        // make sure p_min isn't too close to the three points we already have
        if (p_min < p2)
        {
            const double min_dist = (p2-p1)*tau;
            if (abs(p1-p_min) < min_dist) 
            {
                p_min = p1 + min_dist;
            }
            else if (abs(p2-p_min) < min_dist)
            {
                p_min = p2 - min_dist;
            }
        }
        else
        {
            const double min_dist = (p3-p2)*tau;
            if (abs(p2-p_min) < min_dist) 
            {
                p_min = p2 + min_dist;
            }
            else if (abs(p3-p_min) < min_dist)
            {
                p_min = p3 - min_dist;
            }
        }

        // make sure one side of the bracket isn't super huge compared to the other
        // side.  If it is then contract it.
        const double bracket_ratio = abs(p1-p2)/abs(p2-p3);
        // Force p_min to be on a reasonable side.  But only if lagrange_poly_min_extrap()
        // didn't put it on a good side already.
        if (bracket_ratio >= 10)
        { 
            if (p_min > p2)
                p_min = (p1+p2)/2;
        }
        else if (bracket_ratio <= 0.1) 
        {
            if (p_min < p2)
                p_min = (p2+p3)/2;
        }


        const double f_min = (obj->*f)(p_min);


        // Remove one of the endpoints of our bracket depending on where the new point falls.
        if (p_min < p2)
        {
            if (f1 > f_min && f_min < f2)
            {
                p3 = p2;
                f3 = f2;
                p2 = p_min;
                f2 = f_min;
            }
            else
            {
                p1 = p_min;
                f1 = f_min;
            }
        }
        else
        {
            if (f2 > f_min && f_min < f3)
            {
                p1 = p2;
                f1 = f2;
                p2 = p_min;
                f2 = f_min;
            }
            else
            {
                p3 = p_min;
                f3 = f_min;
            }
        }


        ++f_evals;
    }

    if (f_evals >= max_iter)
    {
        std::cout << "The max number of iterations of single variable optimization have been reached\n"
            << "without converging." << std::endl;
    }

    starting_point = p2;
    return f2;
}
