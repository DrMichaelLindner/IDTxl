"""provides measurement distributions for python estimators"""

import numpy as np
from scipy.stats import chi2, gamma
import math

################################################################ TODO
class EmpiricalMeasurementDistribution():
    """
    ######################################################## TODO
    """

    
    def new_dist(self, numPermutationsToCheck):
        """Create new empty distribtution array"""
        self.distribution = np.zeros(numPermutationsToCheck)

    def EmpiricalMeasurementDistribution(self, distribution, actualValue):
        self.distribution = distribution
        countWhereActualIsNotGreater = 0;
        for i in range(len(distribution)):
            if distribution[i] >= actualValue:
                countWhereActualIsNotGreater += 1
        pValue = countWhereActualIsNotGreater / len(distribution);
        return pValue

    def getTScore(self):
        """get T score of actual value"""
        meanOfDist = np.mean(self.distribution)
        stdOfDist = np.std(self.distribution)

        t = (actualValue - meanOfDist) / stdOfDist
        return t

    def getMeanOfDistribution(self):
        meanOfDist = np.mean(self.distribution)
        return meanOfDist

    def getStdOfDistribution(self):
        stdOfDist = np.std(self.distribution)
        return stdOfDist

    ######################################################################### TODO
    def generateRandomPerturbations(self, n, numberOfPerturbations):
        """Generate numberOfPerturbations perturbations of [0..n-1],
        which are not necessarily distinct."""

        sets = np.zeros((n, numberOfPerturbations))
        for i in range(numberOfPerturbations):
            sets[:,i]=np.random.permutation(n)

        return sets


################################################################ TODO
class AnalyticalMeasurementDistribution():

    def AnalyticalMeasurementDistribution(self, actualValue, pValue):

        self.actualValue = actualValue
        self.pValue = pValue


    def computePValuesForGivenEstimates(self, estimates):

        pValues = np.zeros(len(estimates)) 
        for i in range(len(estimates)):
            pValues[i] = computePValueForGivenEstimate(estimates[i])

        return pValues

        
    def computeEstimatesForGivenPValues(self, pValues):
        
        estimates = np.zeros(len(pValues)) 
        for i in range(len(pValues)):
            estimates[i] = computeEstimateForGivenPValue(pValues[i])

        return estimates
       

################################################################ TODO
class ChiSquareMeasurementDistribution(AnalyticalMeasurementDistribution):


    def ChiSquareMeasurementDistribution(self, actualValue, numObservations, degreesOfFreedom, isBiasCorrected):

        self.actualValue = actualValue
        self.numObservations = numObservations
        self.degreesOfFreedom = degreesOfFreedom
        self.isBiasCorrected = isBiasCorrected

        if degreesOfFreedom > 0:
            self.chi2dist = chi2
            #self.chi2dist = gamma
            #values = self.chi2dist.rvs(degreesOfFreedom, size=numObservations)
            #mean = chi2.stats(degreesOfFreedom, moments='m')
            self.meanOfUncorrectedDistribution = self.chi2dist.stats(self.degreesOfFreedom, moments='m') / (2.0 * numObservations)
        else:
            self.chi2dist = 0.0
            self.meanOfUncorrectedDistribution = 0

        
        self.pValue = self.computePValueForGivenEstimate(actualValue)

        #return pValue


    def computePValueForGivenEstimate(self, estimate):
        if self.chi2dist == 0.0:
            if estimate > 0:
                return 1
            else:
                return 0
        if self.isBiasCorrected:
            cdf = self.chi2dist.cdf(2.0 * self.numObservations * (estimate + self.meanOfUncorrectedDistribution), self.degreesOfFreedom)
        else:
            cdf = self.chi2dist.cdf(2.0 * self.numObservations * estimate, self.degreesOfFreedom)
            #cdf = self.chi2dist.cdf(2.0 * self.numObservations * estimate, self.degreesOfFreedom/2, loc=0, scale=2)
            
        return 1 - cdf    

    def computeEstimateForGivenPValue(self, pValue):
        if self.chi2dist == 0.0:
            # All p-values map to estimate 0
            return 0

        #uncorrectedEstimate = self.chi2dist.ppf((1 - pValue) / (2.0 * self.numObservations), df = self.degreesOfFreedom)
        
        print(pValue, self.degreesOfFreedom)
        print(type(pValue))
        uncorrectedEstimate = self.chi2_ppf_cheb(pValue, self.degreesOfFreedom)
        

        #uncorrectedEstimate = self.chi2dist.ppf((1 - pValue) / (2.0 * self.numObservations), self.degreesOfFreedom/2, scale=2)
        #uncorrectedEstimate = self.chi2dist.ppf((1 - pValue) / (2.0 * self.numObservations), 1.8, scale=2)
        #uncorrectedEstimate = self.chi2dist.ppf((1- pValue) , self.degreesOfFreedom/2, scale=2)
        
        if self.isBiasCorrected:
            return uncorrectedEstimate - self.meanOfUncorrectedDistribution
        else:
            return uncorrectedEstimate;



    def getMeanOfDistribution(self):
        if self.isBiasCorrected:
            return 1
        else:
            return self.meanOfUncorrectedDistribution

    def getMeanOfUncorrectedDistribution(self):
        return self.meanOfUncorrectedDistribution

    def getStdOfDistribution(self):
        if self.chi2dist == 0.0:
            return 0

        std =  np.square(self.chi2dist.stats(self.degreesOfFreedom, moments='v') / (2.0 * self.numObservations))
        return std


    def chi2_ppf_cheb(self, p: float, df: int, deg=40, p_min=1e-6, p_max=1 - 1e-6, bisect_tol=1e-10):
        """
        Inverse CDF (quantile/PPF) of chi-square distribution using a Chebyshev
        approximation on [p_min, p_max], with bisection fallback outside.

        Parameters
        ----------
        p : float or array_like
            Probability values in (0, 1).
        df : float
            Degrees of freedom (> 0).
        deg : int
            Degree (number of nodes) for Chebyshev approximation.
        p_min, p_max : float
            Interval on which the Chebyshev approximation is valid.
        bisect_tol : float
            Tolerance for bisection when outside [p_min, p_max].

        Returns
        -------
        x : float or ndarray
            Quantile(s) such that P(Chi2(df) <= x) = p.
        """

        print(p)
        print(type(p))
        

        scalar = np.ndim(p) == 0
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        

        print(p)
        print(p_arr)

        if np.any((p_arr <= 0.0) | (p_arr >= 1.0)):
            raise ValueError("p must be in (0, 1).")
        if df <= 0:
            raise ValueError("df must be > 0.")

        # Build Chebyshev approximation to chi2.ppf on [p_min, p_max]
        cheb = ChebyshevApprox(
            a=p_min,
            b=p_max,
            n=deg,
            func=lambda pp: chi2.ppf(pp, df),
        )

        def ppf_one(pp):
            if p_min <= pp <= p_max:
                return cheb.eval(pp)
            # Fallback: bisection using chi2.cdf
            # Find bounds [lo, hi] such that cdf(lo) <= pp <= cdf(hi)
            lo = 0.0
            hi = max(df, 1.0)
            while chi2.cdf(hi, df) < pp:
                hi *= 2.0
            # Bisection
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                c = chi2.cdf(mid, df)
                if abs(hi - lo) < bisect_tol:
                    return mid
                if c < pp:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        res = np.array([ppf_one(pp) for pp in p_arr])
        return res[0] if scalar else res


class ChebyshevApprox:
    """
    Chebyshev approximation of a function f on [a, b] with degree n.
    After construction, call eval(x) to get the approximated value.
    """
    def __init__(self, a, b, n, func):
        """
        a, b: interval endpoints
        n: number of Chebyshev nodes (degree ~ n-1)
        func: callable, func(x) defined on [a, b]
        """
        self.a = a
        self.b = b
        self.n = n
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)

        # Function values at Chebyshev nodes
        f = [
            func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa)
            for k in range(n)
        ]

        # Chebyshev coefficients
        fac = 2.0 / n
        self.c = [
            fac * sum(
                f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                for k in range(n)
            )
            for j in range(n)
        ]

    def eval(self, x):
        """Evaluate Chebyshev approximation at x in [a, b]."""
        a, b = self.a, self.b
        if not (a <= x <= b):
            raise ValueError(f"x={x} out of Chebyshev interval [{a}, {b}]")
        y = (2.0 * x - a - b) / (b - a)  # map [a,b] -> [-1,1]
        y2 = 2.0 * y
        d, dd = self.c[-1], 0.0
        # Clenshaw recurrence
        for cj in self.c[-2:0:-1]:
            d, dd = y2 * d - dd + cj, d
        return y * d - dd + 0.5 * self.c[0]



