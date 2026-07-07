"""provides measurement distributions for python estimators"""




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

        self.pValues = np.zeros(len(estimates)) 
        for i in range(len(estimates)):
            self.pValues[i] = computePValuesForGivenEstimate(estimates[i])

        self.estimates = estimates
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
            values = self.chi2dist.rvs(degreesOfFreedom, size=numObservations)
            mean = chi2.stats(degreesOfFreedom, moments='m')
            self.meanOfUncorrectedDistribution = self.chi2dist.stats(degreesOfFreedom, moments='m') / (2.0 * numObservations)
        else:
            self.chi2dist = 0.0
            self.meanOfUncorrectedDistribution = 0

        
        pValue = self.computePValueForGivenEstimate(actualValue)

        return pValue


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

        return 1 - cdf    

    def computeEstimateForGivenPValue(self, pValue):
        if self.chi2dist == 0.0:
            # All p-values map to estimate 0
            return 0

        uncorrectedEstimate = self.chi2dist.ppf((1 - pValue) / (2.0 * self.numObservations))
        if isBiasCorrected:
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

        std =  np.square(self.chi2dist.stats(moments='v') / (2.0 * self.numObservations))
        return std



