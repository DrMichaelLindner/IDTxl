

###############################
# Bayesian Discrete estimators
###############################

class PythonBayesian(PythonEstimator):
    ########################################################################### TODO
    def __init__(self, settings):
        settings.setdefault('discretise_method', 'none')

        settings.setdefault('approach', 'analytical')
        if settings['approach'] == 'analytical':
            settings.setdefault('dprior', 1.0)
        elif settings['approach'] == 'numerical':
            settings.setdefault('dprior', 0.5)
            settings.setdefault('nsamples', 4000)
        else:
            raise ValueError(f'Invalid approach setting {settings['approach']}. Need to be "analytical" or "numerical"')
        
        settings.setdefault('return_full_res', False)

        if settings['dprior'] < 0.0:
            raise ValueError(f'Invalid dprior setting {settings['dprior']}. dprior needs to be >= 0.0')


        settings.setdefault('base', 2.0)
        
        ############################################################## TODO
        settings.setdefault('local_values', False)
        super().__init__(settings)

    ################################################################# TODO
    def is_analytic_null_estimator(self):
        return False


class PythonBayesianDiscreteMI(PythonBayesian, PythonDiscrete):
    """Calculate MI with Python Baysian implementation using dirichlet prior method 
    (Analytical Approach) or Monte Carlo Method (Numerical Approach). Models the unknown probability distributions of categorical 
    variables using a conjugate Dirichlet prior. Adding pseudo-counts (e.g., α = 0.5 or α = 1)
    acts as a principled Laplace-style smoother to handle sparse data
    
    Calculate the mutual information between two variables  

    Args:
        settings : dict [optional]
            sets estimation parameters:
            
            - approach : string [optional] - 'analytical' for using fast dirichlet prior 
              method or 'numerical' for using the posterior Monte Carlo method
              if 'numerical' is used additionally
              - nsamples : int [optional] - number of Monte Carlo repetitions (default=4000)
            - dprior : float [optional] - Dirichlet prior strength (defaults: 'analytical': Laplace smoothing 
              dprior=1.0 'numerical': dprior=0.5).
            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and or 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph1, alph2
              and alphc
            - alph1 : int [optional] - number of discrete bins/levels for var1
              (default=2, or the value set for n_discrete_bins)
            - alph2 : int [optional] - number of discrete bins/levels for var2
              (default=2, or the value set for n_discrete_bins)
            - base : float [optiona] - 2.0 for returning cmi in bits (default) or
              np.e for returning nats

            ######################################################################## TODO
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
    """

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        
        ############################################################################## TODO remove
        if "local_values" in settings:
            raise ValueError("The analytical approach of this estimator currently does not support local_values arguments.")
        
        try:
            n_discrete_bins = int(settings['n_discrete_bins'])
            settings['alph1'] = n_discrete_bins
            settings['alph2'] = n_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        settings.setdefault('alph1', int(2))
        settings.setdefault('alph2', int(2))
        settings.setdefault('lag_mi', int(0))
        super().__init__(settings)


    def calculateAverageMI_analytical(self, var1, var2):
        """Bayesian MI for discrete variables using a symmetric Dirichlet(alpha) prior."""
        n = var1.shape[0]
        count_x = Counter(var1)
        count_y = Counter(var2)
        count_xy = Counter(zip(var1, var2))

        xs = list(count_x.keys())
        ys = list(count_y.keys())

        kx = len(xs)
        ky = len(ys)

        # posterior-mean probabilities under symmetric Dirichlet prior
        px = {a: (count_x[a] + self.settings['dprior']) / (n + self.settings['dprior'] * kx) for a in xs}
        py = {b: (count_y[b] + self.settings['dprior']) / (n + self.settings['dprior'] * ky) for b in ys}
        pxy_denom = n + self.settings['dprior'] * kx * ky
        pxy = {(a, b): (count_xy[(a, b)] + self.settings['dprior']) / pxy_denom for a in xs for b in ys}

        mi = 0.0
        for a in xs:
            for b in ys:
                p = pxy[(a, b)]

                mi += p * math.log(p / (px[a] * py[b]))

        return mi  / math.log(self.settings['base'])

    """
    def calculateLocalMI_analytical(self, var1, var2):
        #Bayesian local MI for discrete variables using a symmetric Dirichlet(alpha) prior.
        n = var1.shape[0]
        


        print(type(var1))
        print(var1.shape)

        def row_to_tuples(arr: np.ndarray):
            return [tuple(row) for row in arr]

        x_tuples = row_to_tuples(var1)
        y_tuples = row_to_tuples(var2)
        xy_tuples = list(zip(x_tuples, y_tuples))


        count_x = Counter(var1)
        count_y = Counter(var2)
        count_xy = Counter(zip(var1, var2))

        K = len(count_xy)

        # Precompute denominator for joint probability estimate
        denom_joint = n + K * self.settings['dprior']

        # Precompute marginal denominators
        Kx = len(count_x)
        Ky = len(count_y)
        denom_x = n + Kx * self.settings['dprior']
        denom_y = n + Ky * self.settings['dprior']

        def log(val: float) -> float:
            if val <= 0:
                return -np.inf
            return np.log(val) / np.log(self.settings['base'])

        

        lmi = np.empty(n, dtype=float)

        for s in range(n):
            xy = xy_tuples[s]
            x = x_tuples[s]
            y = y_tuples[s]

            n_xy = count_xy[xy]
            n_x = count_x[x]
            n_y = count_y[y]

            p_xy = (n_xy + alpha) / denom_joint
            p_x = (n_x + alpha) / denom_x
            p_y = (n_y + alpha) / denom_y

            # local MI: log p(x,y) / (p(x)p(y))
            if p_xy <= 0 or p_x <= 0 or p_y <= 0:
                lmi[s] = -np.inf
            else:
                lmi[s] = log(p_xy) - log(p_x) - log(p_y)

        return lmi
    """


    def dirichlet_sample(self, alphas):
        gammas = [random.gammavariate(a, 1.0) for a in alphas]
        s = sum(gammas)
        return [g / s for g in gammas]

    def mi_from_joint(self, px, py, pxy, xs, ys):
        mi = 0.0
        for i, a in enumerate(xs):
            for j, b in enumerate(ys):
                p = pxy[i][j]
                if p > 0:
                    mi += p * math.log(p / (px[i] * py[j]))
        return mi / math.log(self.settings['base'])

    def calculateLocalMI_numerical(self, var1, var2):
        """Bayesian MI for discrete variables using Monte Carlo method"""
        n = len(var1)
        cx = Counter(var1)
        cy = Counter(var2)
        cxy = Counter(zip(var1, var2))
        xs = list(cx.keys())
        ys = list(cy.keys())

        kx, ky = len(xs), len(ys)

        samples = []
        for _ in range(self.settings['nsamples']):
            px = self.dirichlet_sample([cx[a] + self.settings['dprior'] for a in xs])
            py = self.dirichlet_sample([cy[b] + self.settings['dprior'] for b in ys])
            flat = self.dirichlet_sample([cxy[(a, b)] + self.settings['dprior'] for a in xs for b in ys])
            pxy = [flat[i * ky:(i + 1) * ky] for i in range(kx)]
            samples.append(self.mi_from_joint(px, py, pxy, xs, ys))

        samples.sort()
        #mean = sum(samples) / self.settings['nsamples']
        lo = samples[int(0.025 * self.settings['nsamples'])]
        hi = samples[int(0.975 * self.settings['nsamples'])]
        
        #return mean, lo, hi
        return samples, lo, hi
        

    def estimate(self, var1, var2):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        
        assert (
            var1.shape[0] == var2.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]})"
        
        # Discretise variables if requested.
        var1, var2 = self._discretise_vars(var1, var2)

        # Then collapse any multivariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])
        
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi']]
            var2 = var2[self.settings['lag_mi']:]

        if self.settings['approach'] == 'analytical':
            if self.settings['local_values']:
                mi = self.calculateLocalMI_analytical(var1, var2)
            else:
                mi = self.calculateAverageMI_analytical(var1, var2)
        else:
            mi, lo, hi = self.calculateLocalMI_numerical(var1, var2)
            if not self.settings['local_values']:
                mi = np.mean(mi)

        return mi


################################################################ TODO
class PythonBayesianDiscreteCMI(PythonBayesian, PythonDiscrete):
    """Calculate CMI with Python Baysian implementation using dirichlet prior method 
    (Analytical Approach). Models the unknown probability distributions of categorical 
    variables using a conjugate Dirichlet prior. Adding pseudo-counts (e.g., α = 0.5 or α = 1)
    acts as a principled Laplace-style smoother to handle sparse data
    
    Calculate the conditional mutual information between two variables given
    the third. 

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - approach : string [optional] - 'analytical' (default) for using fast dirichlet prior 
              method or 'numerical' for using the posterior Monte Carlo method
              if 'numerical' is used additionally
              - nsamples : int [optional] - number of Monte Carlo repetitions (default=4000)
              - return_full_res : bool [optional] - If True an additional result output
                dictionary will be created including (default = False): 
                    - "posterior_samples": local cmi values
                    - "mean": mean of local cmi values
                    - "std": standard diviation of local cmi values
                    - "quantiles": quantiles of local cmi values
                    "observations": requested observations,
                    "categories": {
                    "var1": var1 values,
                    "var2": var2 values,
                    "conditional": conditional values
                    },
                    "posterior_joint_samples": joint_samples,
            - dprior : float [optional] - Dirichlet prior strength (defaults: 'analytical': Laplace smoothing 
              dprior=1.0 'numerical': dprior=0.5).

            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and or 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph1, alph2
              and alphc
            - alph1 : int [optional] - number of discrete bins/levels for var1
              (default=2, or the value set for n_discrete_bins)
            - alph2 : int [optional] - number of discrete bins/levels for var2
              (default=2, or the value set for n_discrete_bins)
            - alphc : int [optional] - number of discrete bins/levels for
              conditional (default=2, or the value set for n_discrete_bins)
            - base : float [optiona] - 2.0 for returning cmi in bits (default) or
              np.e for returning nats
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)

    """
    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        settings.setdefault('alphc', int(2))
        
        try:
            n_discrete_bins = int(settings['n_discrete_bins'])
            settings['alph1'] = n_discrete_bins
            settings['alph2'] = n_discrete_bins
            settings['alphc'] = n_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        settings.setdefault('alph1', int(2))
        settings.setdefault('alph2', int(2))
        settings.setdefault('alphc', int(2))
        super().__init__(settings)


    def calculateLocalCMI_numerical(self, var1, var2, conditional):
        """Bayesian posterior sampling of conditional mutual information I(X;Y|Z)
        for discrete data under a Dirichlet-Multinomial model."""
        
        n = var1.shape[0]

        var1_values = np.unique(var1)
        var2_values = np.unique(var2)
        cond_values = np.unique(conditional)

        n1 = len(var1_values)
        n2 = len(var2_values)
        nc = len(cond_values)

        counts = np.zeros((n1, n2, nc), dtype=np.float64)
        np.add.at(counts,(var1, var2, conditional), 1.0)

        prior = np.full_like(counts, self.settings['dprior'])
        
        posterior_parameters = prior + counts
        posterior_parameters = posterior_parameters.reshape(-1)
        
        random_state=None
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        # Gamma normalization is equivalent to np.random.dirichlet and makes
        # the vectorized multidimensional shape explicit.
        gamma_draws = rng.gamma(
            shape=posterior_parameters,
            scale=1.0,
            size=(self.settings['nsamples'], posterior_parameters.size),
        )

        joint_samples = gamma_draws / gamma_draws.sum(axis=1, keepdims=True)
        joint_samples = joint_samples.reshape(self.settings['nsamples'], n1, n2, nc)

        requested = np.column_stack((var1, var2, conditional))

        # Marginals for every posterior draw.
        pxz = joint_samples.sum(axis=2)          # p(x,z)
        pyz = joint_samples.sum(axis=1)          # p(y,z)
        pz = joint_samples.sum(axis=(1, 2))       # p(z)

        local_samples = np.empty((self.settings['nsamples'], len(requested)), dtype=np.float64)

        for j, (xi, yi, zi) in enumerate(requested):
            numerator = joint_samples[:, xi, yi, zi] * pz[:, zi]
            denominator = pxz[:, xi, zi] * pyz[:, yi, zi]

            local_samples[:, j] = np.log(numerator / denominator)

        if self.settings['return_full_res']:
            quantiles = np.quantile(
                local_samples,
                [0.025, 0.5, 0.975],
                axis=0,
            )
            
            full_res = {"posterior_samples": local_samples,
                        "mean": local_samples.mean(axis=0),
                        "std": local_samples.std(axis=0, ddof=1),
                        "quantiles": quantiles,
                        "observations": requested,
                        "categories": {
                        "var1": var1_values,
                        "var2": var2_values,
                        "conditional": cond_values
                        },
                        "posterior_joint_samples": joint_samples,
                        }

            return local_samples, full_res
        
        else:
            return local_samples


    def calculateLocalCMI_analytical(self, var1, var2, conditional):
        """
        Bayesian local conditional mutual information for discrete data.
        """
        n = len(var1)
        if n == 0:
            return [], 0.0

        # Support sets
        x_vals = sorted(set(var1))
        y_vals = sorted(set(var2))
        z_vals = sorted(set(conditional))

        # Count tables
        c_xyz = defaultdict(int)
        c_xz = defaultdict(int)
        c_yz = defaultdict(int)
        c_z = defaultdict(int)

        for xi, yi, zi in zip(var1, var2, conditional):
            c_xyz[(xi, yi, zi)] += 1
            c_xz[(xi, zi)] += 1
            c_yz[(yi, zi)] += 1
            c_z[zi] += 1

        K_xyz = len(x_vals) * len(y_vals) * len(z_vals)
        K_xz = len(x_vals) * len(z_vals)
        K_yz = len(y_vals) * len(z_vals)
        K_z = len(z_vals)

        logb = lambda t: math.log(t) / math.log(self.settings['base'])

        local_values = []
        for xi, yi, zi in zip(var1, var2, conditional):
            p_xyz = (c_xyz[(xi, yi, zi)] + self.settings['dprior']) / (n + self.settings['dprior'] * K_xyz)
            p_xz  = (c_xz[(xi, zi)] + self.settings['dprior']) / (n + self.settings['dprior'] * K_xz)
            p_yz  = (c_yz[(yi, zi)] + self.settings['dprior']) / (n + self.settings['dprior'] * K_yz)
            p_z   = (c_z[zi] + self.settings['dprior']) / (n + self.settings['dprior'] * K_z)

            p_xy_given_z = p_xyz / p_z
            p_x_given_z = p_xz / p_z
            p_y_given_z = p_yz / p_z

            local = logb(p_xy_given_z / (p_x_given_z * p_y_given_z))
            local_values.append(local)

        return local_values

    def estimate(self, var1, var2, conditional=None):
        """Estimate bayesian conditional mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2
            
        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
            
        """
        # Return MI if no conditioning variable was provided.
        if conditional is None:
            #if (self.est_mi is None):
            self.est_mi = PythonBayesianDiscreteMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        assert (
            var1.shape[0] == var2.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]}, conditional: {conditional.shape[0]})"

        # Discretise if requested.
        var1, var2, conditional = self._discretise_vars(var1, var2,
                                                        conditional)

        # Then collapse any mulitvariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])
        conditional = utils.combine_discrete_dimensions(conditional,
                                                        self.settings['alphc'])

        var1 = self._ensure_one_dim_input(var1)
        var2 = self._ensure_one_dim_input(var2)
        conditional = self._ensure_one_dim_input(conditional)

        if self.settings['approach'] == 'analytical':            
            cmi = self.calculateLocalCMI_analytical(var1, var2, conditional)
            if not self.settings['local_values']:
                cmi = np.mean(cmi)
        else:
            cmi = self.calculateLocalCMI_numerical(var1, var2, conditional)
            if not self.settings['local_values']:
                cmi = np.mean(cmi)

        return cmi


