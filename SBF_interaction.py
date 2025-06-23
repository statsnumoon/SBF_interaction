from utils import *

class BSBF_inter:
    """
    BSBF_inter: Code for "Identification and estimation of interaction effects in nonparametric additive regression"

    This class implements a kernel-based smooth backfitting procedure for additive regression models
    with interaction effects. It simultaneously estimates main effects and pairwise interactions
    while performing variable selection through regularization.

    Core methodology:
    - Kernel-based estimation of 1D, 2D, 3D, and 4D marginal and joint densities
    - Simple projection and centering for identifying additive and interaction components

    Parameters:
    - path: string, directory path to store intermediate KDE outputs
    - X: array-like of shape (n, d), input covariates (should be scaled to [0,1]^d if is_scaling is False)
    - Y: array-like of shape (n, 1), response vector
    - h: array of length d, bandwidth vector for KDE
    - norm: callable, norm function used for shrinkage (default: L2 norm)
    - ngrid: int, number of evaluation grid points per covariate
    - is_scaling: bool, whether to apply MinMax scaling to X
    - Kdenom_method: str, 'numeric' or 'exact' method for KDE normalization
    - max_iter: int, maximum number of backfitting iterations
    - L2_epsilon: float, convergence threshold for iterative updates
    - verbose: bool, whether to print iteration logs

    Key methods:
    - save_kde(): Computes and saves 3D and 4D KDE tensors required for projection
    - fit(): Fits main and interaction functions; returns selected variables
    - predict(): Computes predicted value when covariate values are given
    """

    def __init__(self,path,X,Y,h,norm=np.linalg.norm,ngrid=101,is_scaling=False,Kdenom_method='numeric',max_iter=100,L2_epsilon=1e-8,verbose=True):   
        """
        Initialize the BSBF_inter object for additive regression with interaction effects.
    
        Parameters:
        - path: str, directory path to save KDE results
        - X: (n, d) ndarray, covariate matrix
        - Y: (n, 1) ndarray, response vector
        - h: (d,) array, bandwidths for each covariate
        - norm: function, norm function for regularization (default: L2)
        - ngrid: int, number of grid points per covariate (for KDE and projection)
        - is_scaling: bool, if True applies MinMaxScaler to X
        - Kdenom_method: str, method to normalize kernel: 'numeric' or 'exact'
        - max_iter: int, maximum number of iterations for projection
        - L2_epsilon: float, stopping criterion threshold for iterative updates
        - verbose: bool, controls whether to print convergence logs
    
        Initializes:
        - self.kvalues: list of kernel weights (d, g, n)
        - self.kde_1d: 1D marginal density estimates (d, g, 1)
        - self.kde_2d: 2D joint density estimates (d, d, g, g)
        """
        self.is_scaling = is_scaling
        if self.is_scaling:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
            
        self.is_scaling = is_scaling
        self.path = path
        self.X = X
        self.Y = Y    
        self.barY = Y.mean()
        self.h = h
        self.norm = norm
        self.is_scaling=is_scaling
        self.Kdenom_method = Kdenom_method
        self.max_iter = max_iter
        self.L2_epsilon = L2_epsilon        
        self.verbose = verbose

        g = ngrid
        n,d = X.shape
        
        self.d = d
        self.g = g
        self.n = n
        self.ngrid = ngrid
        
        x = np.array([np.linspace(0,1,ngrid) for _ in range(d)])
        
        kvalues = Kh(x,X,h) + 1e-8

        if Kdenom_method=='numeric':
            kdenom_all = np.array([np.trapz(kvalues[j],x[j],axis=0) for j in range(d)])
        elif Kdenom_method=='exact':
            kdenom_all = Kdenom_exact(x,X,h)

        for j in range(d):
            kvalues[j] = kvalues[j]/kdenom_all[j]
        
        # 1d kde
        kde_1d = np.zeros((d,g,1))
        for j in range(d):
            kde_j = kvalues[j] @ np.ones((n,1))/n
            kde_j[kde_j==0] = 0
            integrand = np.trapz(kde_j,x[j],axis=0)
            kde_1d[j] = kde_j/integrand
        
        # 2d kde
        kde_2d = np.zeros((d,d,g,g))
        for j in range(d):
            for k in range(j+1,d):
                kde_jk = (kvalues[j] @ kvalues[k].T)/n
                kde_jk[kde_jk==0] = 0
                kde_2d[j,k] = kde_jk
                kde_2d[k,j] = kde_jk.T
                
        self.kvalues = kvalues
        self.kde_1d = kde_1d
        self.kde_2d = kde_2d
        
    def save_kde(self):
        """
        Efficient computation and storage of 3D and 4D kernel density estimates (KDEs)
        needed for simple projection in high-dimensional regression.
        Results are saved as .npy files in subfolders '3d-kde/' and '4d-kde/'.
        """
        os.makedirs(self.path, exist_ok=True)
        
        path = self.path
        X = self.X
        h = self.h
        ngrid = self.g
        kvalues = self.kvalues
        n, d = X.shape
    
        # 3D KDE computation
        print('3d-KDE is started...')
        for j, k, l in tqdm([(j, k, l) for j in range(d) for k in range(j+1, d) for l in range(k+1, d)]):
            tmp_jk = np.einsum('ij,jk->ijk', kvalues[k], kvalues[l].T)  # (g, n, g)
            kde_jkl = np.einsum('ij,kjl->ikl', kvalues[j], tmp_jk) / n  # (g, g, g)
            kde_jkl[kde_jkl == 0] = 0
            pathjkl = os.path.join(path, f'3d-kde/{(j, k, l)}')
            os.makedirs(pathjkl, exist_ok=True)
            np.save(os.path.join(pathjkl, 'kde_3d.npy'), kde_jkl)
    
        # 4D KDE computation
        print('4d-KDE is started...')
        for j, k, l, r in tqdm([(j, k, l, r) for j in range(d) for k in range(j+1, d) for l in range(k+1, d) for r in range(l+1, d)]):
            tmp_jk = np.einsum('ij,jk->ijk', kvalues[j], kvalues[k].T)  # (g, n, g)
            tmp_lr = np.einsum('ij,jk->ijk', kvalues[l], kvalues[r].T)  # (g, n, g)
            kde_jklr = np.einsum('ijl,rjm->ilrm', tmp_jk, tmp_lr) / n    # (g, g, g, g)
            kde_jklr[kde_jklr == 0] = 0
            pathjklr = os.path.join(path, f'4d-kde/{(j, k, l, r)}')
            os.makedirs(pathjklr, exist_ok=True)
            np.save(os.path.join(pathjklr, 'kde_4d.npy'), kde_jklr)
            
    def fit(self):
        """    
        Returns:
        - self: updated model object
        - self.barY: global sample mean of Y
        - self.hatm_main: estimated additive main functions, shape (d, g)
        - self.hatm_int: estimated bivariate interaction functions, shape (d, d, g, g)
        - self.idx_main: list of selected main indices
        - self.idx_int: list of selected interaction index pairs
        """
        print("Start Fitting...")
        
        # Load data and setup
        X = self.X
        Y = self.Y
        barY = Y.mean()
        h = self.h
        g = self.g
        n = self.n
        d = self.d
        path = self.path
        max_iter = self.max_iter
        L2_epsilon = self.L2_epsilon        
        verbose = self.verbose
        ngrid = self.ngrid
        kvalues = self.kvalues
        kde_1d = self.kde_1d
        kde_2d = self.kde_2d
    
        x = np.array([np.linspace(0, 1, ngrid) for _ in range(d)])
    
        # Step 1: Initialize interaction mean estimator hatmu_int (d,d,g,g)
        is_conv = 0
        hatmu_int = np.zeros((d,d,g,g))
        for j in range(d):
            for k in range(j+1,d):
                hatpjk = kde_2d[j,k]
                hatmu_int[j,k] = (np.einsum('ijk,j -> ik',
                                            np.einsum('ij,jk -> ijk', kvalues[j], kvalues[k].T),
                                            Y.reshape(-1) - barY) / n) / hatpjk
    
        # Step 2: Iteratively refine interaction estimators
        hatm_int = np.zeros((max_iter+1,d,d,g,g))
        it = 0
        while it < max_iter:
            start_time = time.time()
            it += 1
            hatm_int[it] = hatmu_int
    
            for j in range(d):
                for k in range(j+1, d):
                    hatpjk = kde_2d[j,k]
                    for l in range(d):
                        for r in range(l+1, d):
    
                            # Case: all indices disjoint (j,k) ∩ (l,r) = ∅
                            if j not in (l,r) and k not in (l,r):
                                (j_org, k_org, l_org, r_org), (j_idx, k_idx, l_idx, r_idx) = get_idx(np.array([j,k,l,r]))
                                pathjklr = path + f'4d-kde/{(j_org, k_org, l_org, r_org)}/'
                                hatpjklr = np.load(pathjklr + 'kde_4d.npy').transpose((j_idx, k_idx, l_idx, r_idx))
                                if l < j:
                                    hatm_int[it,j,k] -= np.trapz(np.trapz(hatm_int[it,l,r][np.newaxis,np.newaxis,...] * hatpjklr,
                                                                          x[l], axis=2), x[r], axis=2) / hatpjk
                                elif l > j:
                                    hatm_int[it,j,k] -= np.trapz(np.trapz(hatm_int[it-1,l,r][np.newaxis,np.newaxis,...] * hatpjklr,
                                                                          x[l], axis=2), x[r], axis=2) / hatpjk
    
                            # Case: j == l and k ≠ r
                            elif j == l and k != r:
                                (j_org, k_org, r_org), (j_idx, k_idx, r_idx) = get_idx(np.array([j,k,r]))
                                pathjkr = path + f'3d-kde/{(j_org, k_org, r_org)}/'
                                hatpjkr = np.load(pathjkr + 'kde_3d.npy').transpose((j_idx, k_idx, r_idx))
                                if r < k:
                                    hatm_int[it,j,k] -= np.trapz(hatm_int[it,l,r][:,np.newaxis] * hatpjkr, x[r], axis=2) / hatpjk
                                if r > k:
                                    hatm_int[it,j,k] -= np.trapz(hatm_int[it-1,l,r][:,np.newaxis] * hatpjkr, x[r], axis=2) / hatpjk
    
                            # Case: j == r
                            elif j == r:
                                (l_org, j_org, k_org), (l_idx, j_idx, k_idx) = get_idx(np.array([l,j,k]))
                                pathljk = path + f'3d-kde/{(l_org, j_org, k_org)}/'
                                hatpljk = np.load(pathljk + 'kde_3d.npy').transpose((l_idx, j_idx, k_idx))
                                hatm_int[it,j,k] -= np.trapz(hatm_int[it,l,r][...,np.newaxis] * hatpljk, x[l], axis=0) / hatpjk
    
                            # Case: k == l
                            elif k == l:
                                (j_org, k_org, r_org), (j_idx, k_idx, r_idx) = get_idx(np.array([j,k,r]))
                                pathjkr = path + f'3d-kde/{(j_org, k_org, r_org)}/'
                                hatpjkr = np.load(pathjkr + 'kde_3d.npy').transpose((j_idx, k_idx, r_idx))
                                hatm_int[it,j,k] -= np.trapz(hatm_int[it-1,l,r][np.newaxis,...] * hatpjkr, x[r], axis=2) / hatpjk
    
                            # Case: j ≠ l and k == r
                            elif j != l and k == r:
                                (l_org, j_org, k_org), (l_idx, j_idx, k_idx) = get_idx(np.array([l,j,k]))
                                pathljk = path + f'3d-kde/{(l_org, j_org, k_org)}/'
                                hatpljk = np.load(pathljk + 'kde_3d.npy').transpose((l_idx, j_idx, k_idx))
                                if l < j:
                                    hatm_int[it,j,k] -= np.trapz(hatm_int[it,l,r][:,np.newaxis,:] * hatpljk, x[l], axis=0) / hatpjk
                                if l > j:
                                    hatm_int[it,j,k] -= np.trapz(hatm_int[it-1,l,r][:,np.newaxis,:] * hatpljk, x[l], axis=0) / hatpjk
    
            # Check convergence
            L2 = np.sum(np.array([
                np.trapz(np.trapz((hatm_int[it-1,j,k] - hatm_int[it,j,k])**2 * kde_2d[j,k],
                                  x[j], axis=0), x[k], axis=0)
                for j in range(d) for k in range(j+1,d)
            ]))
            if verbose:
                print(f'iter: {it}, time: {time.time()-start_time:.2f}, L2: {L2:.2e}')
            if L2 < L2_epsilon:
                if verbose:
                    print('SBF converges')
                is_conv = 1
                break
    
        # Step 3: Center interaction terms and extract additive parts
        hatm_main_f = np.zeros((d,g))
        hatm_int_f = np.zeros((d,d,g,g))
        for j in range(d):
            for k in range(j+1,d):
                hatmjk_cen, hatmj_cen, hatmk_cen = inter_center(hatm_int[it,j,k], X[:,[j,k]], ngrid)
                hatm_int_f[j,k] = hatmjk_cen
                hatm_main_f[j] += hatmj_cen
                hatm_main_f[k] += hatmk_cen
    
        self.niter = it
        self.hatm_main = hatm_main_f
        self.hatm_int = hatm_int_f
        self.idx_main = list(range(d))
        self.idx_int = [(j,k) for j in range(d) for k in range(j+1,d)]
    
        return self, self.barY, self.hatm_main, self.hatm_int, self.idx_main, self.idx_int
    
    def predict(self, x_grid):
        """
        Parameters:
        - x_grid: ndarray of shape (d,) or (n, d), input covariate(s) on [0,1]^d scale

        Returns:
        - predicted values: ndarray of shape (1,) or (n,1)
        """

        # Use fitted main and interaction components
        if self.hatm_main is not None and self.hatm_int is not None:
            hatm_main, hatm_int = self.hatm_main, self.hatm_int
        else:
            _, hatm_main, hatm_int, idx_main, idx_int = self.fit()

        d = self.d
        func_val = 0

        # Case 1: Single input (x_grid has shape (d,))
        if x_grid.ndim == 1:
            for j in range(d):
                func_val += linear_interpolation_1d(hatm_main[j], x_grid[j]).reshape(-1, 1)
                for k in range(j + 1, d):
                    func_val += linear_interpolation_2d(hatm_int[j, k], x_grid[[j, k]]).reshape(-1, 1)

        # Case 2: Batch input (x_grid has shape (n, d))
        else:
            for j in range(d):
                func_val += linear_interpolation_1d(hatm_main[j], x_grid[:, j]).reshape(-1, 1)
                for k in range(j + 1, d):
                    func_val += linear_interpolation_2d(hatm_int[j, k], x_grid[:, [j, k]]).reshape(-1, 1)

        # Return prediction by adding back the mean
        return self.Y.mean().reshape(-1, 1) + func_val.reshape(-1, 1)

