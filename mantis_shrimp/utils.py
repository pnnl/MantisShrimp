import numpy as np
from scipy.spatial import KDTree
import torch

def get_photopath(index:int):
    index = str(index).zfill(7)
    a = index[0]
    b = index[1]
    c = index[2]
    d = index[3]
    
    path = f'/rcfs/projects/mantis_shrimp/mantis_shrimp/data/photometry/{a}/{b}/{c}/{d}/{index}.npy'
    return path

#def central_crop(img):

def galex_transform(a,alpha=0.05):
    aimg = a/alpha
    aimg[aimg<0] = 0
    aimg[aimg>1.0] = 1.0
    return aimg

def my_transform(img,a=0.2):
    img = np.arcsinh(img/a)/np.arcsinh(1/a)
    #img[img<-1] = -1
    #img[img>1] = 1
    
    return img

def new_transform(a,alpha=0.1,Q=6.0,minimum=0.0):
    #I = np.mean(a,0) #average values across each pixel
    #M = minimum + np.sinh(Q)/(alpha*Q)
    aimg = (np.arcsinh(alpha*Q*(a - minimum))/Q)
    aimg[aimg<minimum] = minimum
    #aimg[aimg>1.0] = 1.0
    return aimg

def reverse_new_transform(aimg,alpha=0.1,Q=6.0,minimum=0.0):
    a = np.sinh(aimg*Q)/alpha/Q - minimum
    return a

def cde_loss(cde_estimates, z_grid, z_test):
    """
    Calculates conditional density estimation loss on holdout data

    @param cde_estimates: a torch tensor where each row is a density
    estimate on z_grid
    @param z_grid: a torch tensor of the grid points at which cde_estimates is evaluated
    @param z_test: a torch tensor of the true z values corresponding to the rows of cde_estimates

    @returns The CDE loss (up to a constant) for the CDE estimator on
    the holdout data and the SE error

    From Dalmasso et al, 2019.
    URL: https://github.com/lee-group-cmu/cdetools/blob/master/python/src/cdetools/cde_loss.py
    """

    if len(z_test.shape) == 1:
        z_test = z_test.reshape(-1, 1)
    if len(z_grid.shape) == 1:
        z_grid = z_grid.reshape(-1, 1)

    n_obs, n_grid = cde_estimates.shape
    n_samples, feats_samples = z_test.shape
    n_grid_points, feats_grid = z_grid.shape

    if n_obs != n_samples:
        raise ValueError("Number of samples in CDEs should be the same as in z_test."
                         "Currently %s and %s." % (n_obs, n_samples))
    if n_grid != n_grid_points:
        raise ValueError("Number of grid points in CDEs should be the same as in z_grid."
                         "Currently %s and %s." % (n_grid, n_grid_points))

    if feats_samples != feats_grid:
        raise ValueError("Dimensionality of test points and grid points need to coincise."
                         "Currently %s and %s." % (feats_samples, feats_grid))

    z_min = np.min(z_grid, axis=0)
    z_max = np.max(z_grid, axis=0)
    z_delta = (z_max - z_min)/(n_grid_points-1)

    integrals = z_delta * np.sum(cde_estimates**2, axis=1)

    #indices of the true value of occurence
    nn_ids = np.argmin(abs(z_grid.T-z_test),1).flatten()
    likeli = cde_estimates[(tuple(np.arange(n_samples)), tuple(nn_ids))]

    losses = integrals - 2 * likeli
    loss = np.mean(losses)
    se_error = np.std(losses, axis=0) / (n_obs ** 0.5)

    return loss, se_error

def cde_loss_torch(cde_estimates, z_grid, z_test):
    """
    Calculates conditional density estimation loss on holdout data

    @param cde_estimates: a torch tensor where each row is a density
    estimate on z_grid
    @param z_grid: a torch tensor of the grid points at which cde_estimates is evaluated
    @param z_test: a torch tensor of the true z values corresponding to the rows of cde_estimates

    @returns The CDE loss (up to a constant) for the CDE estimator on
    the holdout data and the SE error

    From Dalmasso et al, 2019.
    URL: https://github.com/lee-group-cmu/cdetools/blob/master/python/src/cdetools/cde_loss.py
    """

    if len(z_test.shape) == 1:
        z_test = z_test.reshape(-1, 1)
    if len(z_grid.shape) == 1:
        z_grid = z_grid.reshape(-1, 1)

    n_obs, n_grid = cde_estimates.shape
    n_samples, feats_samples = z_test.shape
    n_grid_points, feats_grid = z_grid.shape

    if n_obs != n_samples:
        raise ValueError("Number of samples in CDEs should be the same as in z_test."
                         "Currently %s and %s." % (n_obs, n_samples))
    if n_grid != n_grid_points:
        raise ValueError("Number of grid points in CDEs should be the same as in z_grid."
                         "Currently %s and %s." % (n_grid, n_grid_points))

    if feats_samples != feats_grid:
        raise ValueError("Dimensionality of test points and grid points need to coincise."
                         "Currently %s and %s." % (feats_samples, feats_grid))

    z_min = torch.min(z_grid, axis=0)[0]
    z_max = torch.max(z_grid, axis=0)[0]
    z_delta = (z_max - z_min)/(n_grid_points-1)

    integrals = z_delta * torch.sum(cde_estimates**2, axis=1)

    #indices of the true value of occurence
    nn_ids = torch.argmin(abs(z_grid-z_test),1,keepdim=True).squeeze(1)
    likeli = cde_estimates[(tuple(torch.arange(n_samples)), tuple(nn_ids))]

    losses = integrals - 2 * likeli
    loss = torch.mean(losses)
    se_error = torch.std(losses, axis=0) / (n_obs ** 0.5)

    return loss, se_error

def Wasserstein_p(y_hat,y_true,p: int, CLASS_BINS):
    #this is just the wasserstein metric with the
    #true value probability distribution being the dirac delta
    PDF = torch.nn.functional.softmax(y_hat,dim=-1)
    CDF = torch.cumsum(PDF,1)
    indicator = torch.ones_like(CDF)
    #index where CDF indicator is False.
    Q = torch.argmin(abs(CLASS_BINS[None,:]-y_true[:,None]),1,keepdim=True).squeeze(1)
    for i in range(len(Q)):
        indicator[i,0:Q[i]] = 0 #indicator is now a step-function CDF
    
    #remove batch size scaling, and remove Num_Classes scaling
    return torch.sum(torch.pow(torch.pow(torch.abs(CDF - indicator),p),1/p))/len(CDF)/PDF.shape[1]

def Wasserstein1(y_hat,y_true):
    #this is just the wasserstein metric with the
    #true value probability distribution being the dirac delta
    PDF = torch.nn.functional.softmax(y_hat,dim=-1)
    CDF = torch.cumsum(PDF,1)
    indicator = torch.ones_like(CDF)
    #index where CDF indicator is False.
    Q = torch.argmin(abs(CLASS_BINS[None,:]-y_true[:,None]),1,keepdim=True).squeeze(1)
    for i in range(len(Q)):
        indicator[i,0:Q[i]] = 0 #indicator is now a step-function CDF
    return torch.sum(torch.abs(CDF - indicator))/len(CDF) #remove batch size scaling

def Wasserstein_p_normal(y_hat,y_true,p: int, sigma: float=0.05):
    #this is just the wasserstein metric with the
    #true value probability distribution being the normal distribution
    #with mu given by the 1-dimensional vector y_true and sigma
    #being given as a float. TODO, accept sigma as a tensor.
    PDF = torch.nn.functional.softmax(y_hat,dim=-1)
    CDF = torch.cumsum(PDF,1)

    #CDF of a normal distribution
    CDF_prime = 0.5 * (1 + (torch.special.erf((CLASS_BINS[None,:] - y_true[:,None])/(sigma*np.sqrt(2)))))
    
    #remove batch size scaling, and remove Num_Classes scaling
    return torch.sum(torch.pow(torch.pow(torch.abs(CDF - CDF_prime),p),1/p))/len(CDF)/PDF.shape[1], CDF_prime

def CDF_normal(x,mu,sigma):
    return 0.5 * (1 + (torch.special.erf((x[None,:] - mu[:,None])/(sigma[:,None]*np.sqrt(2)))))

def PDF_normal(x,mu,sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * torch.exp(-0.5 * ((x[None,:]-mu[:,None])/sigma)**2) * dz

def PDF_truncnormal(x,mu,sigma,a=0.0,b=1.0):
    rho = PDF_normal(x,mu,sigma)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Phi_a = 0.5 * (1 + (torch.special.erf((alpha/np.sqrt(2))))) #[y_true]
    Phi_b = 0.5 * (1 + (torch.special.erf((beta/np.sqrt(2))))) #[y_true]
    Z = Phi_b - Phi_a
    return rho/(Z)

def Wasserstein_p_truncnormal(y_hat,y_true,p: int, sigma: float=0.05,a:float=0.0,b:float =1.0):
    #this is just the wasserstein metric with the
    #true value probability distribution being the normal distribution
    #with mu given by the 1-dimensional vector y_true and sigma
    #being given as a float. TODO, accept sigma as a tensor.
    PDF = y_hat
    CDF = torch.cumsum(PDF,1)

    #CDF of a normal distribution
    Phi_a = 0.5 * (1 + (torch.special.erf((a - y_true)/(sigma*np.sqrt(2))))) #[y_true]
    Phi_b = 0.5 * (1 + (torch.special.erf((b - y_true)/(sigma*np.sqrt(2))))) #[y_true]

    CDF_prime = (CDF_normal(CLASS_BINS,y_true,sigma) - Phi_a[:,None]) / (Phi_b[:,None] - Phi_a[:,None])
    
    #remove batch size scaling, and remove Num_Classes scaling
    return torch.sum(torch.pow(torch.pow(torch.abs(CDF - CDF_prime),p),1/p))/len(CDF)/PDF.shape[1]

def get_vec_std(y_hat):
    mu = torch.sum(CLASS_BINS[None,:]*y_hat,1)# [batch_size]
    var = torch.sum(y_hat * torch.square(CLASS_BINS[None,:] - mu[:,None]),1)
    return torch.sqrt(var)
