import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch 
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps import csfd, planck

from mantis_shrimp import query
from mantis_shrimp import preprocess
from mantis_shrimp import models
from mantis_shrimp import augmentation
from mantis_shrimp import utils
from mantis_shrimp import extract_magnitude

from einops import rearrange

import matplotlib.gridspec as gridspec

SAVEPATH = '/tmp/mantis_shrimp_fits/'

from dustmaps.config import config
config.reset()
current_dir = os.path.dirname(os.path.abspath(__file__))
config['data_dir'] = os.path.join(current_dir,'dustmaps')


#SO: the only option here is clobber=False; that still means the dustmaps have to be placed exactly
#in the correct place, with the correct filename, and have a correct hash to what dustmaps hardcodes internally.
csfd.fetch()
csfdquery = csfd.CSFDQuery()
# planck.fetch()
planckquery = planck.PlanckQuery()

if __name__ == '__main__':
    
    #TODO add basic usage.
    pass

def luptitude(a,alpha=0.1,Q=6.0,minimum=0.0):
    #I = np.mean(a,0) #average values across each pixel,
    #M = minimum + np.sinh(Q)/(alpha*Q),
    """
    Computes the luptitude transformation of an array.

    Parameters:
    a (array-like): The input array to be transformed. Typically, 
                    this will be an image or data array.
    alpha (float, optional): The softening parameter that controls 
                             the strength of the transformation. 
                             Default value is 0.1.
    Q (float, optional): The dynamic range parameter. Default value 
                         is 6.0.
    minimum (float, optional): A minimum value to be subtracted from 
                               the input array `a`. Default value is 
                               0.0.

    Returns:
    array-like: The transformed array, where the luptitude 
                transformation has been applied element-wise.
    """
    aimg = (np.arcsinh(alpha*Q*(a - minimum))/Q)
    return aimg

def pairwise_row_differences(arr):
    """
    Compute the pairwise differences between rows of a NumPy array.

    Parameters:
    arr (np.ndarray): A 2D NumPy array where differences will be computed between rows.

    Returns:
    np.ndarray: A 3D array where output[i, j, :] is the difference between row i and row j.
    """
    if len(arr.shape) != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    # Number of rows
    n = arr.shape[0]
    
    # Compute pairwise differences using broadcasting
    differences = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    
    return differences
    

def get_data(user_index: int, user_ra: float, user_dec: float, SAVEPATH=SAVEPATH):
    """
    Retrieves and processes astronomical data for a given user-specified object.

    This function takes user-specified coordinates (right ascension and 
    declination) and an index for identification, then queries various 
    astronomical databases (GALEX, Pan-STARRS, UnWISE) to retrieve the 
    corresponding data. The queried data is processed, converted to 
    float32 format, and extinction values are calculated.

    Parameters:
    user_index (int): Unique identifier for the object. Can be an integer or 
                      any unique identifier.
    user_ra (float): Right ascension of the object in degrees.
    user_dec (float): Declination of the object in degrees.
    SAVEPATH (str, optional): Directory path where the data will be saved. 
                              This should be defined earlier in the script or 
                              provided as an argument.

    Returns:
    tuple: A tuple containing the following elements:
        - galex_data (np.ndarray): GALEX image data processed and converted 
                                   to float32 format.
        - panstarrs_data (np.ndarray): Pan-STARRS image data processed and 
                                       converted to float32 format.
        - unwise_data (np.ndarray): UnWISE image data processed and converted 
                                    to float32 format.
        - ebvs (np.ndarray): Extinction values from CSFD and Planck queries in 
                             float32 format.
    """

    #wrap these in array.
    names = np.array([user_index,]) #can be strings, integers, whatever. as long as they are unique to each object.
    ra = np.array([user_ra,]) #right ascension
    dec = np.array([user_dec,]) #declination

    #using these coordinates, make a query.
   # query.GetGalex(names,ra,dec,SAVEPATH,clobber=True)
   # query.GetUnwise(names,ra,dec,SAVEPATH,clobber=True)
   # query.GetPanstarrs(names,ra,dec,SAVEPATH,clobber=True)
    query.Get_All_Fits(names,ra,dec,SAVEPATH,clobber=True)

    #using the result of that query, process and load the data.
    filters = ['Galex','g','r','i','z','y','Unwise']
    paths = []

    galex_data = []
    panstarrs_data = []
    unwise_data = []
    for name in names:
        for filter in filters:
            filename = os.path.join(SAVEPATH,f'{name}.{filter}.fits')
            assert os.path.exists(filename)
            paths.append(filename)

        #our 'preprocessing' really just converts fits -> floats
        #we will do more preprocessing later.
        galex_img = preprocess.process_galex(paths[0])
        panstarrs_img = preprocess.process_panstarrs(paths[1:6])
        unwise_img = preprocess.process_unwise(paths[6])

        galex_data.append(galex_img)
        panstarrs_data.append(panstarrs_img)
        unwise_data.append(unwise_img)

    #convert the data to float32
    galex_data = np.array(galex_data).astype(np.float32)
    panstarrs_data = np.array(panstarrs_data).astype(np.float32)
    unwise_data = np.array(unwise_data).astype(np.float32)
    
    #Next we will grab the extinction values.
    ebv_csfds = []
    ebv_plancks = []
    for raMean,decMean in zip(ra,dec):
        coords = SkyCoord(raMean*u.deg, decMean*u.deg, frame='icrs')
        ebv_csfd = csfdquery(coords)
        ebv_planck = planckquery(coords)

        ebv_csfds.append(ebv_csfd)
        ebv_plancks.append(ebv_planck)

    ebvs = np.array([ebv_csfds,ebv_plancks]).T
    ebvs = ebvs.astype(np.float32)

    return (galex_data, panstarrs_data, unwise_data, ebvs)

def get_feature_vector(data: tuple):
    """
    Extracts and processes feature vectors from astronomical data.

    Parameters:
    data (tuple): A tuple containing the following elements:
        - x_galex (np.ndarray): GALEX image data.
        - x_ps (np.ndarray): Pan-STARRS image data.
        - x_unwise (np.ndarray): UnWISE image data.
        - ebvs (np.ndarray): Extinction values from CSFD and Planck queries.

    Returns:
    np.ndarray: A normalized feature vector ready for analysis or model input.
    """
    # Unpack the input data
    x_galex, x_ps, x_unwise, ebvs = data
    
    inner_fluxes = []
    inner_forced_photos = []
    
    # Loop over 9 indices to extract fluxes and forced photos from the images
    for index in range(9):
        flux, forced_photo = extract_magnitude.extract_magnitude(x_galex[0],x_ps[0],x_unwise[0],index)
        inner_fluxes.append(flux)
        inner_forced_photos.append(forced_photo)

    # Convert fluxes to luptitudes (a logarithmic magnitude scale)
    luptitudes = luptitude(np.asarray(inner_fluxes))[:, None]

    # Calculate pairwise differences of luptitudes
    diffs = pairwise_row_differences(luptitudes)

    # Rearrange diffs for proper concatenation later
    diffs = rearrange(diffs,'a b c -> c (a b)')

    # Mask out columns of diffs that are all zeros
    mask = np.eye(9).reshape(-1).astype(bool)
    diffs = diffs[:,~mask]

    # Concatenate diffs, ebvs, and luptitudes to form the feature vector
    feature_vector = np.concatenate([diffs,ebvs, luptitudes.T],axis=1)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mean_path = os.path.join(current_dir, "MODELS_final","calpit_stats","calpit_mean.npy")
    std_path = os.path.join(current_dir, "MODELS_final","calpit_stats","calpit_mean.npy")
    calpit_mean = np.load(mean_path)
    calpit_std = np.load(std_path)

    # Normalize the feature vector
    normalized_feature_vector = ((feature_vector.copy()-calpit_mean)/(calpit_std))/3

    return normalized_feature_vector

def evaluate(user_index: int,user_ra: float, user_dec: float, model, data: tuple, SAVEPATH=SAVEPATH, device='cpu'):

    """
    Evaluates the model for a given user index and coordinates, returning the point estimate of redshift and the probability distribution function (PDF).

    Args:
        user_index (int): The index of the user.
        user_ra (float): The right ascension (RA) coordinate of the user.
        user_dec (float): The declination (DEC) coordinate of the user.
        model: The machine learning model used for prediction.
        data (tuple): A tuple containing the data arrays [GALEX, PS, unWISE, EBV] for evaluation. Each of these should be iterable and contain the necessary input data for the model.
        SAVEPATH (str, optional): The path to save any necessary temporary files or data. Default is the value of SAVEPATH.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        tuple: 
            - point_z (float): The point estimate of the redshift.
            - PDF (numpy.ndarray): The probability distribution function of the redshift.
            - x (torch.Tensor): The concatenated input tensor (first example) used in the model.

    Note:
        Ensure that the data tuple contains the appropriate data arrays and is properly preprocessed before passing to this function.
    """

    
   # data = get_data(user_index, user_ra, user_dec, SAVEPATH)
    

    CLASS_BINS_npy = np.linspace(0,1.6,400).astype(np.float32)
    CLASS_BINS = torch.from_numpy(CLASS_BINS_npy).to(device)

    #### WE ARE READY TO PLACE INTO MODEL!!!
    PDFs = []
    point_zs = []
    with torch.no_grad():
        for x_galex,x_ps,x_unwise,ebv in zip(data[0],data[1],data[2],data[3]):

            #transform over
            x_galex = utils.galex_transform(x_galex,0.05)
            x_ps =  utils.new_transform(x_ps,0.01,8.0)
            x_unwise = utils.new_transform(x_unwise,0.1,6.0)

            #convert to torch, add first batch dimension
            x_galex = torch.from_numpy(x_galex)[None]
            x_ps = torch.from_numpy(x_ps)[None]
            x_unwise = torch.from_numpy(x_unwise)[None]
            ebv = torch.from_numpy(ebv)[None]

            #send to device
            x_galex = x_galex.to(device)
            x_ps = x_ps.to(device)
            x_unwise = x_unwise.to(device)
            ebv = ebv.to(device)

            #remove nans.
            x_ps[torch.isnan(x_ps)] = 0.0

            #augment, needed if you want to use earlyfusion since i need to upsample
            x_galex, x_ps, x_unwise = augmentation.augment_fn(x_galex,x_ps,x_unwise)

            x = torch.concatenate([x_galex, x_ps, x_unwise],1)
            y_hat = model(x,ebv)

            point_z = torch.sum(CLASS_BINS * torch.nn.functional.softmax(y_hat,dim=-1),-1)[0]

            PDFs.append(torch.nn.functional.softmax(y_hat,dim=-1)[0].cpu().numpy()) #[n_classes]
            point_zs.append(point_z.cpu().numpy()) #[1]
        
        #now we are ready to visualize.
        return point_zs[0], PDFs[0], x[0]
        
def visualization(point_z, PDF, x, name, ra, dec, CLASS_BINS_npy):
    """
        Visualizes the input data and the predicted redshift probability distribution function (PDF).
        
        Args:
            point_z (float): The point estimate of the redshift.
            PDF (numpy.ndarray): The probability distribution function of the redshift.
            x (torch.Tensor): The concatenated input tensor used in the model.
            name (str): The name or identifier for the visualization.
            ra (float): The right ascension (RA) coordinate of the object.
            dec (float): The declination (DEC) coordinate of the object.
            CLASS_BINS_npy (numpy.ndarray): The array of redshift bins used for classification.
        
        Returns:
            matplotlib.figure.Figure: The figure object containing the visualization.
        
        Note:
            Ensure that the input tensor `x` consists of the expected data format and dimensions.
            The function creates a figure with two subplots:
            - The left subplot visualizes the input data for various filters.
            - The right subplot visualizes the probability distribution function (PDF) of the redshift.
        
        Visualization Details:
            1. The left side provides context by displaying images from different filters used in the input tensor `x`.
               It includes titles and hides the axes.
            2. The right side displays the PDF and highlights the point prediction with a vertical line.
               It also calculates and displays the 1-sigma confidence interval.

    """

    FILTERNAMES = ['Galex:FUV','Galex:NUV','PS:g','PS:r','PS:i','PS:z','PS:y','WISE:W1','WISE:W2']

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 2.4]) 

    #fig = plt.figure(figsize=(12, 9))
   # gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 2.4]) # Modified width ratio
    

    #Left side: visualize the input for the user to give them context
    ax1 = fig.add_subplot(gs[0,0])
    plt.xticks([])
    plt.yticks([])
    ax1.set_title('z_mine={:.3f} \n ra={}, dec={}'.format(point_z,   
                                                          ra,
                                                          dec))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0,0])
    #visualize the pointing

    axes = []
    for i in range(9):
        ax = fig.add_subplot(gs_sub[i])
        ax.set_title(FILTERNAMES[i])
        axes.append(ax)

    for i, ax in enumerate(axes):    
        if i==0: #switch UV because they are actually stored backwards.
            ax.imshow(x[1].detach().cpu().numpy())
        elif i==1:
            ax.imshow(x[0].detach().cpu().numpy())
        else:
            ax.imshow(x[i].detach().cpu().numpy())
        #plt.imshow(average_over[0,i].detach().cpu().numpy())
        #plt.colorbar()
        ax.set_xticks([])
        ax.set_yticks([])

    #calculate 1 sigma above and 1 sigma below confidence interval:
    #first make sure the PDF is normalized as a PMF.
    PDF_new = PDF.copy() / np.sum(PDF.copy())
    CDF = np.cumsum(PDF_new)
    below =  CLASS_BINS_npy[np.argmin(np.abs(CDF-0.1586))]
    above = CLASS_BINS_npy[np.argmin(np.abs(CDF-0.8413))]

    below = point_z - below
    above = above - point_z

    if below < 0:
        below = 0
    if above < 0:
        above = 0

    #Right side: visualize the PDF 
    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('P(z) = {:.4f} +{:.4f} -{:.4f}'.format(point_z,above,below))
    plt.plot(CLASS_BINS_npy,PDF,'r--',label='P(z)')
    plt.vlines(point_z,np.min(PDF),np.max(PDF),'c',label='Point Prediction')
    plt.legend()

    return fig


