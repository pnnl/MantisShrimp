# Standard library imports
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import os

# Third-party imports
import numpy as np
import pandas as pd
import requests
from astropy.table import Table
from pathlib import Path


###TODO check that these are the same as in my giant sbatch scripts.
def fetch_data(session, url, payload, files):
    """
    Sends a POST request to a specified URL with the provided payload and files using a given session and returns the response text.

    Parameters:
    session (requests.Session): The session to use for making the POST request.
    url (str): The URL to which the POST request is to be sent.
    payload (dict): The data to send in the body of the POST request.
    files (dict): The files to send in the POST request.

    Returns:
    str: The text content of the response.

    Raises:
    requests.exceptions.HTTPError: If the response status code indicates an error.
    """
    # Send POST request with provided payload and files
    r = session.post(url, data=payload, files=files)
    # Raise an error for bad status codes
    r.raise_for_status()
    # Return response text
    return r.text


def is_file_exists(filepath):
    """
    Checks if a file exists at a specified filepath to minimize repetitive I/O operations.

    Parameters:
    filepath (str): The path to the file to check.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    # Check if file exists
    return os.path.exists(filepath)

def fetch_and_save(url, filepath, clobber):
    """
    Fetches content from a specified URL and saves it to a given filepath if the response status code is 200 and either `clobber` is True or the file does not exist.

    Parameters:
    url (str): The URL from which to fetch content.
    filepath (str): The path to the file where the content should be saved.
    clobber (bool): Flag to indicate whether to overwrite the file if it already exists.
    """
     # Fetch content from URL
    response = requests.get(url)
    # Save content to file if status code is 200 and either clobber is True or file does not exist
    if response.status_code == 200 and (clobber or not Path(filepath).exists()):
        with open(filepath, "wb") as f:
            f.write(response.content)

def PS_query_table(tra, tdec, size=170, filters="grizy", format="fits", imagetypes="stack"):
    """
    Queries the PS1 (Pan-STARRS1) catalog for image data based on provided coordinates and other parameters.

    Parameters:
    tra (list): List of right ascension (RA) values.
    tdec (list): List of declination (DEC) values.
    size (int, optional): Size of the cutout image. Default is 170.
    filters (str, optional): Filters to use (e.g., "grizy"). Default is "grizy".
    format (str, optional): The format of the image ("jpg", "png", "fits"). Default is "fits".
    imagetypes (str or list, optional): Types of images to query. Default is "stack".

    Returns:
    astropy.table.Table: Table containing the queried image data and URLs.
   
    Raises:
    ValueError: If format is not one of "jpg", "png", "fits".
    """
    
    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

    # Validate the format input
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")

    # Convert imagetypes to comma-separated string if it is not a string
    if not isinstance(imagetypes, str):
        imagetypes = ",".join(imagetypes)
    
    positions = "\n".join(f"{ra} {dec}" for ra, dec in zip(tra, tdec))
    
    with requests.Session() as session:
        # Keep the connection alive
        session.headers.update({'Connection': 'keep-alive'})

         # Prepare payload for the POST request
        payload = {'filters': filters, 'type': imagetypes}
        files = {'file': ('positions.txt', positions)}
        
        # Fetch data concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            future = executor.submit(fetch_data, session, ps1filename, payload, files)
             # Obtain the response text
            response_text = future.result()
            
        # Read the response into an astropy Table
        tab = Table.read(response_text, format="ascii")
        
        # Generate URLs for the images
        urlbase = f"{fitscut}?size={size}&format={format}"
        tab["url"] = [f"{urlbase}&ra={ra}&dec={dec}&red={filename}"
                      for filename, ra, dec in zip(tab["filename"], tab["ra"], tab["dec"])]
    
    return tab


def GetGalex(names:np.ndarray, tra:np.ndarray, tdec:np.ndarray, savepath:str, pix:int=32, layer:str ='galex', clobber=False) :
    """
    Downloads GALEX image cutouts for given astronomical objects and saves them as FITS files.

    Parameters:
    names (np.ndarray): Array of object names.
    tra (np.ndarray): Array of right ascension (RA) values.
    tdec (np.ndarray): Array of declination (DEC) values.
    savepath (str): Directory path where the FITS files will be saved.
    pix (int, optional): Size of the image cutout in pixels. Default is 32.
    layer (str, optional): Image layer to use from the server. Default is 'galex'.
    clobber (bool, optional): Flag to indicate whether to overwrite existing files. Default is False.
    """
    
    for index,ra,dec in zip(names,tra,tdec):
        # Construct the file name
        fname2 = str(index)+".Galex"+'.fits'
        # Create the full file path
        filepath = os.path.join(savepath,fname2)
        # Skip downloading if file exists and clobber is False
        if os.path.exists(filepath) and not(clobber):
            #no clobber
            continue
        # Construct the URL for the GALEX image cutout
        url = f'https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&size={pix}&pixscale=1.5&layer={layer}'
        response = requests.get(url,)
        #if the server doesn't respond, try again
        while response.status_code != 200:
            response = requests.get(url)
        #response.raise_for_status()

         # Save the response content as a FITS file
        open(filepath,"wb").write(response.content)

def GetUnwise(names:np.ndarray, tra:np.ndarray, tdec:np.ndarray, savepath:str, pix:int=32, layer:str ='unwise-neo7', clobber=False):
    """
    Downloads unWISE image cutouts for given astronomical objects and saves them as FITS files.

    Parameters:
    names (np.ndarray): Array of object names.
    tra (np.ndarray): Array of right ascension (RA) values.
    tdec (np.ndarray): Array of declination (DEC) values.
    savepath (str): Directory path where the FITS files will be saved.
    pix (int, optional): Size of the image cutout in pixels. Default is 32.
    layer (str, optional): Image layer to use from the server. Default is 'unwise-neo7'.
    clobber (bool, optional): Flag to indicate whether to overwrite existing files. Default is False.
    """
    for index,ra,dec in zip(names,tra,tdec):
        # Construct the file name
        fname2 = str(index)+'.Unwise'+'.fits'
        # Create the full file path
        filepath = os.path.join(savepath,fname2)
        # Skip downloading if file exists and clobber is False
        if os.path.exists(filepath) and not(clobber):
            #no clobber
            continue
        # Construct the URL for the unWISE image cutout
        url = f'https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&size={pix}&pixscale=2.75&layer={layer}'
        response = requests.get(url,)
        #if the server doesn't respond, try again
        while response.status_code != 200:
            response = requests.get(url)
        #response.raise_for_status()
        # Save the response content as a FITS file
        open(filepath,"wb").write(response.content)


def GetPanstarrs(names:np.ndarray, tra:np.ndarray, tdec:np.ndarray, savepath:str, clobber=False):
    """
    Downloads Pan-STARRS (PS1) image cutouts for given astronomical objects and saves them as FITS files.

    Parameters:
    names (np.ndarray): Array of object names.
    tra (np.ndarray): Array of right ascension (RA) values.
    tdec (np.ndarray): Array of declination (DEC) values.
    savepath (str): Directory path where the FITS files will be saved.
    clobber (bool, optional): Flag to indicate whether to overwrite existing files. Default is False.
    """
    # get the PS1 info for those positions
    table = PS_query_table(tra,tdec,filters="grizy")

    #Going to use this to track the match onto my spectroscopic dataset
    table['index'] = names.repeat(5)

    #no clobber-- if the file already exists then we can ignore it!
    mask = []
    for i in range(len(table)):
        filter = table[i]['filter']
        index = table[i]['index']
        
        fname2 = str(index)+'.'+filter+'.fits'
        if os.path.exists(os.path.join(savepath,fname2)) and not(clobber):
            mask.append(False)
        else:
            mask.append(True)
    table = table[mask]

    # if you are extracting images that are close together on the sky,
    # sorting by skycell and filter will improve the performance because it takes
    # advantage of file system caching on the server

    table.sort(['projcell','subcell','filter'])

    # extract cutout for each position/filter combination
    for i,row in enumerate(table):
        index = row['index']
        fname2 = str(index)+'.'+filter+'.fits'
        filepath = os.path.join(savepath,fname2)
        
        ra = row['ra']
        dec = row['dec']
        projcell = row['projcell']
        subcell = row['subcell']
        filter = row['filter']
        index = row['index']

        # create a name for the image -- could also include the projection cell or other info
        fname = "t{:08.4f}{:+07.4f}.{}.fits".format(ra,dec,filter)

        url = row["url"]
        
        response = requests.get(url,)
        #if the server doesn't respond, try again
        counter=0
        while response.status_code != 200:
            response = requests.get(url)
            time.sleep(1)
            counter+=1
            
            if counter==10:
                response.raise_for_status()
        # Save the response content as a FITS file
        open(filepath,"wb").write(response.content)

def Get_All_Fits(names: np.ndarray, tra: np.ndarray, tdec: np.ndarray, savepath: str, pix: int = 32, clobber: bool = False):
    """
    Downloads GALEX, unWISE, and Pan-STARRS (PS1) image cutouts for given astronomical objects and 
    saves them as FITS files in the specified directory.

    Parameters:
    names (np.ndarray): Array of object names.
    tra (np.ndarray): Array of right ascension (RA) values.
    tdec (np.ndarray): Array of declination (DEC) values.
    savepath (str): Directory path where the FITS files will be saved.
    pix (int, optional): Size of the image cutout in pixels. Default is 32.
    clobber (bool, optional): Flag to indicate whether to overwrite existing files. Default is False.
    """
    ps_cached_tables = {}
    
    for index, ra, dec in zip(names, tra, tdec):
        # Define file paths for GALEX and unWISE
        fname_galex = f"{index}.Galex.fits"
        filepath_galex = os.path.join(savepath, fname_galex)
        
        fname_unwise = f"{index}.Unwise.fits"
        filepath_unwise = os.path.join(savepath, fname_unwise)

        # Query the PS1 table for the specified positions
        table = PS_query_table(tra, tdec, filters="grizy")
        table['index'] = names.repeat(5)

        # Mask to filter out existing files if clobber is False
        mask = []
        for i in range(len(table)):
            filter = table[i]['filter']
            index = table[i]['index']
            
            fname2 = str(index)+'.'+filter+'.fits'
            if os.path.exists(os.path.join(savepath,fname2)) and not(clobber):
                mask.append(False)
            else:
                mask.append(True)
                
        table = table[mask]
        
        # if you are extracting images that are close together on the sky,
        # sorting by skycell and filter will improve the performance because it takes
        # advantage of file system caching on the server

        # Sorting by skycell and filter to optimize performance
        table.sort(['projcell','subcell','filter'])

        # Generate URLs and file paths for PS1 images
        ps_urls = [row["url"] for row in table]
        ps_filenames = [os.path.join(savepath, f"{index}.{row['filter']}.fits") for row in table]

        # Construct URLs for GALEX and unWISE cutouts
        url_galex = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&size={pix}&pixscale=1.5&layer=galex"
        url_unwise = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&size={pix}&pixscale=1.5&layer=unwise-neo7"

        # Combine URLs and file paths
        urls = [(url_galex, filepath_galex), (url_unwise, filepath_unwise)]
        urls.extend(list(zip(ps_urls, ps_filenames)))

        # Download the images concurrently using a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_and_save, url, filepath, clobber) for url, filepath in urls]
            for future in futures:
                future.result()  # This will raise exceptions if any occurred during fetch_and_save
