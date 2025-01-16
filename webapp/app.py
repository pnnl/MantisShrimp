#TODO: alert user and refuse to operate if dec < -30
#TODO: webapp should get rid of the name, since it will just be one at a time
#TODO: API should allow sending multiple requests at once

from dustmaps.config import config
config.reset()
config['data_dir'] = './../data/dustmaps/'


from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import os
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec
import numpy as np
import torch 
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps import csfd, planck
import io
import base64
import time
import tarfile

#from mantisshrimp import query
from mantis_shrimp import preprocess
from mantis_shrimp import models
from mantis_shrimp import augmentation
from mantis_shrimp import utils
from mantis_shrimp import pipeline

from calpit import CalPit
from calpit.nn.umnn import MonotonicNN 
from calpit.utils import normalize

from werkzeug.utils import secure_filename

import timm

import uuid

import json

app = Flask(__name__)
# Set up the Limiter
limiter = Limiter(get_remote_address,
    app=app,
    default_limits=["10 per second",]
)


#We need to create a file to save some data for the user.
SAVEPATH = '/tmp/mantis_shrimp_fits/'
if not(os.path.exists(SAVEPATH)):
    os.mkdir(SAVEPATH)
device = 'cpu'

######################################
#instantiate model
model, CLASS_BINS_npy = models.trained_early(device)

######################################
# DustMap definition
######################################
#csfd.fetch()
#csfdquery = csfd.CSFDQuery()
#planck.fetch()
#planckquery = planck.PlanckQuery()

def is_number_with_decimal(s):
    """
    Checks if a given string can be converted to a floating-point number (i.e., contains a decimal).

    Parameters:
    s (str): The string to check.

    Returns:
    bool: True if the string can be converted to a floating-point number, False otherwise.
    """

    try:
        float(s)
        return True
    except ValueError:
        return False

@app.route('/', methods=['GET','POST'])
def index():
    """
    Handles the main index route for the web application. Renders the main form and processes user input
    to fetch astronomical data and visualize photometric redshift predictions.

    Supports both GET and POST methods.

    Returns:
    flask.Response: The HTML response for rendering the index template.
    """
    if request.method == 'POST':

        # Validate RA and DEC coordinates
            
        user_index = str(request.form['NAME'])

        if is_number_with_decimal(request.form['RA']) and is_number_with_decimal(request.form['DEC']):
            user_ra = float(request.form['RA'])
            user_dec = float(request.form['DEC'])
        else:
            try:
                user_ra = float(SkyCoord(request.form['RA'], request.form['DEC']).ra.degree)
                user_dec = float(SkyCoord(request.form['RA'], request.form['DEC']).dec.degree)
                
            except ValueError as e:
                error = "INVALID COORDINATES PROVIDED"
                return render_template('index.html', plot_url=None, download_url = None, error = error)
        if user_dec < -30:
            error = "DECLINATION TOO LOW, < -30 NOT ALLOWED"
            return render_template('index.html', plot_url=None, download_url = None, error = error)
        t1 = time.time()
        data = pipeline.get_data(0, user_ra, user_dec, SAVEPATH)
        photo_z, PDF, x = pipeline.evaluate(0, user_ra, user_dec, model=model, data = data, SAVEPATH=SAVEPATH, device=device,)

        if request.form.get('calpit'):

            monotonicnn_model = MonotonicNN(84, [1024,1024,1024,1024], sigmoid=True)
            monotonicnn_model.to(device)
            calpit_model = CalPit(model=monotonicnn_model)
    
            normalized_feature_vector = pipeline.get_feature_vector(data)
            
            #this relative path is okay since this will also exist relative to this file in the cloned repo
            ckpt = torch.load('../mantis_shrimp/MODELS_final/calpit_checkpoint.pt',map_location=torch.device(device),weights_only=True)
    
            calpit_model.model.load_state_dict(ckpt)

            calpit_model.model.train(False)
    
            y_grid = np.linspace(0,1.6,400)
    
            PDF = PDF[None, :]
    
            cde_test = normalize((PDF.copy()+1e-2)/np.sum(PDF.copy()+1e-2, axis = 1)[:,None], np.linspace(0,1.6,400))
    
            new_cde = calpit_model.transform(normalized_feature_vector, cde_test, y_grid, batch_size=16)
    
            new_cde = (new_cde.copy())/np.sum(new_cde.copy(), axis = 1)[:, None]
            
            #update photoz based off the new cde
            photo_z = np.sum(new_cde*CLASS_BINS_npy)

            # Normalize if requested
            if request.form.get('normalize'):

                new_cde = normalize(new_cde, np.linspace(0,1.6,400))

                normalized = True
                
            else:

                normalized = False
                
            fig = pipeline.visualization(photo_z, new_cde.squeeze(), x, 0, user_ra, user_dec, CLASS_BINS_npy)

            usecalpit = True


        else:

            if request.form.get('normalize'):

                new_cde = normalize(PDF, np.linspace(0,1.6,400))

                normalized = True
                
            else:
                
                new_cde = PDF
                
                normalized = False

            fig = pipeline.visualization(photo_z, new_cde, x, 0, user_ra, user_dec, CLASS_BINS_npy)

            usecalpit = False
        
        #store figure into byte object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Prepare data for saving and download
        data = {
            'RA': user_ra,
            'DEC': user_dec,
            'photo_z': photo_z.tolist(),
            'CDE': new_cde.tolist(),
            'CalPit': usecalpit,
            'PDF': normalized
        }

        if not user_index == "":
            
            unique_filename = secure_filename(user_index + '.json')
        else:
            # Generate a unique filename for the download
            unique_filename = 'RA_{:.5f}__DEC_{:.5f}.json'.format(user_ra,user_dec)
        
        tmp_filepath = os.path.join('/tmp', unique_filename)

        # Save data to a JSON file
        with open(tmp_filepath, 'w') as json_file:
            json.dump(data, json_file)

        download_url = url_for('download_file', filename=unique_filename)

        tarfile_name = 'RA_{:.5f}__DEC_{:.5f}.tar'.format(user_ra,user_dec)
        
        fits_url = url_for('download_fits', tarfile_name = tarfile_name)

        return render_template('index.html', plot_url=plot_url, download_url = download_url, fits_url = fits_url, error = None)
    return render_template('index.html',plot_url=None, download_url = None, error = None)

@app.route('/download/<filename>')
def download_file(filename):
    """
    Route handler for downloading a specified file from the server.

    Parameters:
    filename (str): The name of the file to be downloaded.

    Returns:
    flask.Response: Response containing the file to be downloaded.
    """
    # Construct the full path to the file in the /tmp directory
    tmp_filepath = os.path.join('/tmp', filename)
    # Send the file as an attachment
    return send_file(tmp_filepath, as_attachment=True)

@app.route('/fits/<tarfile_name>')
def download_fits(tarfile_name):
    """
    Route handler for downloading a tarball of FITS files.

    Parameters:
    tarfile_name (str): The name of the tar file to be created and downloaded.

    Returns:
    flask.Response: Response containing the tarball of FITS files to be downloaded.
    """
    filters = ['Galex', 'g', 'r', 'i', 'z', 'y', 'Unwise']
    filepaths = []
    # Collect all file paths for the specified filters
    for filter in filters:
        filename = os.path.join(SAVEPATH, f'0.{filter}.fits')
        assert os.path.exists(filename), f"File does not exist: {filename}"
        filepaths.append(filename)

    # Create the tarfile path in the /tmp directory
    tarfile_path = os.path.join('/tmp', tarfile_name)
    with tarfile.open(tarfile_path, "w:gz") as tar:
        for filepath in filepaths:
            try:
                # Add the file to the tar file
                tar.add(filepath, arcname=os.path.basename(filepath))
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"An error occurred while adding {filepath}: {e}")
    # Send the tar file as an attachment
    return send_file(tarfile_path, as_attachment=True)
    
@app.route('/about')
def about():
    """
    Route handler for rendering the About page.

    Returns:
    flask.Response: Response containing the rendered 'about' HTML template.
    """
    return render_template('about.html')
    
@app.route('/predict', methods=['POST'])
@limiter.limit("10 per second") #LIMIT
def predict():
    data = request.get_json(force=True)
    
    user_index = int(data['NAME'])
    if is_number_with_decimal(data['RA']) and is_number_with_decimal(data['DEC']):
            
            user_ra = float(data['RA'])
            user_dec = float(data['DEC'])
    
    else:
        
        user_ra = float(SkyCoord(data['RA'],data['DEC']).ra.degree)
        user_dec = float(SkyCoord(data['RA'],data['DEC']).dec.degree)
    
    data = pipeline.get_data(0, user_ra, user_dec, SAVEPATH)
    photo_z, PDF, x = pipeline.evaluate(0, user_ra, user_dec, model=model, data = data, SAVEPATH=SAVEPATH, device=device,)

    #monotonicnn_model = MonotonicNN(84, [1024,1024,1024,1024], sigmoid=True)
    #monotonicnn_model.to(device)
    #calpit_model = CalPit(model=monotonicnn_model)

    #normalized_feature_vector = pipeline.get_feature_vector(data)
            
    #this relative path is okay since this will also exist relative to this file in the cloned repo
    #ckpt = torch.load('../mantis_shrimp/MODELS_final/calpit_checkpoint.pt',map_location=torch.device(device),weights_only=True)

    #calpit_model.model.load_state_dict(ckpt)

    #calpit_model.model.train(False)

    y_grid = np.linspace(0,1.6,400)
    PDF = PDF[None, :]
    #cde_test = normalize((PDF.copy()+1e-2)/np.sum(PDF.copy()+1e-2, axis = 1)[:,None], y_grid)
    #new_cde = calpit_model.transform(normalized_feature_vector, cde_test, y_grid, batch_size=16)
    #new_cde = (new_cde.copy())/np.sum(new_cde.copy(), axis = 1)[:, None]
    #new_cde = normalize(PDF, np.linspace(0,1.6,400))


    payload = {
        'RA': user_ra,
        'DEC': user_dec,
        'photo_z': photo_z.tolist(),
        'CDE': PDF.tolist(),
        'CalPit': False,
        'PDF': False}
    return jsonify(payload)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded", retry_after=e.description), 429

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000
    
    app.run(host='0.0.0.0', port=port, debug=False)
