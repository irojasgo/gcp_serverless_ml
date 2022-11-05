#Libraries are imported
import numpy as np
import pandas as pd
import os
import joblib
from google.cloud import storage


## Gloabl model variable
model = None


# Download model file from cloud storage bucket
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = "modeltestirojasgo" #Your bucket name goes here
    PROJECT_ID         = "up-cloud-computing-364502" #Your project ID goes here
    GCS_MODEL_FILE     = "SpotifyModel.joblib" #Your model's name goes here

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "local_model.joblib")


# Main entry point for the cloud function
def spotify_predict(request):

    # Use the global model variable 
    global model

    if not model:

        download_model_file()
        model = joblib.load(open("/tmp/local_model.joblib", 'rb'))
    
    
    # Get the features sent for prediction
    params = request.get_json()

    if (params is not None) and ('features' in params):
        results_dict = {}

        #Define prediction dataframe
        predict_df =pd.DataFrame([params['features']])
        predict_df.columns=['danceability', 'energy','key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','liveness', 'valence']

        # Run a test prediction
        result  = model.predict(predict_df)

        #Assign prediction probability to results dictionary
        results_dict['probability'] = model.predict_proba(predict_df)[0][np.argmax(model.predict_proba(predict_df)[0])]

        #Assing prediction to results dictionary
        if result[0]==0:
            results_dict['tag']='Not-Liked'
        else:
            results_dict['tag']='Not-Liked'
        return results_dict        
    else:
        return "Request was empty. Nothing to predict"