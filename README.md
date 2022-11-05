# gcp_serverless_ml
Project 2: Serverless Machine Learning Model Deployment | Google Cloud
- Trained a machine learning model and pipelined all preprocessing steps
- Encapsulated the model into a binary file
- Deployed the model as an endpoint using GCP's cloud functions
- Tested the model submitting requests from different endpoints
NOTE: For this to work, the model generated by the classifier script should be uploaded to a bucket in cloud storage. funcion_code.py contains the cloud function definition per-se. The cloud function needs to be triggered via an HTTP call (depending on security constraints, allowing annonymous requests and not requiring HTTPS might be a good idea for testing)
