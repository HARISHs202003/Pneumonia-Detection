# Deep Learning approach to Pneumonia Detection and Classification from Chest X-ray 

# A Flask **pneumonia detection** web application 


## Setting up the Web-App Locally 

1. Extract the downloaded project folder.

2. Follow the video and Install the TensorFlow and CUDA toolkit 
	https://youtu.be/b9e3J-NJ8TY
	
3. Open the Anaconda prompt in the project folder

4. Switch to tf environment using the following command
	>>> conda activate tf
	
5. In tf environment, Install Requirements using the command
	>>> pip install -r requirements.txt
	
## Run the web app using the trained model

1. Switch to tf environment using the following command
	>>> conda activate tf
	
2. Run the following command to launch Flask Webapp
	>>> python app.py
	
3. The app is running at 
	http://127.0.0.1:5000

	

## To train the model locally

1. Switch to tf environment using the following command
	>>> conda activate tf

2. Run the following command to start training model
	>>> python Pneumonia_Detection.py
	
	Once the training is completed the trained model will be saved in the models directory
	The trained model can be used to predict the label for new images
	
	To run the trained model use the step 1, 2, and 3 of ## Run the Web-App using the trained model

#### Download Dataset

Dataset Name: Chest X-Ray Images (Pneumonia)

Dataset Link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

1. Once the dataset is downloaded, you will get an archive folder
2. Extract the archive folder
3. Copy and paste the three folders train, test, and val into the dataset folder of the project directory.