[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10587778&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## DESCRIPTION
Text generation is hot news right now!

For this assignment, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts that do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

## METHODS
This project consists of two scripts. One that trains the classifier (model.py) and one that generates an output based on a prompt (generate_text.py). Some in-class functions are included in the final script. <br >

**Training the model**
This project begins with loading and cleaning the data. First, the headlines from all articles are extracted and made lowercase to create the corpus. The headlines are then ran through the functions ```tokenizing()``` and ```padded_sequences()``` to prepare them for the model training. The model is generated in the ```create_model()``` function and it is trained and saved in the ```out``` folder.

**Generating text from a user-generated prompt**
The ```generating_text.py``` script loads the trained model from the ```out``` folder and executes a function, which predicts an output based on a prompt. Based on the prompt the model predicts what the next word most likely would be, generating a sentence of x number of words. When running the script, the user must input a word to generate new text and the number of words to follow. 

## HOW TO INSTALL AND RUN THE PROJECT
**Get the data:**<br >

The data is NOT included in the repository of this project and needs to be loaded separately. The script takes this into account. However, the script expects an unzipped folder. 
- You can find a link to the data here. You may need to unzip the data first, which is not included in the script. <br >
Make sure that the data is loaded *outside* of this repository in a name of your choice, which you will need to provide when running the scripts.

**Installation:**
1. First you need to clone this project repository 
2. Navigate from the root of your directory to ```assignment-3---rnns-for-text-generation-AneliaAB```
3. Run the setup file, which will install all the requirements by writing ```bash setup.sh``` in the terminal

**Run the script:**<br>
4. Navigate to the folder src by writing cd ```src``` in the terminal, assuming your current directory is ```assignment-3---rnns-for-text-generation-AneliaAB```

You can choose to run both scripts in this order: first ```model.py```, second ```generating_text.py```. However, note that when running the script ```generaiting_text.py``` both model.py and generating_text.py will be executed automatically, so there is no need to manually run both scripts in the terminal.

5.	Run the script by writing ```python generating_text.py``` in the terminal <br >
- Firstly, the user will be asked to provide a folder name of where the data is located. Note that the folder needs to be loaded outside of the repository and include ONLY .cvs files (13 total). 
- Secondly the user will need to provide a word which will be used to generate a sentence.
- Thirdly the user will need to provide a number (of numerical value) of words to be generated. 

## DISCUSSION OF RESULTS
The results are saved in the ```out``` folder, in the form of the trained model. When loaded in the ```generate_text.py``` script, the model is able to generate sentences based on user input. The user is asked to both provide a word and a word-count, which gives flexibility and the user can play around with different word and sentence lengths.
