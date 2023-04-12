[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10587778&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## Project Description by Ross
Text generation is hot news right now!

For this assignment, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts that do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

## Data
This project uses headlines from the New York Times articles instead of comments due to problems loading the comments data. Therefore the data used in this project is different from the data described above. You can find a link to the New York Times articles here: https://www.kaggle.com/datasets/aashita/nyt-comments (same link as above).

## How to Install and Run the Project
Installation:
1. First you need to clone the repository and load the article data into a separate folder, fx. under the name **news_data**
2. Navigate from the root of your repository to **assignment-3---rnns-for-text-generation-AneliaAB**
 ```cd assignment-3---rnns-for-text-generation-AneliaAB```
3. Run the setup file, which will install all the requirements
 ```bash setup.sh```

Now you are ready to run the scripts:
1. Run the generating_text script
  ```python generating_text.py```
2. Answer the prompts in the terminal
- write the name of the folder that you would like to use (ex. news_data)
- Write a word to generate new text
- How many words do you wish to generate?

Tip! Example answers to the prompts:
- news_data 
- danish 
- 4

If you have trouble loading the data:
The script takes the name of the data folder (via a prompt in the terminal), which the user loads themselves in a separate folder, and puts it into this filepath: ```filepath = f"../../{gather_folder()}/"```. Be aware that you will need to change the filepath in the **model.py** script if the filepath to your data folder is different OR write the path to your folder inside the prompt instead of just the name of your folder. 

## Repositery description
```out```
- the saved model is located in the out folder. this model is created in the model.py script and loaded and used in the **generating_text.py** script.

```src```
two python scripts
- **model.py** loads the data and creates the model
- **generating_text.py** generates text from a user-suggested prompt

```README.md```
Description of the assignment, instructions on how to install and run the project, and challenges. 

```requirements.txt```
Libraries needed for this project

```setup.sh```
created enviorment and installs the libraries in the requirements.txt file

## Challenges 
One of the challenges I faced with this assignment was writing the python script as clean and compact as possible. What I'd like to improve in this project is the **generating_text.py** script by splitting it into two scripts - one loading the data and tokenizing; and another script creating and saving the trained model. I had trouble getting the two scripts to communicate with each other, that's why I decided to combine everything into one python script, which makes it confusing and difficult to follow.