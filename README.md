# Lyrics Classification and Generation Project

This project focuses on the preprocessing, classification, and generation of song lyrics using machine learning techniques. It includes steps for data cleaning, tokenization, training a classification model, and training a generation model to create new lyrics.

## Datasets

The project utilizes several datasets, each containing a different number of songs per artist:
- `best_5songs_perartist.csv`
- `best_10songs_perartist.csv`
- `best_20songs_perartist.csv`
- `best_50songs_perartist.csv`
- `best_100songs_perartist.csv`
- `best_200songs_perartist.csv`

## Setup Instructions

1. **Clone the Repository:**
   ```sh
   git clone 
   cd lyrics-analysis-generation

2. **Create and Activate Virtual Environment:**

    ```sh
    python3 -m venv venv
    ource venv/bin/activate  
    # On Windows: venv\Scripts\activate

## Data Preprocessing

### Tokenization
Tokenize the lyrics using the provided Token_vector.ipynb notebook to prepare the data for further processing.

### Text Cleaning
Clean the lyrics data using the text_cleanning.ipynb notebook to remove unnecessary characters and ensure uniform formatting.

### Classification Model

Train a model to classify song genres based on their lyrics using the classifier.ipynb notebook. This involves feature extraction, model training, and evaluation.

### Lyrics Generation Model

Train a model to generate new song lyrics in the style of a specific artist using the generator.ipynb notebook or the generator.py script. The model is trained to predict the next word in a sequence given the previous words.
