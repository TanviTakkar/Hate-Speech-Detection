from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from pydantic import BaseModel
from bs4 import BeautifulSoup
import re
from symspellpy import SymSpell
import pkg_resources
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import import_ipynb
from hate_speech_project import HateSpeechAnalysis

hatespeech=HateSpeechAnalysis(use_path=False)
model_path = 'best-fine-tuned-distilbert-hate-speech'

tokenizer_path = 'best-fine-tuned-distilbert-hate-speech'

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
term_index = 0  # column of the term names in the dictionary text file
count_index = 1  # column of the term frequencies in the dictionary text file

if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
    print("Dictionary file not found")
# Function to remove HTML tags
def remove_html_tags(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# Function to remove special characters and digits
def remove_special_chars_and_digits(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

# Function to correct spellings
def correct_spellings(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    else:
        return text


# Define label mapping
label_map = {0: 'non-hateful', 1: 'hateful'}
model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


hateSpeech = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Initialize FastAPI app
app = FastAPI()

# Define input data model
class TextData(BaseModel):
    text: str

# API endpoint for prediction
@app.post("/classifier/")
def classifier(data: TextData):
    try:
        '''
        sentence= pd.DataFrame({'Content':[str(data.text)]})

        sentence['Content'] = sentence['Content'].apply(remove_html_tags)
        sentence['Content'] = sentence['Content'].apply(remove_special_chars_and_digits)
        sentence['Content'] = sentence['Content'].apply(correct_spellings)
        # Get prediction from the model
        classification = hateSpeech(sentence['Content'].to_list())
        print(classification)
        
        # Map the predicted label to human-readable label
        for result in classification:
            label_id = int(result['label'].split('_')[-1])  # Extract the label ID
            human_readable_label = label_map[label_id]
         '''
        sentence=hatespeech.preprocess_sentence(data.text)
        pre_processed_sentence=hatespeech.preprocess_test_sentence(text=sentence)
        topic=hatespeech.extract_sentence_topic(sentence=pre_processed_sentence)
        sentence = sentence + " " + topic
        human_readable_label,result= hatespeech.test_model(sentence=sentence, model=model, tokenizer=tokenizer)
        
        return {"text": str(data.text), "Label": str(human_readable_label), "score":float(result)}
        
        
    except Exception as e:
        return {"error": str(e.with_traceback)}
# Run the FastAPI server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
