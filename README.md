# Pokédex

## Overview

This application combines comprehensive identification features of a multimodal input app with the quick and broad description handling capabilities of a Retrieval-Augmented Generation (RAG) system. The goal is to create a versatile and efficient Pokémon Identification App that can handle text, image, and combined queries, providing detailed and accurate information about Pokémon.

## Datasets

- [Pokémon Dataset by Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/pokemon)
- [Pokémon Image Dataset by hlrhegemony](https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset/data)

## To-Do List

1. **Data Preparation:**

   - Download Pokémon images and preprocess them.
   - Load and preprocess the general info CSV file.

2. **Model Training:**

   - Train/fine-tune the CLIP model on the Pokémon image dataset.
   - Fine-tune a BERT/GPT model on the general info dataset.
   - Generate and store text and image embeddings.

3. **Database Setup:**

   - Define schema for Pokémon metadata and image URLs in a relational database.
   - Store text and image embeddings in a vector database.

4. **Backend Development:**

   - Create scripts to preprocess data, generate embeddings, and store them in the respective databases.
   - Develop API endpoints for text, image, and combined searches.
   - Integrate embedding search with vector and relational databases.

5. **User Interface:**

   - Create input fields for text descriptions and image uploads.
   - Design results display to show Pokémon images and general info.
   - Connect the UI with backend APIs to handle queries and display results.

6. **Enhanced RAG Integration:**

   - Implement functions to handle broad descriptions and generate responses using RAG.
   - Test the app for various queries and input types to ensure accurate and efficient retrieval of Pokémon data.

7. **Deployment:**
   - Deploy the backend on a suitable cloud platform (e.g., AWS, Heroku).
   - Deploy the frontend and ensure it is accessible to users.

## Key Components

### Data Preparation

1. **Image Dataset**: Contains images and names of Pokémon.
2. **General Info Dataset**: Contains names, types, HP levels, attack levels, defense levels, and detailed descriptions.

### Models

1. **CLIP Model**: For embedding text and image inputs.
2. **BERT/GPT Model**: For embedding and handling broad text descriptions.
3. **Retrieval-Augmented Generation (RAG)**: To enhance broad description handling and provide relevant information.

### Databases

1. **Relational Database**: Stores Pokémon metadata, image URLs, and detailed descriptions.
2. **Vector Database**: Stores embeddings for efficient similarity searches.

### Backend

1. **Data Ingestion**: Processes and stores data and embeddings.
2. **Search API**: Handles text, image, and combined queries to return matching Pokémon data.
3. **RAG Integration**: Enhances search capabilities by handling broad descriptions.

### User Interface

1. **Query Input**: Allows users to input text descriptions or upload images.
2. **Results Display**: Shows the Pokémon image and general information.

## Implementation Steps

### Data Preparation

#### Download and preprocess images:

\`\`\`python
from PIL import Image
import os

def preprocess_image(image_path):
image = Image.open(image_path)
image = image.resize((224, 224)) # Example resize
return image

image_folder = "path_to_images"
processed_images = [preprocess_image(os.path.join(image_folder, img)) for img in os.listdir(image_folder)]
\`\`\`

#### Load and preprocess CSV:

\`\`\`python
import pandas as pd

general_info = pd.read_csv('path_to_csv/pokemon.csv')
general_info = general_info.fillna(0)
\`\`\`

### Model Training

#### CLIP training:

\`\`\`python
import clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("example.jpg")).unsqueeze(0)
text = clip.tokenize(["a blue turtle"]).to(device)

with torch.no_grad():
image_features = model.encode_image(image)
text_features = model.encode_text(text)
\`\`\`

#### BERT/GPT fine-tuning:

\`\`\`python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("a water type Pokémon", return_tensors="pt")
outputs = model(\*\*inputs)
\`\`\`

### Database Setup

#### Relational Database:

\`\`\`sql
CREATE TABLE pokemon (
id SERIAL PRIMARY KEY,
name TEXT,
type TEXT,
hp INT,
attack INT,
defense INT,
description TEXT,
image_url TEXT
);
\`\`\`

#### Vector Database:

\`\`\`python
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch

pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")
index_name = "pokemon-index"
if index_name not in pinecone.list_indexes():
pinecone.create_index(index_name, dimension=384)

index = pinecone.Index(index_name)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(text):
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
embeddings = model(\*\*inputs).last_hidden_state.mean(dim=1)
return embeddings.numpy()

pokemon_data = [
{"name": "Pikachu", "description": "Electric type Pokémon, yellow, small, mouse-like"},
{"name": "Charizard", "description": "Fire/Flying type Pokémon, large, dragon-like"},
]

for pokemon in pokemon_data:
description = pokemon["description"]
embedding = generate_embeddings(description)
index.upsert([(pokemon["name"], embedding)])
\`\`\`

### Backend Development

#### Data Ingestion:

\`\`\`python
for pokemon in dataset:
image_embedding = encode_image_clip(pokemon['image'])
text_embedding = encode_text_clip(pokemon['description'])
store_embeddings_in_vector_db(pokemon['id'], image_embedding, text_embedding)
store_metadata_in_relational_db(pokemon)
\`\`\`

#### API Endpoints:

\`\`\`python
from flask import Flask, request, jsonify
app = Flask(**name**)

@app.route('/search', methods=['POST'])
def search():
query = request.json.get('query')
embeddings = encode_text_clip(query)
closest_match_id = vector_db.search(embeddings)
pokemon_data = relational_db.get_pokemon_by_id(closest_match_id)
return jsonify(pokemon_data)
\`\`\`

### User Interface

#### Frontend Design:

\`\`\`html

<!-- Example HTML for query input -->
<input type="text" id="query" placeholder="Describe the Pokémon">
<input type="file" id="image">
<button onclick="search()">Search</button>
\`\`\`

#### Integrate with Backend:

\`\`\`javascript
async function search() {
const query = document.getElementById('query').value;
const response = await fetch('/search', {
method: 'POST',
headers: {
'Content-Type': 'application/json'
},
body: JSON.stringify({ query: query })
});
const data = await response.json();
displayResults(data);
}
\`\`\`

### Enhanced RAG Integration:

\`\`\`python
import openai
openai.api_key = "your-openai-api-key"

def retrieve_pokemon(user_query):
user_embedding = generate_embeddings(user_query)
query_result = index.query(user_embedding, top_k=5)
return query_result["matches"]

def generate_response(retrieved_data, user_query):
pokemon_list = "\n".join([f"{match['id']}: {match['score']}" for match in retrieved_data])
prompt = f"Based on the following descriptions, find the closest match for the query '{user_query}':\n\n{pokemon_list}"
gpt_response = openai.Completion.create(
engine="davinci",
prompt=prompt,
max_tokens=150
)
return gpt_response.choices[0].text.strip()
\`\`\`

## Folder Strcuture

# pokedex-app

## data/

- raw/
- processed/
- combined/

## backend/

- app/
  - **init**.py
  - main.py
  - api/
    - **init**.py
    - endpoints.py
  - models/
    - **init**.py
    - clip_model.py
    - bert_model.py
  - database/
    - **init**.py
    - setup.py
    - queries.py
- requirements.txt
- README.md

## frontend/

- public/
- src/
  - components/
    - SearchBar.js
    - ResultsDisplay.js
  - App.js
  - index.js
- package.json
- README.md

## scripts/

- download_images.py
- preprocess_images.py
- preprocess_csv.py
- combine_datasets.py

## README.md
