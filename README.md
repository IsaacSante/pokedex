# Pokédex

## Overview

This application combines comprehensive identification features of a multimodal input app with the quick and broad description handling capabilities of a Retrieval-Augmented Generation (RAG) system. The goal is to create a versatile and efficient Pokémon Identification App that can handle text, image, and combined queries, providing detailed and accurate information about Pokémon.

## Datasets

- [Pokémon Dataset by Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/pokemon)
- [Pokémon Image Dataset by hlrhegemony](https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset/data)

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

## Folder Strcuture

### pokedex-app

#### data/

- raw/
- processed/
- combined/

#### backend/

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

#### frontend/

- public/
- src/
  - components/
    - SearchBar.js
    - ResultsDisplay.js
  - App.js
  - index.js
- package.json
- README.md

#### scripts/

- download_images.py
- preprocess_images.py
- preprocess_csv.py
- combine_datasets.py

#### README.md
