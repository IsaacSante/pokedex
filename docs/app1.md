# Pokémon Information App MVP

## Objectives:

1. **Input Handling**:
   - Accept broad text descriptions from the user about a Pokémon.
2. **Processing**:
   - Use a Retrieval-Augmented Generation (RAG) architecture to identify the closest matching Pokémon based on the user's description.
3. **Output**:
   - Print the relevant Pokémon information to the console.

## Components and Workflow:

### 1. Database Setup:

- **Purpose**: Store Pokémon information including names, types, powers, and descriptions.
- **Structure**: Use PostgreSQL (or any preferred database).
- **Schema**:
  ```sql
  CREATE TABLE pokemon (
    id SERIAL PRIMARY KEY,
    name TEXT,
    type TEXT,
    powers TEXT,
    description TEXT
  );
  ```

### 2. Indexing Pokémon Data in Pinecone:

- **Purpose**: Create embeddings for Pokémon descriptions and store them in Pinecone for efficient similarity search.
- **Tools**:

  - Pinecone (Vector Database)
  - Transformers and Torch (for generating embeddings)

- **Code**:

  ```python
  import pinecone
  from transformers import AutoTokenizer, AutoModel
  import torch

  # Initialize Pinecone
  pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")

  # Create Pinecone index
  index_name = "pokemon-index"
  if index_name not in pinecone.list_indexes():
      pinecone.create_index(index_name, dimension=384)  # Adjust dimension based on embedding model

  # Connect to the index
  index = pinecone.Index(index_name)

  # Load LLM tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

  # Function to generate embeddings
  def generate_embeddings(text):
      inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
      with torch.no_grad():
          embeddings = model(**inputs).last_hidden_state.mean(dim=1)
      return embeddings.numpy()

  # Index Pokémon data
  pokemon_data = [
      {"name": "Pikachu", "description": "Electric type Pokémon, yellow, small, mouse-like"},
      {"name": "Charizard", "description": "Fire/Flying type Pokémon, large, dragon-like"},
      # Add other Pokémon data here
  ]

  for pokemon in pokemon_data:
      description = pokemon["description"]
      embedding = generate_embeddings(description)
      index.upsert([(pokemon["name"], embedding)])
  ```

### 3. Retrieval and Response Generation:

- **Purpose**: Retrieve the closest matching Pokémon based on user query and generate a response using an LLM.
- **Tools**:

  - Pinecone (for retrieval)
  - OpenAI API (for generating responses)

- **Code**:

  ```python
  # Function to retrieve Pokémon based on user query
  def retrieve_pokemon(user_query):
      user_embedding = generate_embeddings(user_query)
      query_result = index.query(user_embedding, top_k=5)  # Adjust top_k based on desired number of results
      return query_result["matches"]

  import openai

  openai.api_key = "your-openai-api-key"

  def generate_response(retrieved_data, user_query):
      pokemon_list = "
  ".join([f"{match['id']}: {match['score']}" for match in retrieved_data])
      prompt = f"Based on the following descriptions, find the closest match for the query '{user_query}':\n\n{pokemon_list}"
      gpt_response = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          max_tokens=150
      )
      return gpt_response.choices[0].text.strip()
  ```

### 4. Main Script:

- **Purpose**: Accept user input, process the input using the retrieval and response generation functions, and print the result.
- **Code**:

  ```python
  def main():
      user_query = input("Enter a description of a Pokémon: ")
      retrieved_data = retrieve_pokemon(user_query)
      response = generate_response(retrieved_data, user_query)
      print("Response:\n", response)

  if __name__ == "__main__":
      main()
  ```

## Summary:

This MVP setup focuses on handling broad text descriptions of Pokémon and providing relevant information through a combination of Pinecone for vector similarity search and OpenAI's GPT for response generation. The system is designed to be simple and extendable, laying the groundwork for further enhancements like adding a user interface or incorporating image recognition.
