## To-Do List

1. **Data Preparation:**

   - Download Pokémon images and preprocess them.
   - Load and preprocess the general info CSV file.
   - Ensure each dataset uses the Pokémon name as a unique identifier.

2. **Model Training:**

   - Train/fine-tune the CLIP model on the Pokémon image dataset to create image embeddings.
   - Generate text embeddings using the CLIP model for Pokémon descriptions from the general info dataset.

3. **Database Setup:**

   - Define schema for Pokémon metadata, image URLs, and embeddings in a relational database.
   - Store text and image embeddings in a vector database.
   - Ensure each Pokémon’s data can be accessed using the name as the key.

4. **Backend Development:**

   - Create scripts to preprocess data, generate embeddings, and store them in the respective databases.
   - Develop API endpoints for text, image, and combined searches.
   - Integrate embedding search with vector and relational databases, using names as identifiers.

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
