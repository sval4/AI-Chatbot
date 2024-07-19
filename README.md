# AI-Chatbot
A conversational chatbot that uses llama-2-7b-chat.ggmlv3.q8_0.bin model, langchain, and FAISS to read info from passed in URLs and answers questions based on the info it has learned from the URLs.

## How to Run:
Run ingest.py in a linux terminal in order to first create the vector store locally. Then run:
+ **Flask**: python3 model.py
## Link to Access Docker Image:
+ https://hub.docker.com/repository/docker/sval4/chatbot (Current Docker Image is not up to date with code in Github)
+ **Command to Run Docker Image**: docker run -p _Port_:5000 _dockerImageID_
    + _Port_: Refers to any port like 8080. To view the application go to web browser and type in localhost:8080
    + _dockerImageID_: Enter the image ID of the image you are trying to run


// Function to calculate cosine similarity between two strings using TF-IDF vectors
function cosineSimilarity(str1, str2) {
    // Step 1: Tokenize and preprocess the strings (e.g., lowercase, remove punctuation)
    const tokens1 = preprocessString(str1);
    const tokens2 = preprocessString(str2);

    // Step 2: Create a set of unique words (vocabulary)
    const vocabulary = new Set([...tokens1, ...tokens2]);

    // Step 3: Create TF (Term Frequency) vectors
    const tfVector1 = createTFVector(tokens1, vocabulary);
    const tfVector2 = createTFVector(tokens2, vocabulary);

    // Step 4: Create IDF (Inverse Document Frequency) vector
    const idfVector = createIDFVector([tokens1, tokens2], vocabulary);

    // Step 5: Calculate TF-IDF vectors
    const tfidfVector1 = createTFIDFVector(tfVector1, idfVector);
    const tfidfVector2 = createTFIDFVector(tfVector2, idfVector);

    // Step 6: Calculate cosine similarity
    const similarity = calculateCosineSimilarity(tfidfVector1, tfidfVector2);
    return similarity;
}

// Function to preprocess a string (tokenization, lowercase, remove punctuation)
function preprocessString(str) {
    return str.toLowerCase().match(/\b\w+\b/g); // Match words only (remove punctuation)
}

// Function to create TF vector
function createTFVector(tokens, vocabulary) {
    const tfVector = Array.from(vocabulary, word => tokens.filter(token => token === word).length / tokens.length);
    return tfVector;
}

// Function to create IDF vector
function createIDFVector(documents, vocabulary) {
    const idfVector = Array.from(vocabulary, word => Math.log(documents.length / documents.filter(tokens => tokens.includes(word)).length));
    return idfVector;
}

// Function to create TF-IDF vector
function createTFIDFVector(tfVector, idfVector) {
    const tfidfVector = tfVector.map((tf, i) => tf * idfVector[i]);
    return tfidfVector;
}

// Function to calculate cosine similarity
function calculateCosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
    const magnitude1 = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
    const magnitude2 = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
    const similarity = dotProduct / (magnitude1 * magnitude2);
    return similarity;
}

// Example usage:
const text1 = "The quick brown fox jumps over the lazy dog.";
const text2 = "A fast brown fox leaps over a lazy hound.";

const similarityScore = cosineSimilarity(text1, text2);
console.log(`Cosine Similarity Score: ${similarityScore}`);

