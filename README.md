# AI-Chatbot
A conversational chatbot that uses llama-2-7b-chat.ggmlv3.q8_0.bin model, langchain, and FAISS to read info from passed in URLs and answers questions based on the info it has learned from the URLs.

## How to Run:
Run ingest.py in a linux terminal in order to first create the vector store locally. Then run:
+ **Flask**: python3 model.py
## Link to Access Docker Image:
+ https://hub.docker.com/repository/docker/sval4/chatbot
+ **Command to Run Docker Image**: docker run -p _Port_:5000 _dockerImageID_
    + _Port_: Referse to any port that is not 5000 like 8080. To view the application go to web browser and type in localhost:8080
    + _dockerImageID_: Enter the image ID of the image you are trying to run