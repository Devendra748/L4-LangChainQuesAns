# LangChain: Q&A

LangChain is a tool that enables querying and retrieving information from a collection of documents using natural language processing techniques. It provides a way to interact with a product catalog or any other dataset to find specific items of interest.

## Installation
To install LangChain, use the following command:
```pip install --upgrade langchain```
Make sure to have the required dependencies installed. You can also refer to the installation guide for more details.

## Usage
Here's an example of how you can use LangChain in your Python code:
```import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

# Load document from a CSV file
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# Create an index of the documents
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Query the index
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query)

# Display the response
display(Markdown(response))
```

This example demonstrates how to load a collection of documents from a CSV file, create an index for efficient querying, and retrieve information based on a specific question.

You can customize the behavior of LangChain by adjusting the parameters and options according to your needs.

## Additional Features
LangChain provides additional functionalities, such as document embeddings and conversational models, to enhance the Q&A process. Here's an example of how to use some of these features:
```
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

# Load documents
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

# Generate document embeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi, my name is Harrison")

# Create an index with embeddings
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Perform similarity search
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

# Use a conversational model for question-answering
retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0.0)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.call_as_llm(f"{qdocs} Question: Please list all your shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))

# Perform retrieval-based question-answering
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
```
This example showcases how to generate document embeddings, perform similarity search, and utilize a conversational model for question-answering.

Feel free to explore and experiment with different combinations of these features to suit your specific use case.

## Contributing
If you encounter any issues or have suggestions for improvements, please feel free to open an issue. Contributions to the project are also welcome. You can submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License. Please refer to the LICENSE file for more details.

## Acknowledgements
LangChain makes use of various open-source libraries and models. We acknowledge and thank the developers and contributors of these projects for their valuable work.

Please refer to the documentation and the GitHub repository for LangChain for more information, examples, and detailed usage instructions.
