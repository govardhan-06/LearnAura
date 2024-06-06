# LearnAura

## Introduction
**LearnAura** is an AI-powered academic companion designed to assist students in their studies by answering queries using various tools and resources. The application processes uploaded PDF documents to retrieve relevant information and uses multiple APIs to provide comprehensive responses to user queries.

## Features
- **PDF Processing**: Upload and process multiple PDF documents to extract text.
- **Text Chunking**: Split extracted text into manageable chunks for efficient processing.
- **Vector Store**: Create a vector store from text chunks using Cohere embeddings and Chroma.
- **Tool Integration**: Utilize various tools and APIs including:
  - **Document Retriever Tool**
  - **Tavily Search**
  - **Wikipedia Query**
  - **Arxiv Query**
- **Interactive Chat Interface**: Engage with the chatbot to ask questions and receive detailed answers.

## Technologies and Frameworks Used
- **Streamlit**: For building the web application interface.
- **LangChain**: Provides the core functionalities for creating the chatbot and integrating various tools.
  - **LangChain Groq**: For integrating the Groq large language model.
  - **LangChain Chroma**: For creating and managing vector stores.
  - **LangChain Cohere**: For generating embeddings using Cohere's models.
  - **LangChain Core**: For creating prompts and agents.
  - **LangChain Community**: For integrating community-contributed tools like Arxiv and Wikipedia wrappers.
- **PyPDF2**: For reading and extracting text from PDF documents.
- **dotenv**: For managing environment variables securely.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/LearnAura.git
   cd LearnAura
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following keys with your respective API credentials:
   ```
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   TAVILY_API_KEY=your_tavily_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the Streamlit interface in your browser (usually at `http://localhost:8501`).

3. Use the sidebar to upload your PDF documents and click on "Submit & Process".

4. Interact with LearnAura through the chat interface. Ask questions and get responses based on the uploaded documents and other integrated tools.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## Contact

If you have any questions or feedback, feel free to contact us at govardhanar06@gmail.com
