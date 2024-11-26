# **Chat-with-PDF-Gemini**
This project is a web application that allows users to upload PDF documents and ask context-based questions. Using the power of Google Gemini (Large Language Model) and LangChain, the application processes the uploaded documents, extracts relevant information, and generates accurate responses based on the provided context.

## **Features**
- **Multi-PDF Support**: Upload multiple PDF files simultaneously.
- **Contextual Question Answering**: Answers questions using the provided context from the uploaded documents.
- **AI-Powered Processing**: Utilizes Google Gemini for embeddings and conversational AI.
- **Efficient Document Handling**: Leverages vector-based search using FAISS for fast and accurate information retrieval.
- **Web Interface**: Built with Streamlit for a user-friendly experience.

## **Technologies Used**
### **Programming Language**
- Python

### **Libraries and Frameworks**
- **[Streamlit](https://streamlit.io/)**: For building the web interface.
- **[Google Generative AI](https://developers.generativeai.google/)**: For interacting with Google Gemini API.
- **[LangChain](https://docs.langchain.com/)**: For chaining the question-answering workflow.
- **[PyPDF2](https://pypi.org/project/PyPDF2/)**: For extracting text from PDF documents.
- **[FAISS](https://github.com/facebookresearch/faiss)**: For vector-based similarity search.

## **Application Workflow**

### 1. **PDF Upload and Text Extraction**
   - Users upload one or multiple PDF documents.
   - The `get_pdf_text()` function extracts the text from the uploaded files using PyPDF2.

### 2. **Text Chunking**
   - Long documents are split into manageable chunks using LangChain's `RecursiveCharacterTextSplitter` to ensure effective processing by the AI model.

### 3. **Vectorization and Indexing**
   - The text chunks are embedded into vector space using **GoogleGenerativeAIEmbeddings**.
   - A FAISS index is created and stored locally for fast similarity-based retrieval.

### 4. **Contextual Question Answering**
   - When a question is asked, the application:
     1. Loads the FAISS index.
     2. Finds the most relevant document chunks using similarity search.
     3. Uses a predefined **PromptTemplate** and **Google Gemini** to generate an accurate, context-aware response.

### 5. **Web Interface**
   - Built with Streamlit, the interface provides:
     - A sidebar for PDF uploads.
     - A text input field for user queries.
     - Dynamic responses displayed in the application.

## **How to Run the Application**

### **1. Clone the Repository**
```bash
git clone https://github.com/Nayan7/Chat-with-PDF-Gemini.git
cd chat-with-pdf-gemini

### **2. Install Dependencies**
Ensure you have Python installed on your system, then install the required dependencies. It is highly recommended to use a virtual environment for this.
Create a Virtual Environment (optional but recommended):
python -m venv venv

Activate the virtual environment:
For Windows:
venv\Scripts\activate
For Mac:
source venv/bin/activate

Install the necessary Python packages by running the following command:
pip install -r requirements.txt

### **3. Set up API Keys**
You will need a Google API key to interact with Google Gemini. Follow these steps:

1) Sign up for Google Cloud and enable the Generative AI API.
2) Obtain your API key from Google Cloud.

Create a .env file in the root directory of your project, and add the following line with your API key:
GOOGLE_API_KEY=your_google_api_key_here

### **4. Run the application**
Now that you have set up the dependencies and API key, you can run the application:
streamlit run app.py

This will start the application locally, and you should see a link in your terminal. Open the link in your browser to interact with the app.

### **5. Upload PDFs and Ask Questions**
- Use the sidebar to upload multiple PDF files.
- Type your question in the input box and receive answers generated from the context of the uploaded PDFs.

## **How to Run the Application**
1) Upload multiple PDF files via the sidebar.
2) Type a question like "What is the summary of the second document?".
3) Get an accurate response based on the uploaded content.

## **Future Enhancements**
- Add support for other document formats (e.g., Word, Excel).
- Integrate caching mechanisms to improve processing speed.
- Enhance the UI/UX with custom themes and layouts.

## **Contributions**
Contributions are welcome! Feel free to open issues or submit pull requests.

## **Acknowledgements**
- Google Generative AI for providing state-of-the-art embeddings and LLM capabilities.
- LangChain for simplifying chain creation for contextual question-answering.
- FAISS for efficient similarity search.
