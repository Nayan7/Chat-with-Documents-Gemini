from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from docx import Document
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_word_text(word_docs):
    text = ''
    for word in word_docs:
        # Directly pass the UploadedFile object to python-docx
        doc = Document(word)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to 
    provide all the details, if the answer is not in the provided context just say, 'answer is not available in the context', 
    do not provide the wrong answer\n\n
    Context:\n {context}? \n
    Question: \n {question} \n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
    chain = load_qa_chain(model, chain_type = 'stuff', prompt = prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')

    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents':docs, 'question': user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response['output_text'])

def main():
    st.set_page_config('Chat With Multiple Documents')
    st.header("Chat with your Documents (Word & PDF) using Gemini👻")

    user_question = st.text_input("Ask a Question from the documents uploaded")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        st.markdown(
            """
            **Instructions:**
            - Upload the PDF and/or Word documents you want to process.
            - After uploading, click on the **Submit & Process** button to process your documents.
            - Once processing is complete, ask your questions in the input box above.
            """
        )

        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        word_docs = st.file_uploader("Upload your Word Files", accept_multiple_files=True, type=["docx"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_pdf_text = get_pdf_text(pdf_docs) if pdf_docs else ''
                raw_word_text = get_word_text(word_docs) if word_docs else ''

                raw_text = raw_pdf_text + raw_word_text
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
                else:
                    st.warning("No valid documents to process. Please upload PDF or Word files.")
                #raw_text = get_pdf_text(pdf_docs)
                #text_chunks = get_text_chunks(raw_text)
                #get_vector_store(text_chunks)
                #st.success("Done")

if __name__ == '__main__':
    main()
