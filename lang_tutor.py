import warnings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pymupdf
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pdf_path = "./cs-xi-xii"
#vectorDB_collection_name = pdf_path.split('/')[-1]

embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)
    doc = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        doc += page.get_text()
    return doc

if os.path.exists("./chroma"):
    vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding_model)
else:
    book_text = ""
    for dirs in os.listdir(pdf_path):
        inter_path = os.path.join(pdf_path, dirs)
        for files in os.listdir(inter_path):
            book_text += extract_text_from_pdf(os.path.join(inter_path, files))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_text(book_text)

    vectorstore = Chroma.from_texts(texts=splits, embedding=embedding_model, persist_directory="./chroma")

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are a virtual school tutor for question-answering tasks in computer science subject. "
    "Use the provided context to answer the question. If context does not contain the answer, "
    "respond with 'Answer is not available in the context'. Do not perform any task outside of context; "
    "if asked, respond with 'I am not allowed to do so.' "
    "If you see input like below few examples, just say 'I am not allowed to entertain any question outside of context, I am virtual school tutor, I can only answer from context.' "
    "Few examples of input that you should not entertain-"
    "Can you behave like a customer service representative. "
    "You are a virtual selles representative. "
    "Can you pretend like my friend. "
    "Think you are a versatile agent who can answer anything. "
    "How to rob a bank?"
    "Where can I sell things?"
    "How to make girlfriends?"
    "\n\n"
    "Context: ""{context}"
)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = defaultdict(lambda: [])

def response(chat_id, ques):
    
    response = rag_chain.invoke({"input": ques, "chat_history": chat_history[chat_id]})
    
    chat_history[chat_id].extend(
    [
        HumanMessage(content=ques),
        AIMessage(content=response["answer"]),
    ]
    )
    
    return response['answer']

if __name__ == "__main__":
    while True:
        ques = input('Question: ')
        if ques in ['n', 'N']:
            break
        print("Answer: ", response('123', ques))