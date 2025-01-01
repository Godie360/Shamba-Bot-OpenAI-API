from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from enum import Enum
import os
import glob
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

# Pydantic Models
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message in English or Swahili")
    session_id: Optional[str] = Field(default="default", description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "How do I plant maize?",
                "session_id": "user123"
            }
        }

class ChatResponse(BaseModel):
    answer: str = Field(..., description="SHAMBA BOT's response")
    source_documents: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To plant maize, first prepare your soil well...",
                "source_documents": ["farming_guide.pdf", "maize_planting.pdf"]
            }
        }

class ProcessingStatus(BaseModel):
    status: DocumentStatus
    message: str
    document_id: Optional[str] = None
    chunks_processed: Optional[int] = None
    error: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="SHAMBA BOT - AI Agricultural Advisor",
    description="""
    SHAMBA BOT is an AI agricultural advisor that helps farmers with agricultural information.
    Upload agricultural documents and get expert farming advice in English or Swahili.
    """,
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
docs_dir = 'documents'
db_dir = 'shamba_bot_db'
os.makedirs(docs_dir, exist_ok=True)
os.makedirs(db_dir, exist_ok=True)

# Initialize LLM and embeddings
llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4-turbo-preview",
    temperature=0.7
)

streaming_llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4-turbo-preview",
    temperature=0.7,
    streaming=True
)

embedding = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-large",
    dimensions=1536
)

# Chat history storage
chat_stores = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_stores:
        chat_stores[session_id] = ChatMessageHistory()
    return chat_stores[session_id]

def update_processing_status(document_id: str, status: ProcessingStatus):
    """Update document processing status."""
    status_file = f"{docs_dir}/{document_id}_status.json"
    with open(status_file, 'w') as f:
        json.dump(status.dict(), f)

async def process_pdf(file_path: str, document_id: str) -> ProcessingStatus:
    """Process PDF and generate embeddings."""
    try:
        # Update status to processing
        update_processing_status(document_id, ProcessingStatus(
            status=DocumentStatus.PROCESSING,
            message="Loading PDF document",
            document_id=document_id
        ))

        # Load PDF
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = loader.load_and_split(text_splitter)
        
        if not documents:
            raise ValueError("No content found in PDF")
            
        # Filter for agricultural content
        agricultural_terms = [
            # English terms
            'agriculture', 'farming', 'crop', 'livestock', 'soil', 'harvest',
            'fertilizer', 'irrigation', 'pesticide', 'seeds', 'planting',
            # Swahili terms
            'kilimo', 'ukulima', 'mazao', 'mifugo', 'udongo', 'mavuno',
            'mbolea', 'umwagiliaji', 'dawa', 'mbegu', 'kupanda'
        ]
        
        agricultural_docs = [
            doc for doc in documents 
            if any(term in doc.page_content.lower() for term in agricultural_terms)
        ]
        
        if not agricultural_docs:
            raise ValueError("No agricultural content found in document")
        
        update_processing_status(document_id, ProcessingStatus(
            status=DocumentStatus.PROCESSING,
            message="Generating embeddings",
            document_id=document_id
        ))

        # Prepare texts and metadata
        texts = [doc.page_content for doc in agricultural_docs]
        metadatas = [{
            "document_id": document_id,
            "chunk_id": i,
            "source": os.path.basename(file_path),
            "date_processed": datetime.now().isoformat()
        } for i in range(len(agricultural_docs))]
        
        # Add to vector store
        db = Chroma(
            persist_directory=db_dir,
            embedding_function=embedding
        )
        
        # Using add_texts instead of add_documents
        db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=[f"{document_id}_chunk_{i}" for i in range(len(texts))]
        )
        
        status = ProcessingStatus(
            status=DocumentStatus.COMPLETED,
            message="Document processed successfully",
            document_id=document_id,
            chunks_processed=len(agricultural_docs)
        )
        update_processing_status(document_id, status)
        return status
        
    except Exception as e:
        error_status = ProcessingStatus(
            status=DocumentStatus.FAILED,
            message="Processing failed",
            document_id=document_id,
            error=str(e)
        )
        update_processing_status(document_id, error_status)
        raise e

def build_qa_chain(streaming: bool = False):
    """Build QA chain for SHAMBA BOT."""
    db = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding
    )
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )
    
    system_prompt = """You are SHAMBA BOT, an AI agricultural advisor created by SMART SHAMBA LLC.
    Use the following agricultural information to answer farming questions.
    Answer in the same language as the question (English or Swahili).
    If you don't know the answer, say so politely.
    
    Wewe ni SHAMBA BOT, mshauri wa kilimo wa AI uliyetengenezwa na SMART SHAMBA LLC.
    Tumia maelezo yafuatayo ya kilimo kujibu maswali ya ukulima.
    Jibu kwa lugha ile ile ya swali (Kiingereza au Kiswahili).
    Ikiwa hujui jibu, sema kwa upole.
    
    Context: {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    selected_llm = streaming_llm if streaming else llm
    document_chain = create_stuff_documents_chain(selected_llm, qa_prompt)
    
    return create_retrieval_chain(retriever, document_chain)

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "Welcome to SHAMBA BOT API 🌱"}


@app.post("/upload",
    tags=["Document Management"],
    summary="Upload Agricultural Document")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file with agricultural content")
):
    """Upload and process agricultural PDF document."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    try:
        document_id = str(uuid.uuid4())
        file_path = f"{docs_dir}/{document_id}_{file.filename}"
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process in background
        background_tasks.add_task(process_pdf, file_path, document_id)
        
        return {
            "message": "Document upload started",
            "status": "processing",
            "document_id": document_id,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/documents/{document_id}/status",
    tags=["Document Management"],
    summary="Check Document Processing Status")
async def get_processing_status(document_id: str):
    """Get document processing status."""
    status_file = f"{docs_dir}/{document_id}_status.json"
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404,
            detail="Document status not found"
        )
    
    with open(status_file, 'r') as f:
        return json.load(f)

@app.post("/chat",
    tags=["Chat"],
    summary="Chat with SHAMBA BOT",
    response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat with SHAMBA BOT."""
    try:
        history = get_session_history(request.session_id)
        chain = build_qa_chain(streaming=False)
        
        response = chain.invoke({
            "input": request.message,
            "chat_history": history.messages
        })
        
        # Extract source documents
        source_documents = list(set([
            os.path.basename(doc.metadata.get('source', ''))
            for doc in response.get('context', [])
            if doc.metadata.get('source')
        ]))
        
        history.add_user_message(request.message)
        history.add_ai_message(response['answer'])
        
        return ChatResponse(
            answer=response['answer'],
            source_documents=source_documents
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.websocket("/ask/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    try:
        while True:
            # Receive data from the client
            data = await websocket.receive_json()
            message = data.get("message")
            session_id = data.get("session_id", "default")
            
            if not message:
                # Respond with an error if the message is missing
                await websocket.send_json({"error": "Message required"})
                continue
            
            # Retrieve chat history and build the QA chain
            history = get_session_history(session_id)
            chain = build_qa_chain(streaming=True)
            
            # Stream response chunks
            async for chunk in chain.astream({
                "input": message,
                "chat_history": history.messages
            }):
                if "answer" in chunk:
                    await websocket.send_json({
                        "chunk": chunk["answer"]  # Send the meaningful answer
                    })
                else:
                    print("Unexpected chunk structure:", chunk)
            
            # Update history with user and AI messages
            history.add_user_message(message)
            history.add_ai_message(chunk.get("answer", ""))
            
            # Signal the end of the stream
            await websocket.send_json({"end": True})
    
    except WebSocketDisconnect:
        print("WebSocket connection disconnected")
    except Exception as e:
        # Send error details back to the client
        print("Error:", str(e))
        await websocket.send_json({"error": str(e)})    

@app.get("/health",
    tags=["System"],
    summary="Health Check")
async def health_check():
    """Check system health."""
    try:
        db = Chroma(persist_directory=db_dir, embedding_function=embedding)
        collection = db.get()
        document_count = len(set([
            meta.get('document_id') 
            for meta in collection['metadatas'] 
            if meta and 'document_id' in meta
        ]))
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_chunks": len(collection['ids']),
            "total_documents": document_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )