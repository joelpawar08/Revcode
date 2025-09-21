import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import weaviate
from weaviate.classes.init import Auth
from weaviate.collections.classes.internal import GenerativeSearchReturnType
import time
import traceback

# --- Logging Configuration ---
def setup_logging():
    """Set up comprehensive logging configuration."""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(
        f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()

# --- Configuration ---
class Config:
    """Application configuration."""
    
    # Weaviate Configuration
    WEAVIATE_URL: str = os.getenv(
        "WEAVIATE_URL", 
        "https://zg0jpid3qqkztv81ago3bq.c0.asia-southeast1.gcp.weaviate.cloud"
    )
    WEAVIATE_API_KEY: str = os.getenv(
        "WEAVIATE_API_KEY",
        "Sm5ublBIMkZvdFRuUGFJQ19vbFp0MmJvR1l0WUhadjFQT2ZTTm1nZVJuNDdpQlVUMXhmUmd0RytlZmVBPV92MjAw"
    )
    COHERE_API_KEY: str = os.getenv(
        "COHERE_API_KEY",
        "vWoOCbbQSb6MsGVzVeawpqyS9zni3XLA1sPRsqIG"
    )
    
    # Collection Configuration
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "PythonDocs")
    
    # API Configuration
    API_TITLE: str = "RAG API with Weaviate"
    API_DESCRIPTION: str = "A production-ready RAG (Retrieval-Augmented Generation) API using Weaviate and Cohere"
    API_VERSION: str = "1.0.0"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "https://localhost:3000",
        "https://localhost:3001",
        "https://localhost:8080",
        "*"  # Allow all origins for development - remove in production
    ]
    
    # Security Configuration
    TRUSTED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*.localhost", "*"]
    
    # Search Configuration
    DEFAULT_LIMIT: int = 3
    DEFAULT_ALPHA: float = 0.5
    MAX_CONTENT_LENGTH: int = 1000

config = Config()

# --- Initialize Weaviate Client ---
def create_weaviate_client():
    """Create and configure Weaviate client."""
    try:
        client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=config.WEAVIATE_URL,
            auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
            headers={
                "X-Cohere-Api-Key": config.COHERE_API_KEY,
                "X-Cohere-BaseURL": "https://api.cohere.com"
            }
        )
        logger.info("Weaviate client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create Weaviate client: {str(e)}")
        raise

async_client = create_weaviate_client()

# --- Request/Response Models ---
class PromptRequest(BaseModel):
    """Request model for the answer endpoint."""
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="The question or prompt to search for",
        json_schema_extra={"example": "What is Python?"}
    )

class SearchResult(BaseModel):
    """Individual search result model."""
    source: str = Field(..., description="Source of the information")
    answer: str = Field(..., description="Generated answer or content")
    confidence: float = Field(default=0.0, description="Confidence score if available")

class AnswerResponse(BaseModel):
    """Response model for the answer endpoint."""
    results: List[SearchResult]
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    total_results: int = Field(..., description="Total number of results returned")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str  # Changed from datetime to str
    weaviate_ready: bool
    version: str

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: str  # Changed from datetime to str

# --- Collection Management Functions ---
async def check_collection_exists(collection_name: str) -> bool:
    """Check if a collection exists in Weaviate."""
    try:
        collection = async_client.collections.get(collection_name)
        await collection.config.get()
        return True
    except Exception as e:
        logger.warning(f"Collection {collection_name} does not exist or is not accessible: {str(e)}")
        return False

async def list_available_collections() -> List[str]:
    """List all available collections."""
    try:
        collections = await async_client.collections.list_all()
        return [collection.name for collection in collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        return []

# --- Middleware and Exception Handlers ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    try:
        logger.info("Starting up RAG API...")
        await async_client.connect()
        logger.info("Connected to Weaviate successfully")
        
        # Test connection
        is_ready = await async_client.is_ready()
        if not is_ready:
            logger.warning("Weaviate is not ready during startup")
        else:
            logger.info("Weaviate is ready and operational")
            
        # Check if collection exists
        collection_exists = await check_collection_exists(config.COLLECTION_NAME)
        if not collection_exists:
            logger.warning(f"Collection '{config.COLLECTION_NAME}' does not exist")
            available_collections = await list_available_collections()
            logger.info(f"Available collections: {available_collections}")
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("Shutting down RAG API...")
        await async_client.close()
        logger.info("Disconnected from Weaviate successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# --- FastAPI App Setup ---
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Trusted Host Middleware ---
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.TRUSTED_HOSTS
)

# --- Request Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"after {process_time:.3f}s - {str(e)}"
        )
        raise

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(ErrorResponse(
            error="HTTP Exception",
            detail=exc.detail,
            timestamp=datetime.now().isoformat()
        ).model_dump())
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).model_dump())
    )

# --- API Endpoints ---
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API with Weaviate",
        "version": config.API_VERSION,
        "docs_url": "/docs",
        "health_url": "/healthcheck"
    }

@app.get("/healthcheck", response_model=HealthResponse)
async def healthcheck():
    """
    Comprehensive health check endpoint.
    """
    try:
        is_ready = await async_client.is_ready()
        logger.info(f"Health check - Weaviate ready: {is_ready}")
        
        return HealthResponse(
            status="healthy" if is_ready else "degraded",
            timestamp=datetime.now().isoformat(),
            weaviate_ready=is_ready,
            version=config.API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - health check failed"
        )

@app.get("/collections")
async def list_collections():
    """List all available collections in Weaviate."""
    try:
        collections = await list_available_collections()
        collection_exists = await check_collection_exists(config.COLLECTION_NAME)
        
        return {
            "available_collections": collections,
            "configured_collection": config.COLLECTION_NAME,
            "configured_collection_exists": collection_exists
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve collections: {str(e)}")

@app.get("/debug/collection")
async def debug_collection():
    """Debug endpoint to check collection configuration."""
    try:
        collections = await list_available_collections()
        collection_exists = await check_collection_exists(config.COLLECTION_NAME)
        
        result = {
            "collection_name": config.COLLECTION_NAME,
            "collection_exists": collection_exists,
            "available_collections": collections
        }
        
        if collection_exists:
            try:
                collection = async_client.collections.get(config.COLLECTION_NAME)
                config_info = await collection.config.get()
                result["config"] = str(config_info)
                logger.info(f"Collection debug info retrieved for: {config.COLLECTION_NAME}")
            except Exception as config_error:
                result["config_error"] = str(config_error)
        
        return result
        
    except Exception as e:
        logger.error(f"Collection debug failed: {str(e)}")
        return {
            "collection_name": config.COLLECTION_NAME,
            "collection_exists": False,
            "error": str(e),
            "available_collections": []
        }

@app.post("/answer", response_model=AnswerResponse)
async def answer(request: PromptRequest):
    """
    Main RAG endpoint that accepts a prompt and returns generated answers with sources.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: '{request.prompt[:100]}...'")
        
        # Check if Weaviate is ready
        if not await async_client.is_ready():
            raise HTTPException(status_code=503, detail="Weaviate service is not ready")

        # Check if collection exists
        collection_exists = await check_collection_exists(config.COLLECTION_NAME)
        if not collection_exists:
            available_collections = await list_available_collections()
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{config.COLLECTION_NAME}' not found. Available collections: {available_collections}"
            )

        # Get the collection
        collection = async_client.collections.get(config.COLLECTION_NAME)
        logger.debug(f"Using collection: {collection.name}")
        
        results = []
        
        # Try generative search first
        try:
            logger.debug("Attempting generative search...")
            response = await collection.generate.hybrid(
                query=request.prompt,
                single_prompt=f"Based on the context provided, please answer the following question: {request.prompt}",
                limit=config.DEFAULT_LIMIT,
                alpha=config.DEFAULT_ALPHA
            )
            
            # Process generative search results
            for obj in response.objects:
                source = obj.properties.get("source", "Unknown source")
                answer = obj.generated if hasattr(obj, 'generated') and obj.generated else "No generated answer available"
                
                # Safe confidence extraction
                confidence = 0.0
                try:
                    if hasattr(obj, 'metadata') and obj.metadata is not None:
                        if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                            confidence = float(obj.metadata.score)
                        elif hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                            confidence = 1.0 - float(obj.metadata.distance)  # Convert distance to confidence
                except (ValueError, TypeError, AttributeError):
                    confidence = 0.0
                
                results.append(SearchResult(
                    source=source,
                    answer=answer,
                    confidence=confidence
                ))
            
            logger.info(f"Generative search completed with {len(results)} results")
            
        except Exception as gen_error:
            logger.warning(f"Generative search failed, falling back to regular search: {str(gen_error)}")
            
            # Fallback to regular hybrid search
            try:
                response = await collection.query.hybrid(
                    query=request.prompt,
                    limit=config.DEFAULT_LIMIT,
                    alpha=config.DEFAULT_ALPHA
                )
                
                # Process regular search results
                for obj in response.objects:
                    source = obj.properties.get("source", "Unknown source")
                    content = obj.properties.get("content", obj.properties.get("text", "No content available"))
                    
                    # Safe confidence extraction
                    confidence = 0.0
                    try:
                        if hasattr(obj, 'metadata') and obj.metadata is not None:
                            if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                                confidence = float(obj.metadata.score)
                            elif hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                                confidence = 1.0 - float(obj.metadata.distance)  # Convert distance to confidence
                    except (ValueError, TypeError, AttributeError):
                        confidence = 0.0
                    
                    # Truncate content for readability
                    truncated_content = content[:config.MAX_CONTENT_LENGTH]
                    if len(content) > config.MAX_CONTENT_LENGTH:
                        truncated_content += "..."
                    
                    results.append(SearchResult(
                        source=source,
                        answer=f"Retrieved content: {truncated_content}",
                        confidence=confidence
                    ))
                
                logger.info(f"Regular search completed with {len(results)} results")
                
            except Exception as search_error:
                logger.error(f"Both generative and regular search failed: {str(search_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail="Search functionality is currently unavailable"
                )
        
        # Handle case where no results are found
        if not results:
            logger.warning("No results found for the query")
            results.append(SearchResult(
                source="System",
                answer="No relevant information found for your query. Please try rephrasing your question or check if the collection contains relevant data.",
                confidence=0.0
            ))
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query completed in {query_time_ms:.2f}ms with {len(results)} results")
        
        return AnswerResponse(
            results=results,
            query_time_ms=query_time_ms,
            total_results=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        query_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Answer endpoint error after {query_time_ms:.2f}ms: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# --- Additional Utility Endpoints ---
@app.get("/metrics")
async def get_metrics():
    """Get basic API metrics."""
    try:
        is_ready = await async_client.is_ready()
        collection_exists = await check_collection_exists(config.COLLECTION_NAME)
        
        return {
            "weaviate_status": "ready" if is_ready else "not_ready",
            "api_version": config.API_VERSION,
            "collection_name": config.COLLECTION_NAME,
            "collection_exists": collection_exists,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        return {
            "weaviate_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG API server...")
    uvicorn.run(
        "rag:app",  # Fixed to use correct module name
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )