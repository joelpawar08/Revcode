import os
import logging
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import pypdf
import base64

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("weaviate-demo")

# --- Cluster connection ---
WEAVIATE_URL = "https://zg0jpid3qqkztv81ago3bq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "Sm5ublBIMkZvdFRuUGFJQ19vbFp0MmJvR1l0WUhadjFQT2ZTTm1nZVJuNDdpQlVUMXhmUmd0RytlZmVBPV92MjAw"

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    logger.error("WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set.")
    exit(1)

try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
    )
    logger.info("Connected to Weaviate cluster.")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    exit(1)

# --- Collection creation ---
COLLECTION_NAME = "PythonDocs"

if client.collections.exists(COLLECTION_NAME):
    logger.info(f"Collection '{COLLECTION_NAME}' already exists. Deleting for a fresh start.")
    client.collections.delete(COLLECTION_NAME)

try:
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="file", data_type=DataType.BLOB),
            Property(name="filename", data_type=DataType.TEXT)
        ],
        vector_config=[
            Configure.Vectors.text2vec_weaviate(
                name="document_vector",
                source_properties=["text"],
                model="Snowflake/snowflake-arctic-embed-l-v2.0"
            )
        ]
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created with Snowflake internal embeddings.")
except Exception as e:
    logger.error(f"Failed to create collection: {e}")
    client.close()
    exit(1)

# --- Add PDF to collection ---
PDF_PATH = "example.pdf"  # Replace with your PDF file path

if not os.path.isfile(PDF_PATH):
    logger.error(f"PDF file '{PDF_PATH}' not found.")
    client.close()
    exit(1)

# Extract text from PDF using pypdf and encode PDF as base64
try:
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
        f.seek(0)
        reader = pypdf.PdfReader(f)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
    # Encode PDF bytes to base64 string for Weaviate BLOB property
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    logger.info(f"Read PDF file '{PDF_PATH}' ({len(pdf_bytes)} bytes) and extracted text ({len(extracted_text)} characters).")
except Exception as e:
    logger.error(f"Failed to read or extract text from PDF file: {e}")
    client.close()
    exit(1)

try:
    collection = client.collections.get(COLLECTION_NAME)
    collection.data.insert(
        properties={
            "file": pdf_base64,  # Must be base64-encoded string
            "filename": os.path.basename(PDF_PATH),
            "text": extracted_text
        }
    )
    logger.info(f"PDF '{PDF_PATH}' added to collection '{COLLECTION_NAME}' with extracted text.")
except Exception as e:
    logger.error(f"Failed to add PDF to collection: {e}")
finally:
    client.close()
    logger.info("Closed Weaviate client.")