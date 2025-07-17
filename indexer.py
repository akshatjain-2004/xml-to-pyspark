import os
import time
import json
import pickle
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from lxml import etree
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
XML_FILE_PATH = 'SIL_WorkforceEventFact.xml'
VECTOR_STORE_PATH = 'informatica_vector_store'
DOCUMENTS_BACKUP_PATH = 'documents_backup.pkl'
PROGRESS_FILE = 'embedding_progress.json'
BATCH_SIZE = 5  # Process documents in batches
RETRY_DELAY = 65  # Wait 65 seconds (5 seconds buffer) after rate limit

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

def chunk_informatica_xml(file_path):
    """
    Intelligently chunks the Informatica XML file by major object definitions.
    Each Source, Target, Transformation, Mapplet, and Mapping becomes a separate Document.
    """
    logger.info(f"Parsing XML file: {file_path}")
    try:
        # Use 'rb' for binary read mode to handle various encodings and recover from minor errors
        with open(file_path, 'rb') as f:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(f, parser)
    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found. Please update the XML_FILE_PATH variable.")
        return []
    except etree.XMLSyntaxError as e:
        logger.error(f"XML syntax error in '{file_path}': {e}")
        return []

    root = tree.getroot()
    documents = []
    
    # We assume all relevant objects are within a <FOLDER> tag, which is standard.
    for folder in root.findall('.//FOLDER'):
        folder_name = folder.get('NAME', 'unknown_folder')
        
        # Process all direct children of the folder, which are the main objects
        for element in folder:
            # Check if the element is a valid tag to avoid processing comments or other nodes
            if not isinstance(element.tag, str):
                continue

            # Use lxml's tostring to get the raw XML content of the element
            content = etree.tostring(element, pretty_print=True).decode('utf-8')
            tag_name = element.tag
            obj_name = element.get('NAME', 'unnamed')
            
            # Create rich metadata for each document. This is crucial for the agent's tools.
            metadata = {
                'source': file_path,
                'folder_name': folder_name,
                'tag_name': tag_name,
                'object_name': obj_name,
                'object_type': element.get('TYPE', 'N/A') # e.g., 'Expression', 'Lookup Procedure'
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
    logger.info(f"Successfully created {len(documents)} documents from the XML.")
    return documents

def save_documents_backup(documents: List[Document], backup_path: str):
    """Save documents to a backup file to preserve them for later analysis."""
    logger.info(f"Saving document backup to: {backup_path}")
    with open(backup_path, 'wb') as f:
        pickle.dump(documents, f)
    logger.info("Document backup saved successfully.")

def load_documents_backup(backup_path: str) -> List[Document]:
    """Load documents from backup file."""
    logger.info(f"Loading document backup from: {backup_path}")
    with open(backup_path, 'rb') as f:
        documents = pickle.load(f)
    logger.info(f"Loaded {len(documents)} documents from backup.")
    return documents

def save_progress(processed_count: int, total_count: int, progress_file: str):
    """Save processing progress to a JSON file."""
    progress = {
        'processed': processed_count,
        'total': total_count,
        'percentage': (processed_count / total_count) * 100 if total_count > 0 else 0
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress(progress_file: str) -> dict:
    """Load processing progress from JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'processed': 0, 'total': 0, 'percentage': 0}

def create_embeddings_with_rate_limit(documents: List[Document], embeddings_model, start_index: int = 0) -> Optional[FAISS]:
    """
    Create embeddings with rate limit handling and progress tracking.
    """
    total_docs = len(documents)
    processed_docs = start_index
    
    logger.info(f"Starting embedding creation from document {start_index} of {total_docs}")
    
    # Process documents in batches
    for i in range(start_index, total_docs, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_docs)
        batch_docs = documents[i:batch_end]
        
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}: documents {i+1}-{batch_end} of {total_docs}")
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                if i == start_index:
                    # First batch - create new vector store
                    vector_store = FAISS.from_documents(batch_docs, embeddings_model)
                    logger.info("Created initial vector store")
                else:
                    # Add to existing vector store using add_documents method
                    vector_store.add_documents(batch_docs)
                    logger.info(f"Added batch to vector store")
                
                processed_docs = batch_end
                save_progress(processed_docs, total_docs, PROGRESS_FILE)
                
                # Save intermediate progress
                if processed_docs % (BATCH_SIZE * 5) == 0:  # Save every 5 batches
                    temp_path = f"{VECTOR_STORE_PATH}_temp_{processed_docs}"
                    vector_store.save_local(temp_path)
                    logger.info(f"Saved intermediate progress at {temp_path}")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    retry_count += 1
                    logger.warning(f"Rate limit hit. Attempt {retry_count}/{max_retries}. Waiting {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Unexpected error: {e}")
                    raise e
        
        if retry_count >= max_retries:
            logger.error(f"Failed to process batch after {max_retries} attempts. Stopping at document {processed_docs}")
            return vector_store if 'vector_store' in locals() else None
    
    logger.info("All documents processed successfully!")
    return vector_store

def create_vector_store(resume_from_backup: bool = False):
    """
    Creates and saves a FAISS vector store from the chunked XML documents.
    Handles rate limits and preserves documents for further analysis.
    """
    # Check for API key before making any calls
    if not os.getenv('AZURE_OPENAI_API_KEY'):
        logger.error("AZURE_OPENAI_API_KEY environment variable not set.")
        logger.error("Please set your Azure OpenAI API key to proceed with embedding creation.")
        return

    # Load or create documents
    if resume_from_backup and os.path.exists(DOCUMENTS_BACKUP_PATH):
        documents = load_documents_backup(DOCUMENTS_BACKUP_PATH)
    else:
        documents = chunk_informatica_xml(XML_FILE_PATH)
        if not documents:
            logger.error("No documents were generated. Halting vector store creation.")
            return
        
        # Save documents backup for preservation
        save_documents_backup(documents, DOCUMENTS_BACKUP_PATH)

    logger.info("Initializing embeddings model...")
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_version="2024-05-01-preview"
    )

    # Check if we should resume from previous progress
    progress = load_progress(PROGRESS_FILE)
    start_index = progress.get('processed', 0)
    
    if start_index > 0:
        logger.info(f"Resuming from document {start_index} (Progress: {progress['percentage']:.1f}%)")
    
    logger.info("Creating FAISS vector store from documents with rate limit handling...")
    vector_store = create_embeddings_with_rate_limit(documents, embeddings, start_index)

    if vector_store:
        logger.info(f"Saving final vector store to: {VECTOR_STORE_PATH}")
        vector_store.save_local(VECTOR_STORE_PATH)
        logger.info("Vector store created successfully.")
        
        # Clean up temporary files
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        
        # Keep the document backup for further analysis
        logger.info(f"Document backup preserved at: {DOCUMENTS_BACKUP_PATH}")
        logger.info("You can use this backup for further analysis without re-parsing the XML.")
    else:
        logger.error("Failed to create vector store.")

def analyze_documents_from_backup():
    """
    Example function showing how to analyze documents from backup
    without needing to recreate embeddings.
    """
    if not os.path.exists(DOCUMENTS_BACKUP_PATH):
        logger.error("No document backup found. Please run create_vector_store() first.")
        return
    
    documents = load_documents_backup(DOCUMENTS_BACKUP_PATH)
    
    # Example analysis
    logger.info("=== Document Analysis ===")
    logger.info(f"Total documents: {len(documents)}")
    
    # Analyze by object type
    type_counts = {}
    for doc in documents:
        obj_type = doc.metadata.get('tag_name', 'unknown')
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
    
    logger.info("Object type distribution:")
    for obj_type, count in sorted(type_counts.items()):
        logger.info(f"  {obj_type}: {count}")
    
    # Analyze by folder
    folder_counts = {}
    for doc in documents:
        folder = doc.metadata.get('folder_name', 'unknown')
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    
    logger.info("Folder distribution:")
    for folder, count in sorted(folder_counts.items()):
        logger.info(f"  {folder}: {count}")
    
    return documents

def indexer():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'resume':
            create_vector_store(resume_from_backup=True)
        elif sys.argv[1] == 'analyze':
            analyze_documents_from_backup()
    else:
        create_vector_store()

if __name__ == '__main__':
    indexer()
    
    