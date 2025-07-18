#started at 11:08
import pandas as pd
import tiktoken
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any
import json
import logging
from dataclasses import dataclass
import time
from threading import Lock
from dotenv import load_dotenv
import re # Import the regular expression module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

@dataclass
class LineageChunk:
    """Represents a chunk of lineage data with metadata"""
    data: str
    tables: List[str]
    transformations: List[str]
    chunk_id: int
    token_count: int

class InformaticaLineageProcessor:
    def __init__(self, max_tokens_per_chunk: int = 15000, max_workers: int = 5):
        """
        Initialize the Informatica Lineage Processor
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk for processing (increased for efficiency)
            max_workers: Number of parallel threads for processing
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_workers = max_workers
        
        # Initialize OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=4096, # Adjusted for safety, as output tokens also count
            timeout=None,
            max_retries=2,
        )
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        # Thread lock for logging
        self.log_lock = Lock()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens_per_chunk * 3,
            chunk_overlap=200,
            separators=["\n", ",", " "]
        )
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    def load_lineage_csv(self, csv_path: str) -> pd.DataFrame:
        """Load the Informatica lineage CSV file"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def analyze_lineage_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the lineage structure to understand data flow"""
        analysis = {
            "total_rows": len(df),
            "unique_tables": {
                "source": df['FromInstance'].nunique(),
                "target": df['ToInstance'].nunique()
            },
            "transformation_types": df['FromInstanceType'].value_counts().to_dict(),
            "target_types": df['ToInstanceType'].value_counts().to_dict(),
            "fields_mapping": len(df[['FromField', 'ToField']].drop_duplicates())
        }
        return analysis
    
    def create_larger_logical_chunks(self, df: pd.DataFrame) -> List[LineageChunk]:
        """Create larger logical chunks to reduce API calls"""
        chunks = []
        
        # Group by target table to maintain logical coherence
        target_groups = df.groupby('ToInstance')
        
        current_chunk_data = []
        current_chunk_tables = set()
        current_chunk_transformations = set()
        current_token_count = 0
        chunk_id = 0
        
        for target_table, group in target_groups:
            # Convert group to CSV string
            group_csv = group.to_csv(index=False)
            group_tokens = self.count_tokens(group_csv)
            
            # If this group alone exceeds max tokens, split it further
            if group_tokens > self.max_tokens_per_chunk:
                # Save current chunk if it has data
                if current_chunk_data:
                    chunk = self._create_chunk(current_chunk_data, current_chunk_tables, 
                                               current_chunk_transformations, chunk_id, current_token_count)
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk_data = []
                    current_chunk_tables = set()
                    current_chunk_transformations = set()
                    current_token_count = 0
                
                # Split large group into smaller chunks
                sub_chunks = self._split_large_group(group, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                
            # If adding this group would exceed limit, save current chunk
            elif current_token_count + group_tokens > self.max_tokens_per_chunk:
                if current_chunk_data:
                    chunk = self._create_chunk(current_chunk_data, current_chunk_tables, 
                                               current_chunk_transformations, chunk_id, current_token_count)
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with this group
                current_chunk_data = [group_csv]
                current_chunk_tables = {target_table}
                current_chunk_transformations = set(group['FromInstanceType'].unique())
                current_token_count = group_tokens
                
            else:
                # Add to current chunk
                current_chunk_data.append(group_csv)
                current_chunk_tables.add(target_table)
                current_chunk_transformations.update(group['FromInstanceType'].unique())
                current_token_count += group_tokens
        
        # Don't forget the last chunk
        if current_chunk_data:
            chunk = self._create_chunk(current_chunk_data, current_chunk_tables, 
                                       current_chunk_transformations, chunk_id, current_token_count)
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} logical chunks (optimized for larger size)")
        return chunks
    
    def _create_chunk(self, data_list: List[str], tables: set, transformations: set, 
                      chunk_id: int, token_count: int) -> LineageChunk:
        """Create a LineageChunk object"""
        # Combine all CSV data with proper header
        if data_list:
            header = "FromField,FromInstance,FromInstanceType,ToField,ToInstance,ToInstanceType,Transformation\n"
            combined_data = header + "\n".join([d.split('\n', 1)[1] for d in data_list if '\n' in d])
        else:
            combined_data = ""
            
        return LineageChunk(
            data=combined_data,
            tables=list(tables),
            transformations=list(transformations),
            chunk_id=chunk_id,
            token_count=token_count
        )
    
    def _split_large_group(self, group: pd.DataFrame, start_chunk_id: int) -> List[LineageChunk]:
        """Split a large group that exceeds token limit"""
        chunks = []
        chunk_id = start_chunk_id
        
        # Split by rows if group is too large
        rows_per_chunk = max(1, len(group) // ((self.count_tokens(group.to_csv(index=False)) // self.max_tokens_per_chunk) + 1))
        
        for i in range(0, len(group), rows_per_chunk):
            chunk_df = group.iloc[i:i+rows_per_chunk]
            chunk_csv = chunk_df.to_csv(index=False)
            chunk_tokens = self.count_tokens(chunk_csv)
            
            chunk = LineageChunk(
                data=chunk_csv,
                tables=list(chunk_df['ToInstance'].unique()),
                transformations=list(chunk_df['FromInstanceType'].unique()),
                chunk_id=chunk_id,
                token_count=chunk_tokens
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def create_pyspark_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for generating PySpark code from lineage chunk"""
        
        prompt_template = """
You are an expert in converting Informatica lineage data to PySpark code. Your task is to generate only the Python code for the transformations described in the provided lineage data.

Context from previous chunks: {context}

Current lineage data (CSV format):
{lineage_data}

Tables involved in this chunk: {tables}
Transformation types: {transformations}

Please generate PySpark code that:
1. Reads the source dataframes (assume they are already loaded and named after the source tables).
2. Applies the transformations shown in the lineage.
3. Creates a new dataframe with the final transformed data.
4. Includes comments explaining the transformation logic.
5. Follows PySpark best practices.

IMPORTANT: Generate ONLY the PySpark code inside a ```python ... ``` block. Do not include any explanatory text outside of the code block.

Generate clean, production-ready PySpark code:
"""
        
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "lineage_data", "tables", "transformations"]
        )

    def _extract_python_code(self, llm_response: str) -> str:
        """
        Extracts Python code from the LLM's response using regex.
        It specifically looks for a Markdown code block (```python...```).
        
        Args:
            llm_response: The raw string response from the language model.
            
        Returns:
            The extracted Python code as a string, or an empty string if no block is found.
        """
        # Regex to find code within ```python ... ``` block
        # re.DOTALL allows '.' to match newline characters
        code_block_match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        
        if code_block_match:
            # Return the captured group (the code itself), stripped of leading/trailing whitespace
            return code_block_match.group(1).strip()
        else:
            # If no markdown block is found, it's safer to return an empty string or log a warning
            # than to return the whole response which might contain conversational text.
            logger.warning("No python markdown block found in LLM response for the chunk.")
            # We can return the raw response as a fallback, but it might pollute the final script
            return llm_response.strip()

    def process_chunk(self, chunk: LineageChunk, context: str = "") -> str:
        """Process a single chunk and generate PySpark code"""
        try:
            # Create LLMChain with proper prompt template
            prompt_template = self.create_pyspark_prompt_template()
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template
            )
            
            # Prepare input variables
            input_vars = {
                "context": context,
                "lineage_data": chunk.data,
                "tables": ", ".join(chunk.tables),
                "transformations": ", ".join(chunk.transformations)
            }
            
            # Generate PySpark code from the LLM
            raw_result = chain.run(input_vars)
            
            # Extract only the python code from the raw response
            pyspark_code = self._extract_python_code(raw_result)
            
            logger.info(f"Successfully processed chunk {chunk.chunk_id}")
            return pyspark_code
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            return f"# Error processing chunk {chunk.chunk_id}: {str(e)}"
    
    def process_all_chunks(self, csv_path: str, output_path: str = "generated_pyspark_code.py") -> str:
        """Process all chunks and generate complete PySpark code"""
        
        # Load and analyze data
        df = self.load_lineage_csv(csv_path)
        analysis = self.analyze_lineage_structure(df)
        
        logger.info(f"Lineage Analysis: {json.dumps(analysis, indent=2)}")
        
        # Create chunks
        chunks = self.create_larger_logical_chunks(df)
        
        # Process chunks
        generated_code_parts = []
        context = ""
        
        # Add header
        header = f'''
"""
Generated PySpark Code from Informatica Lineage
Auto-generated from {os.path.basename(csv_path)}
Total chunks processed: {len(chunks)}
Analysis: {json.dumps(analysis, indent=2)}
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

# Initialize Spark Session
spark = SparkSession.builder \\
    .appName("InformaticaLineageConversion") \\
    .getOrCreate()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# This script is auto-generated. It assumes that source dataframes are already
# loaded into memory. You will need to add the data loading logic for your
# source tables (e.g., from CSV, Parquet, JDBC, etc.) before this script runs.
# ==============================================================================

'''
        generated_code_parts.append(header)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (ID: {chunk.chunk_id})")
            
            # Generate code for this chunk
            chunk_code = self.process_chunk(chunk, context)
            
            # Add chunk separator
            chunk_header = f"\n# ===== CHUNK {chunk.chunk_id} - Target Tables: {', '.join(chunk.tables)} =====\n"
            generated_code_parts.append(chunk_header)
            generated_code_parts.append(chunk_code)
            
            # Update context with summary for next chunk
            context = f"Previous chunk processed tables: {', '.join(chunk.tables)}. The resulting dataframes are now available."
        
        # Add footer
        footer = '''

# ==============================================================================
# End of auto-generated code. You may need to add logic here to write the
# final dataframes to their target destinations.
# ==============================================================================

# Stop Spark session
logger.info("Stopping Spark session.")
spark.stop()
'''
        generated_code_parts.append(footer)
        
        # Combine all parts
        complete_code = "\n".join(generated_code_parts)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(complete_code)
        
        logger.info(f"Complete PySpark code saved to {output_path}")
        return complete_code

# Usage example
def code_generator():
    # Configuration
    CSV_PATH = "Event_fact_new_rough_lineage.csv"      # Path to your CSV file
    OUTPUT_PATH = "generated_pyspark_code.py"
    
    # Initialize processor
    processor = InformaticaLineageProcessor(max_tokens_per_chunk=15000)
    
    # Process the lineage file
    try:
        generated_code = processor.process_all_chunks(CSV_PATH, OUTPUT_PATH)
        print(f"Successfully generated PySpark code! Check {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        logger.exception("Top-level error")

if __name__ == "__main__":
    code_generator()
