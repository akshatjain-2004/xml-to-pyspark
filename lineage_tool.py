from indexer import indexer
# Chain is correct (totally correct)
import os
from lxml import etree
from dotenv import load_dotenv
import logging
import re

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
# --- Configuration ---
VECTOR_STORE_PATH = 'informatica_vector_store'
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "default_key")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")


# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global retriever to be initialized once
retriever = None
trans = []
flag = 0
lookup = []
lookup_inst = []

def retrieve_all_documents():
    """
    Loads a FAISS vector store and retrieves all stored documents.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        logger.error(f"Vector store not found at '{VECTOR_STORE_PATH}'.")
        logger.error("Please run the 'indexer1.py' script first to create the vector store.")
        return None

    logger.info("Initializing the embeddings model...")
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_version="2024-05-01-preview"
    )

    logger.info(f"Loading vector store from: {VECTOR_STORE_PATH}")
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None

    # Access the docstore and retrieve all documents
    all_documents = [vector_store.docstore.search(id) for id in vector_store.index_to_docstore_id.values()]
    
    return all_documents

def extract_lkp_words(expression):
    # Case-insensitive match: (?i) makes 'lkp' match LKP, lkp, Lkp, etc.
    lkp_words = re.findall(r'(?i)LKP\w*', expression)
    return list(dict.fromkeys(lkp_words))

def clean_csv_field(field_value):
    """Clean field values to prevent CSV parsing issues."""
    if field_value is None:
        return "N/A"
    
    # Convert to string and handle various data types
    field_str = str(field_value).strip()
    
    # Replace problematic characters
    field_str = field_str.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    field_str = re.sub(r'\s+', ' ', field_str)  # Replace multiple spaces with single space
    
    # Escape quotes by doubling them (CSV standard)
    field_str = field_str.replace('"', '""')
    
    # If field contains comma, quote, or newline, wrap in quotes
    if ',' in field_str or '"' in field_str or '\n' in field_str:
        field_str = f'"{field_str}"'
    
    return field_str

def get_retriever():
    """Initializes and returns the vector store retriever."""
    global retriever
    if retriever is None:
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at '{VECTOR_STORE_PATH}'. "
                "Please run the 'indexer.py' script first."
            )
        print("Loading vector store...")
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_deployment="text-embedding-3-small",
            openai_api_version="2024-05-01-preview"
        )
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def find_all_target_tables() -> list:
    """
    Discovers all target table names in the vector store.
    
    Returns:
        List of target table names found in the system.
    """
    print("find_all_target_tables()")
    docs = get_retriever().invoke("target definition")
    
    target_tables = []
    for doc in docs:
        if doc.metadata.get('tag_name') == 'TARGET':
            object_name = doc.metadata.get('object_name')
            if object_name and object_name not in target_tables:
                target_tables.append(object_name)
    
    print(f"Found {len(target_tables)} target table(s): {target_tables}")
    return target_tables

# print(find_all_target_tables())

def find_target_fields(target_name: str) -> list:
    """
    Finds all field names for a given target table definition.
    This is the starting point for the lineage trace.
    Args:
        target_name (str): The exact name of the target object to search for.
                          Pass this as a simple string value, not as a dictionary.
                          Example: "W_WRKFC_EVENT_TYPE_D"
    
    Returns:
        List of field information for the specified target.
    """
    print(f"find_target_fields(target_name: {target_name})")
    all_docs = retrieve_all_documents()
    
    if all_docs:
        logger.info(f"Retrieved {len(all_docs)} total documents. Now filtering...")

        # 2. Use a list comprehension to filter based on metadata
        # This checks for documents where the tag is 'TARGET' and the name is 'W_WRKFC_EVT_F'
        filtered_docs = [
            doc for doc in all_docs 
            if doc.metadata.get('tag_name') == 'TARGET' and 
               doc.metadata.get('object_name') == target_name
        ]
    for doc in filtered_docs:
        if doc.metadata.get('object_name') == target_name and doc.metadata.get('tag_name') == 'TARGET':
            root = etree.fromstring(doc.page_content.encode('utf-8'))
            fields = [field.get('NAME') for field in root.findall('.//TARGETFIELD')]
            return fields
    return []
# print(find_target_fields("W_WRKFC_EVENT_TYPE_D"))

def trace_field_backward_with_transformation_func(to_instance: str, to_field: str) -> dict:
    """
    Traces a single field backward one step and extracts its transformation logic.
    Given a destination instance and field, it finds the source instance and field that connect to it,
    and also retrieves the transformation logic if applicable.
    """
    global lookup
    global lookup_inst
    global trans
    global flag
    flag = 0
    print(f"Executing trace for: {to_instance}.{to_field}")
    if(to_instance[:4] == "mplt"):
        flag = 1
        query = f"connector to instance {to_instance}"
        trans.append(to_instance)
    elif (to_instance == "Input"):
        to_instance = trans[-1]
        query = f"connector to instance {to_instance} field {to_field}"
    else:
        # 1. Find the mapping or mapplet that contains the connection logic.
        query = f"connector to instance {to_instance} field {to_field}"
    docs = get_retriever().invoke(query)
    # with open("output.txt", "w") as file:
    #     file.write(str(docs[0]))
    for doc in docs:
        if doc.metadata.get('tag_name') not in ['MAPPING', 'MAPPLET']:
            continue

        try:
            root = etree.fromstring(doc.page_content.encode('utf-8'))
        except etree.XMLSyntaxError:
            continue

        # 2. Find the specific connector within the document.
        if(flag == 1):
            xpath_query_connector = f".//CONNECTOR[@TOFIELD='{to_field}']"
        else:
            xpath_query_connector = f".//CONNECTOR[@TOINSTANCE='{to_instance}' and @TOFIELD='{to_field}']"
        connectors = root.xpath(xpath_query_connector)

        if not connectors:
            continue

        connector = connectors[0]
        from_instance_name = connector.get("FROMINSTANCE")
        from_field_name = connector.get("FROMFIELD")
        from_instance_type = connector.get("FROMINSTANCETYPE")
        to_instance_type = connector.get("TOINSTANCETYPE")

        transformation_logic = ""  # Default if no specific logic is found

        # 3. Find the transformation logic if the source instance is a transformation.
        if from_instance_type == "Expression":
            print("nice")
            # Find the INSTANCE tag to determine if it's reusable and get the definition name.
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TRANSFORMFIELD[@NAME='{from_field_name}']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("EXPRESSION")
                while( len(transformation_logic) >= 2 and transformation_logic[:2] == "V_"):
                    xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TRANSFORMFIELD[@NAME='{transformation_logic}']"
                    instances = root.xpath(xpath_query_instance)
                    if instances:
                        transformation_logic = instances[0].get("EXPRESSION")
                        lookup = extract_lkp_words(transformation_logic)
                        if len(lookup) > 0:
                            lookup_inst.append(from_instance_name)


        elif from_instance_type == "Filter":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Filter Condition']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("VALUE")
        
        elif from_instance_type == "Update Strategy":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Update Strategy Expression']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("VALUE")
        
        elif from_instance_type == "Source Qualifier":
            transformation_logic = "Sql Query: ; User Defined Join: ; Source Filter: ; Number Of Sorted Ports: 0; Tracing Level: Normal; Select Distinct: NO; Is Partitionable: NO; Pre SQL: ; Post SQL: ; Output is deterministic: NO; Output is repeatable: Never"
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Sql Query']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = "Sql Query: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Source Filter']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Source Filter: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Number Of Sorted Ports']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Number Of Sorted Ports: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Tracing Level']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Tracing Level: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Select Distinct']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Select Distinct: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Is Partitionable']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Is Partitionable: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Pre SQL']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Pre SQL: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Post SQL']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Post SQL: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Output is deterministic']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Output is deterministic: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Output is repeatable']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Output is repeatable: " + instances[0].get("VALUE")
        
        elif from_instance_type == "Lookup Procedure":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Sql Override']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = "Lookup Sql Override: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup table name']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup table name: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Source Filter']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup Source Filter: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Condition']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup Condition: " + instances[0].get("VALUE")

        # Construct the result dictionary
        return {
            "FromField": from_field_name,
            "FromInstance": from_instance_name,
            "FromInstanceType": from_instance_type,
            "Transformation": transformation_logic.strip().replace('\r\n', ' ').replace('\n', ' '),
            "ToField": to_field,
            "ToInstance": to_instance,
            "ToInstanceType": to_instance_type
        }

    return {"error": f"Could not find a valid connection for {to_instance}.{to_field}"}


# print(trace_field_backward_with_transformation_func("W_WRKFC_EVENT_TYPE_D", "ROW_WID"))

def lookup_trace(to_instance: str, to_field: str) -> dict:
    """
    Traces a single field backward one step and extracts its transformation logic.
    Given a destination instance and field, it finds the source instance and field that connect to it,
    and also retrieves the transformation logic if applicable.
    """
    global trans
    global flag
    flag = 0
    print(f"Executing trace for: {to_instance}.{to_field}")
    if(to_instance[:4] == "mplt"):
        flag = 1
        query = f"connector to instance {to_instance}"
        trans.append(to_instance)
    elif (to_instance == "Input"):
        to_instance = trans[-1]
        query = f"connector to instance {to_instance} field {to_field}"
    else:
        # 1. Find the mapping or mapplet that contains the connection logic.
        query = f"connector to instance {to_instance} field {to_field}"
    docs = get_retriever().invoke(query)
    # with open("output.txt", "w") as file:
    #     file.write(str(docs[0]))
    for doc in docs:
        if doc.metadata.get('tag_name') not in ['MAPPING', 'MAPPLET']:
            continue

        try:
            root = etree.fromstring(doc.page_content.encode('utf-8'))
        except etree.XMLSyntaxError:
            continue

        # 2. Find the specific connector within the document.
        if(flag == 1):
            xpath_query_connector = f".//CONNECTOR[@TOFIELD='{to_field}']"
        else:
            xpath_query_connector = f".//CONNECTOR[@TOINSTANCE='{to_instance}' and @TOFIELD='{to_field}']"
        connectors = root.xpath(xpath_query_connector)

        if not connectors:
            continue

        connector = connectors[0]
        from_instance_name = connector.get("FROMINSTANCE")
        from_field_name = connector.get("FROMFIELD")
        from_instance_type = connector.get("FROMINSTANCETYPE")
        to_instance_type = connector.get("TOINSTANCETYPE")

        transformation_logic = ""  # Default if no specific logic is found

        # 3. Find the transformation logic if the source instance is a transformation.
        if from_instance_type == "Expression":
            print("nice")
            # Find the INSTANCE tag to determine if it's reusable and get the definition name.
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TRANSFORMFIELD[@NAME='{from_field_name}']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("EXPRESSION")
                while( len(transformation_logic) >= 2 and transformation_logic[:2] == "V_"):
                    xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TRANSFORMFIELD[@NAME='{transformation_logic}']"
                    instances = root.xpath(xpath_query_instance)
                    if instances:
                        transformation_logic = instances[0].get("EXPRESSION")

        elif from_instance_type == "Filter":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Filter Condition']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("VALUE")
        
        elif from_instance_type == "Update Strategy":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Update Strategy Expression']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = instances[0].get("VALUE")
        
        elif from_instance_type == "Source Qualifier":
            transformation_logic = "Sql Query: ; User Defined Join: ; Source Filter: ; Number Of Sorted Ports: 0; Tracing Level: Normal; Select Distinct: NO; Is Partitionable: NO; Pre SQL: ; Post SQL: ; Output is deterministic: NO; Output is repeatable: Never"
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Sql Query']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = "Sql Query: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Source Filter']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Source Filter: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Number Of Sorted Ports']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Number Of Sorted Ports: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Tracing Level']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Tracing Level: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Select Distinct']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Select Distinct: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Is Partitionable']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Is Partitionable: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Pre SQL']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Pre SQL: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Post SQL']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Post SQL: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Output is deterministic']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Output is deterministic: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Output is repeatable']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Output is repeatable: " + instances[0].get("VALUE")
        
        elif from_instance_type == "Lookup Procedure":
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Sql Override']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = "Lookup Sql Override: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup table name']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup table name: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Source Filter']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup Source Filter: " + instances[0].get("VALUE")
            xpath_query_instance = f".//TRANSFORMATION[@NAME='{from_instance_name}']/TABLEATTRIBUTE[@NAME='Lookup Condition']"
            instances = root.xpath(xpath_query_instance)
            if instances:
                transformation_logic = transformation_logic + "; Lookup Condition: " + instances[0].get("VALUE")

        # Construct the result dictionary
        return {
            "FromField": from_field_name,
            "FromInstance": from_instance_name,
            "FromInstanceType": from_instance_type,
            "Transformation": transformation_logic.strip().replace('\r\n', ' ').replace('\n', ' '),
            "ToField": to_field,
            "ToInstance": to_instance,
            "ToInstanceType": to_instance_type
        }

    return {"error": f"Could not find a valid connection for {to_instance}.{to_field}"}

def lineage():
    """Generates the lineage document in the form of a csv file"""
    # indexer()
    global flag
    global lookup
    global lookup_inst
    ans = f"FromField,FromInstance,FromInstanceType,ToField,ToInstance,ToInstanceType,Transformation"
    trgt_tables = find_all_target_tables()
    i = 1
    for tbl in trgt_tables:
        print(f"Processing Table {i}")
        fields = find_target_fields(tbl)
        total = len(fields)
        print(fields)
        for field in fields:
            print(f"Processing Field {i}/{total} of Table: {tbl}")
            i = i + 1
            trace = trace_field_backward_with_transformation_func(tbl, field)
            if 'error' not in trace:
                if flag == 1:
                    ans += f'\n{trace['FromField']},{trace['FromInstance']},{trace['FromInstanceType']},{trace['ToField']},OUTPUT,{trace['ToInstanceType']},"{trace['Transformation']}"'
                else:
                    ans += f'\n{trace['FromField']},{trace['FromInstance']},{trace['FromInstanceType']},{trace['ToField']},{trace['ToInstance']},{trace['ToInstanceType']},"{trace['Transformation']}"'
                if len(lookup) > 0:
                    for j in lookup:
                        trace_l = lookup_trace(lookup_inst[0], j)
                        if 'error' not in trace_l:
                            if flag == 1:
                                ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},OUTPUT,{trace_l['ToInstanceType']},"{trace['Transformation']}"'
                            else:
                                ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},{trace_l['ToInstance']},{trace_l['ToInstanceType']},"{trace_l['Transformation']}"'
                            trace_l = lookup_trace(trace_l['FromInstance'], "IN_DATASOURCE_NUM_ID")
                            if 'error' not in trace_l:
                                if flag == 1:
                                    ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},OUTPUT,{trace_l['ToInstanceType']},"{trace['Transformation']}"'
                                else:
                                    ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},{trace_l['ToInstance']},{trace_l['ToInstanceType']},"{trace_l['Transformation']}"'
                                while('error' not in trace_l):
                                    trace_l = lookup_trace(trace_l["FromInstance"], trace_l["FromField"])
                                    if 'error' not in trace_l:
                                        if flag == 1:
                                            ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},OUTPUT,{trace_l['ToInstanceType']},"{trace_l['Transformation']}"'
                                        else:
                                            ans += f'\n{trace_l['FromField']},{trace_l['FromInstance']},{trace_l['FromInstanceType']},{trace_l['ToField']},{trace_l['ToInstance']},{trace_l['ToInstanceType']},"{trace_l['Transformation']}"'
                lookup = []
                lookup_inst = []
                while('error' not in trace):
                    trace = trace_field_backward_with_transformation_func(trace["FromInstance"], trace["FromField"])
                    if 'error' not in trace:
                        if flag == 1:
                            ans += f'\n{trace['FromField']},{trace['FromInstance']},{trace['FromInstanceType']},{trace['ToField']},OUTPUT,{trace['ToInstanceType']},"{trace['Transformation']}"'
                        else:
                            ans += f'\n{trace['FromField']},{trace['FromInstance']},{trace['FromInstanceType']},{trace['ToField']},{trace['ToInstance']},{trace['ToInstanceType']},"{trace['Transformation']}"'

            else:
                continue
    with open("new_rough_lineage1.csv","w") as file:
        file.write(ans)
        print("Files saved")
    return ans

if __name__ == '__main__':
    lineage()
    # print(trace_field_backward_with_transformation_func("LKP_EventType","MASTER_VALUE"))