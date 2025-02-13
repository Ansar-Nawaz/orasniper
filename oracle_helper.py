# oracle_helper.py

def resolve_oracle_issue(query):
    """
    Check if the user query relates to Oracle database issues.
    If yes, perform a search over the Oracle documentation and return a formatted response.
    """
    # Check for keywords that indicate Oracle-related queries
    if any(keyword in query.lower() for keyword in ["oracle", "rman", "asm", "database"]):
        results = oracle_doc_search(query)
        if results:
            return format_oracle_results(results)
        else:
            return ("I couldn't find any relevant Oracle documentation for your query. "
                    "Please refer to Oracle's official support resources for more details.")
    # If the query does not pertain to Oracle, return None to allow fallback
    return None

def oracle_doc_search(query):
    """
    Perform a search over a pre-indexed set of Oracle documentation.
    For demonstration purposes, this function uses a static list.
    
    In a production environment, you might integrate with a search engine (e.g., Elasticsearch)
    or use an API to retrieve relevant documentation.
    """
    # Dummy documentation repository: In practice, replace with real document retrieval
    oracle_docs = [
        {
            "title": "RMAN Backup Warning",
            "snippet": "RMAN-06820: warning: failed to archive current log at primary database. "
                       "Ensure that your standby database is properly configured."
        },
        {
            "title": "ASM Space Management",
            "snippet": "Check ASM space using V$ASM_DISKGROUP and ASMCMD 'lsdg' command. Ensure "
                       "sufficient space is available for database file creation."
        },
        {
            "title": "Oracle Database Connection Issues",
            "snippet": "Verify your TNS entries and network connectivity when encountering 'cannot connect to remote database'."
        }
    ]
    
    # A simple filter: return docs where the query matches the title or snippet text
    matching_docs = [doc for doc in oracle_docs if query.lower() in doc["title"].lower() or query.lower() in doc["snippet"].lower()]
    return matching_docs

def format_oracle_results(results):
    """
    Format the Oracle documentation search results for chat output.
    """
    formatted = "\n\n".join([f"**{doc['title']}**\n{doc['snippet']}" for doc in results])
    return formatted if formatted else "No matching Oracle documentation found."
