from langfuse import get_client
from dotenv import load_dotenv

# load API key
load_dotenv()

# shifting the labels in development
langfuse = get_client()

langfuse.update_prompt(
    name="the_rag_app_system_prompt",
    version=None,
    new_labels= ["staging"]
)