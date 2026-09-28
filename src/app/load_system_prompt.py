from langfuse import get_client
from dotenv import load_dotenv

# load the API key
load_dotenv()

langfuse = get_client()

system_prompt = langfuse.get_prompt(
    name="the_rag_app_system_prompt",
    type="text",
    label="staging"
)

print(system_prompt.version)
print(system_prompt.prompt)
print(system_prompt.config)
print(system_prompt.labels)