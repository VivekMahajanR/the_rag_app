from langfuse import get_client
from dotenv import load_dotenv

# load the api key
load_dotenv()

chunk_size = 300
chunk_overlap = 30
output_dimension = 1024
k = 3

system_prompt = """You are a helpful assistant. Answer the user query
based on the given context only. If you do not know the answer
say I don't know. Do not add any preamble to the response.
Always try to answer in simple language. make sure your answer sticks to the input and the available context,
and be factually correct"""

# add system prompt to langfuse

langfuse = get_client()

created_prompt = langfuse.create_prompt(
    name="the_rag_app_system_prompt",
    type="text",
    prompt=system_prompt,
    labels=["staging"],
    config={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_dims": output_dimension,
        "k": k
    }
)

print(created_prompt.prompt)
print(created_prompt.version)
print(created_prompt.labels)

# langfuse.update_prompt(
#     name="the_rag_app_system_prompt",
#     version=1,
#     new_labels=["production"]
# )