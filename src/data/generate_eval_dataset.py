from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from src.app.rag_workflow import graph
from time import sleep

# load api key
load_dotenv()

# create paths
ROOT_DIR = Path(__file__).parent.parent.parent
GOLDENS_PATH = (ROOT_DIR / "data" / "evaluation" / "goldens" / "golden_dataset_deepseek").with_suffix(".json")
EVALUATION_DATA_DIR = ROOT_DIR / "data" / "evaluation" / "eval_dataset"

# create directory
EVALUATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# evaluation dataset
dataset = EvaluationDataset()
# add goldens to dataset
dataset.add_goldens_from_json_file(file_path=GOLDENS_PATH)

# invoke our application
for golden in dataset.goldens:
    final_state = graph.invoke({"query": golden.input})
    sleep(3)
    test_case = LLMTestCase(
        input = golden.input,
        actual_output=final_state.get("response"),
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
    )
    dataset.add_test_case(test_case = test_case)

# save the dataset along with test cases
dataset.save_as(file_type="json",
                directory=EVALUATION_DATA_DIR,
                file_name="evaluation_dataset_deepseek",
                include_test_cases=True)            # if False goldens will be saved not test cases