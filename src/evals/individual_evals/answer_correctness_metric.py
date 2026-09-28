from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.models import OpenRouterModel
from dotenv import load_dotenv
import os

# import API keys
load_dotenv()

deepseek_model = OpenRouterModel(
    model="deepseek/deepseek-v4-flash-0731",
    api_key=os.environ["OPENROUTER_API_KEY"]
)


# define the custom metric
answer_correctness = GEval(
    model=deepseek_model,
    name = "answer_correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0, 5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has miner differences"),
            Rubric(score_range=(10, 10), expected_outcome=r"100% correct")],
    verbose_mode=True
)

test_case = LLMTestCase(
    input="",
    expected_output="Yes. According to the context, ARC-AGI-2 tests whether agents have human-like flexible reasoning in unfamiliar situations. Humans score 100%, while current AI models score only about 4–5%. This indicates that LLMs still lack the human-level cognitive flexibility needed for AGI.",
    actual_output="Yes. According to the context, ARC-AGI-2 tests whether agents have human-like flexible reasoning in unfamiliar situations. Humans score 100% while current models score only 4-5%, which implies current AI models lack that human-like flexible reasoning needed for AGI."
)

answer_correctness.measure(test_case=test_case)