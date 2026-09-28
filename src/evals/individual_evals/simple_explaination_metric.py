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
simple_explaination = GEval(model=deepseek_model,
    name = "simple_explaination",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read the 'actual output' first and then the 'input'",
        "Check whether the response is simple and easy to understand or not",
        "Make sure the response has least number od technical jargons and is student friendly",
    ],
    rubric=[
        Rubric(score_range=(0, 3), expected_outcome="Too Difficult"),
        Rubric(score_range=(4, 7), expected_outcome="Slightly difficult"),
        Rubric(score_range=(8, 9), expected_outcome="moderetely simple"),
        Rubric(score_range=(10, 10), expected_outcome="very simple")
    ],
    verbose_mode=True    
)

test_case = LLMTestCase(
    input="what is logistic regression?",
    actual_output="Logistic regression is a statistical and machine learning method used to predict the probability of a categorical outcome, usually limited to two choices like yes or no, 0 or 1"
)

simple_explaination.measure(test_case=test_case)