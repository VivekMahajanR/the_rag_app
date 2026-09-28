from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.models import OpenRouterModel
from dotenv import load_dotenv
import os

# import api key
load_dotenv()

deepseek_model = OpenRouterModel(
    model="deepseek/deepseek-v4-flash-0731",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

rag_input =  "What is boosting in machine learning and how does it work?"

actual_ouput = """Boosting is an ensemble technique that combines multiple weak learners to form a strong learner. 
It trains models sequentially, where each new model focuses on correcting the errors made by the previous ones. 
AdaBoost was one of the first popular boosting algorithms. Random Forests, on the other hand, build trees independently and 
average their results. Gradient Boosting uses gradient descent to minimize a loss function when adding new learners."""

# test case
test_case = LLMTestCase(
    input=rag_input,
    actual_output=actual_ouput
)

# metric
answer_relevancy_metric = AnswerRelevancyMetric(verbose_mode=True,
                                       model=deepseek_model)

# test the metric
answer_relevancy_metric.measure(test_case=test_case)