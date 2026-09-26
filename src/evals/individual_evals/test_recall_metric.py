from deepeval.metrics import ContextualRecallMetric
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

# rag input
rag_input = "How does bagging work?"

# expected output
expected_output = "Bagging (Bootstrap Aggregating) trains multiple models on different random samples of the training data, drawn with replacement (bootstrap samples). Each model learns independently, then their predictions are combined—majority vote for classification or average for regression. This reduces variance and overfitting, especially with high-variance models like decision trees. Random Forest is a popular bagging extension that also randomly samples features."

# retrieval information
retriecal_context = [
    "Bagging creates multiple bootstrap samples from the training dataset — each sample is drawn randomly with replacement, meaning some data points may appear multiple times while others may be left out.",
    "In bagging, an independent model is trained on each bootstrap sample. For regression tasks, the final output is the average of all individual model predictions.",
    "Random forests are an extension of bagging that also introduces random feature selection at each split, in addition to bootstrap sampling."
    ]

# test case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retriecal_context,
    expected_output=expected_output
)

# metric
recall_metric = ContextualRecallMetric(verbose_mode=True,
                                       model=deepseek_model)

# test the metric
recall_metric.measure(test_case=test_case)