from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.models import OpenRouterModel
from dotenv import load_dotenv
import os

# load API key
load_dotenv()

deepseek_model = OpenRouterModel(
    model="deepseek/deepseek-v4-flash-0731",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

rag_input =  "How does the K-Nearest Neighbors (KNN) algorithm work?"


retrieval_context = [
    "KNN is a non-parametric algorithm that classifies a data point based on the majority class among its K nearest neighbors in the feature space. It requires no training phase, since the entire dataset is stored and used at prediction time. This makes prediction slow for large datasets, since distance must be computed to every point.",
    "Choosing the right value of K is important — small K makes the model sensitive to noise, while large K smooths the decision boundary but may include points from other classes.",
    "Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. KNN is often compared to SVMs as both are used for classification, but SVMs are parametric and build an explicit decision boundary during training, unlike KNN's lazy learning approach."
    ]

test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    actual_output=""
)

contextual_relevancy = ContextualRelevancyMetric(model= deepseek_model, verbose_mode=True)

contextual_relevancy.measure(test_case=test_case)
