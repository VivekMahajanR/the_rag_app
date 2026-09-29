from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric
)
from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.models import OpenRouterModel
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from pathlib import Path
from dotenv import load_dotenv
import os

# load the api key
load_dotenv()

deepseek_model = OpenRouterModel(
    model="deepseek/deepseek-v4-flash-0731",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# define the metrics
recall = ContextualRecallMetric(model=deepseek_model)
precision = ContextualPrecisionMetric(model=deepseek_model)
contextual_relevacy = ContextualRelevancyMetric(model=deepseek_model)
answer_relevancy = AnswerRelevancyMetric(model=deepseek_model)
faithfulness = FaithfulnessMetric(model=deepseek_model,
                                  truths_extraction_limit=5,
                                  penalize_ambiguous_claims=True)

# define the custom metrics
answer_correctness = GEval(
    model=deepseek_model,
    name = "answer_correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0, 5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has miner differences"),
            Rubric(score_range=(10, 10), expected_outcome=r"100% correct")]
)

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
    ]   
)

# define the evaluation dataset path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_PATH = (ROOT_DIR / "data" / "evaluation" / "eval_dataset" / "evaluation_dataset").with_suffix(".json")

if DATASET_PATH.exists():
    # load the dataset
    dataset = EvaluationDataset()

    # load the test cases
    dataset.add_goldens_from_json_file(
        file_path=DATASET_PATH,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        retrieval_context_key_name="retrieval_context"
    )

    # store the test cases in a list
    # test_cases = dataset.test_cases
    
    # Convert goldens into LLMTestCase instances
    test_cases = []
    for golden in dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.actual_output,
            expected_output=golden.expected_output,
            retrieval_context=golden.retrieval_context
        )
        test_cases.append(test_case)

    # evaluate the dataset
    evaluate(test_cases=test_cases,
             metrics=[
                 recall,
                 precision,
                 answer_relevancy,
                 faithfulness,
                 contextual_relevacy,
                 answer_correctness,
                 simple_explaination
             ], 
             async_config=AsyncConfig(throttle_value=3, max_concurrent=5),
             display_config=DisplayConfig(results_folder=(ROOT_DIR / "reports" / "evaluation_results").as_posix(),
                                          file_type="md",
                                          file_output_dir=(ROOT_DIR / "reports" / "evaluation_report").as_posix())
    )
