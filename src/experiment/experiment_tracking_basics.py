import mlflow
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# set the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# set the experiment name
mlflow.set_experiment("the_rag_app")

# start a run
with mlflow.start_run(run_name="test_run") as run:
    # params
    mlflow.log_param("chunk_size", 300)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("output_dim", 1024)
    # metrics
    mlflow.log_metric("recall", 0.93)
    mlflow.log_metric("precision", 0.87)
    mlflow.log_metric("contextual_relevance", 0.98)
    mlflow.log_metric("faithfulness", 0.99)
    mlflow.log_metric("answer_relevalce", 0.67)
    # artifacts
    mlflow.log_artifact(local_path=(ROOT_DIR / "src" / "app" / "rag_workflow.py"),
                        artifact_path="code")

    # log dataset
    mlflow.log_artifact(local_path=(ROOT_DIR / "data" / "evaluation" / "goldens" / "golden_dataset").with_suffix(".json"))    
