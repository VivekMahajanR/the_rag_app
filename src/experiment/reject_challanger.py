import dagshub
import mlflow

from utils.mlflow_utils import CHALLENGER, get_run_by_stage, reject_challenger

# initialize dagshub and mlflow
dagshub.init(repo_owner='VivekMahajanR', repo_name='the_rag_app', mlflow=True)
    
# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/VivekMahajanR/the_rag_app.mlflow")

# fetch the experiment id
experiment_id = mlflow.get_experiment_by_name("rag_app").experiment_id

challenger = get_run_by_stage(CHALLENGER, experiment_id)
print(f"challanger : {challenger.info.run_id} ({challenger.info.run_name})")

reject_challenger(experiment_id)
print(f"cleared staged tag from {challenger.info.run_id}")