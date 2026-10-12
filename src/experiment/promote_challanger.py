import dagshub
import mlflow

from utils.mlflow_utils import (
    CHAMPION,
    CHALLENGER,
    get_run_by_stage,
    promote_challenger
)

# initialize dagshub and mlflow
dagshub.init(repo_owner='VivekMahajanR', repo_name='the_rag_app', mlflow=True)
    
# set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/VivekMahajanR/the_rag_app.mlflow")

# fetch the experiment id
experiment_id = mlflow.get_experiment_by_name("rag_app").experiment_id

champion = get_run_by_stage(CHAMPION, experiment_id)
challenger = get_run_by_stage(CHALLENGER, experiment_id)

print(f"current champion : {champion.info.run_id} ({champion.info.run_name})")
print(f"challenger       : {challenger.info.run_id} ({challenger.info.run_name})")

old_id, new_id = promote_challenger(experiment_id)
print(f"archived {old_id}")
print(f"promoted {new_id} -> {CHAMPION}")