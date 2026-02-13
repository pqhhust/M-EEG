from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.train import RunConfig, ScalingConfig
from hyperparameter_optimization import eval
import numpy as np

# Search space (giữ nguyên)
search_space = {
    "lr_log": tune.uniform(-13.8155, -7.6009),
    "--dropout": tune.uniform(0, 0.2),
    "weight_decay_log": tune.uniform(-13.8155, -2.9957),
    "--clip_value": tune.uniform(0.5, 2.0)
}

# ASHA Scheduler (giữ nguyên config)
asha = ASHAScheduler(
    time_attr="epoch",
    max_t=50,
    grace_period=5,
    reduction_factor=2
)

# Đọc kết quả từ experiment cũ (50 trials)
print("Loading previous experiment results...")
old_experiment_path = "/home/aiotlab/ray_results/finetune_50e_bo_asha"
old_analysis = tune.ExperimentAnalysis(old_experiment_path)

print(f"Found {len(old_analysis.trials)} previous trials")

# Lấy dữ liệu từ trials cũ để warm-start
previous_configs = []
previous_scores = []

for trial in old_analysis.trials:
    if trial.last_result and "kappa" in trial.last_result:
        config = trial.config
        score = trial.last_result["kappa"]
        previous_configs.append(config)
        previous_scores.append(score)

# Tạo BayesOpt với points_to_evaluate (warm-start)
# Lấy top 10 configs tốt nhất để khởi tạo BayesOpt
sorted_indices = np.argsort(previous_scores)  # min mode
top_configs = [previous_configs[i] for i in sorted_indices[:10]]

bo_warmstart = BayesOptSearch(
    space=search_space,  # Phải specify space khi dùng points_to_evaluate
    metric="kappa",
    mode="max",
    points_to_evaluate=top_configs,  # Warm-start với top configs
    random_search_steps=8,
    random_state=42
)

# Tạo experiment MỚI với số trials cao hơn
tuner = Tuner(
    trainable=tune.with_resources(eval, {"gpu": 1, "cpu": 4}),
    tune_config=TuneConfig(
        metric="kappa", mode="max",
        scheduler=asha,
        search_alg=bo_warmstart,  # BayesOpt đã được warm-start
        num_samples=200,  # Tăng lên 300 trials mới
    )
)

# Chạy experiment MỚI
results = tuner.fit()