# use ray job submission for distributed training
uv run ray job submit -- python main.py base --algo.algorithm moetta &
uv run ray job submit -- python main.py base --algo.algorithm moetta --data.corruption potpourri &

# another way to run distributed training with ray, i.e, without --env.local flag
# uv run main.py base --algo.algorithm moetta --data.corruption potpourri

# 
# uv run main.py base --env.local --algo.algorithm moetta --data.corruption potpourri+
# uv run main.py convnext --env.local --algo.algorithm moetta
# uv run main.py vit_large --env.local --algo.algorithm moetta
