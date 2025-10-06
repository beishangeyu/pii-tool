export CLASSPATH='./stanford-ner.jar'
python run.py \
    --dataset_name c4 \
    --data_path '../data/c4/en' \
    --batch_size 1000 \
    --workers 16 \
    --max_tasks_per_child 5