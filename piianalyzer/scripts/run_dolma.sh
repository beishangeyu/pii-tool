export CLASSPATH='/mnt/mingd3/PII_detect/piianalyzer/stanford-ner.jar'
python run.py \
    --dataset_name dolma \
    --data_path '../data/dolma/v1_7' \
    --batch_size 60 \
    --workers 48 \
    --max_tasks_per_child 5