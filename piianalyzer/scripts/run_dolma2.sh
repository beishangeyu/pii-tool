export CLASSPATH='/mnt/mingd3/PII_detect/piianalyzer/stanford-ner.jar'
python run.py \
    --dataset_name dolma_2 \
    --data_path '../data/dolma_2' \
    --batch_size 60 \
    --workers 12 \
    --max_tasks_per_child 5