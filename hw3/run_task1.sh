rm -rf task2_data.pt task2_model.pt preds2.csv

python prepare_task2.py --data_dir datasets/task2 --output task2_data.pt
bash train2.sh task2_data.pt task2_model.pt 
bash test2.sh task2_data.pt task2_model.pt preds2.csv                   
python eval_task2.py --pred preds2.csv                                  