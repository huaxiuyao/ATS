# ATS

## About
Source code of the paper [Meta-learning with an Adaptive Task Scheduler](https://arxiv.org/abs/xxx).


If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2021adaptive,
  title={Meta-learning with an Adaptive Task Scheduler},
  author={Yao, Huaxiu and Wang, Yu and Wei, Ying and Zhao, Peilin and Mahdavi, Mehrdad and Lian, Defu and Finn, Chelsea},
  booktitle={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems},
  year={2021} 
}
```


## Miniimagenet
The processed miniimagenet dataset could be downloaded [here](https://drive.google.com/file/d/10-l1GWesRIoToxFAO9_f2hAd6mVc9ns7/view?usp=sharing). Assume the dataset has been downloaded and unzipped to `/data/miniimagenet`, which has the following file structure:
```
-- miniimagenet  // /data/miniimagenet
  -- miniImagenet
    -- train_task_id.pkl
    -- test_task_id.pkl
    -- mini_imagenet_train.pkl
    -- mini_imagenet_test.pkl
    -- mini_imagenet_val.pkl
    -- training_classes_20000_2_new.npz
    -- training_classes_20000_4_new.npz
```
Then `$datadir` in the following code sould be set to `/data/miniimagenet`. 


### ATS with noise = 0.6
We need to first pretrain the model with no noise. The model has been uploaded to this repo. You can also pretrain the model by yourself. The script for pretraining is as follows:  
(1) 1 shot: 
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.0
```
(2) 5 shot:
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.0
```

Then move the model to the current directory:  
(1) 1 shot:
```
mv $logdir/ANIL_pytorch.data_miniimagenetcls_5.mbs_2.ubs_1.metalr0.001.innerlr0.01.hidden32/model20000 ./model20000_1shot
```
(2) 5 shot:
```
mv $logdir/ANIL_pytorch.data_miniimagenetcls_5.mbs_2.ubs_5.metalr0.001.innerlr0.01.hidden32/model10000 ./model10000_5shot
```

Then with this model, we could run the uniform sampling and ATS sampling. 
For ATS, the script is:  
(1) 1 shot
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0 --num_classes 5 --metatrain_iterations 30000 --replace 0 --noise 0.6 --logdir $logdir --sampling_method ATS --buffer_size 10  --temperature 0.1 --scheduler_lr 0.001 --warmup 2000 --pretrain_iter 20000
```
(2) 5 shot
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --replace 0 --noise 0.6 --logdir $logdir --sampling_method ATS --buffer_size 10 --utility_function sample --temperature 0.1 --scheduler_lr 0.001 --warmup 2000 --pretrain_iter 10000
```

For uniform sampling, we need to use the validation set to finetune the model trained under uniform sampling. The training commands are:  
(1) 1 shot
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0 --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.6
mkdir models
mv ANIL_pytorch.data_miniimagenetcls_5.mbs_2.ubs_1.metalr0.001.innerlr0.01.hidden32_noise0.6/model30000 ./models/ANIL_0.4_model_1shot
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0 --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.6 --finetune
```
(2) 5 shot
```
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.6
mkdir models  // if directory "models" does not exist
mv ANIL_pytorch.data_miniimagenetcls_5.mbs_2.ubs_5.metalr0.001.innerlr0.01.hidden32_noise0.6/model30000 ./models/ANIL_0.4_model_5shot
python3 main.py --meta_batch_size 2 --datasource miniimagenet --datadir $datadir --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --logdir $logdir --noise 0.6 --finetune
```





### ATS with limited budgets
In this setting, pretraining is not needed. You can directly run the following code:  
uniform sampling, 1 shot
```
python3 main.py --meta_batch_size 3 --datasource miniimagenet --datadir ./miniimagenet/ --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --limit_data 1 --logdir ../train_logs --limit_classes 16
```

uniform sampling, 5 shot
```
python3 main.py --meta_batch_size 3 --datasource miniimagenet --datadir ./miniimagenet/ --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --limit_data 1 --logdir ../train_logs --limit_classes 16
```
ATS 1 shot
```
python3 main.py --meta_batch_size 3 --datasource miniimagenet --datadir ./miniimagenet/ --num_updates 5 --num_updates_test 10 --update_batch_size 1 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --replace 0 --limit_data 1 --logdir ../train_logs --sampling_method ATS --buffer_size 6 --utility_function sample --temperature 1 --warmup 0 --limit_classes 16
```

ATS 5 shot
```
python3 main.py --meta_batch_size 3 --datasource miniimagenet --datadir ./miniimagenet/ --num_updates 5 --num_updates_test 10 --update_batch_size 5 --update_batch_size_eval 15 --resume 0  --num_classes 5 --metatrain_iterations 30000 --replace 0 --limit_data 1 --logdir ../train_logs --sampling_method ATS --buffer_size 6 --utility_function sample --temperature 0.1 --warmup 0 --limit_classes 16
```



## Drug
The processed dataset could be downloaded [here](https://drive.google.com/file/d/1GQtES5pt7YD4MWdEqKqxJQW1-rpXhPWZ/view?usp=sharing).
Assume the dataset has been downloaded and unzipped to `/data/drug` which has the following structure:
```
-- drug  // /data/drug
  -- ci9b00375_si_001.txt  
  -- compound_fp.npy               
  -- drug_split_id_group2.pickle  
  -- drug_split_id_group6.pickle
  -- ci9b00375_si_002.txt  
  -- drug_split_id_group17.pickle  
  -- drug_split_id_group3.pickle  
  -- drug_split_id_group9.pickle
  -- ci9b00375_si_003.txt  
  -- drug_split_id_group1.pickle   
  -- drug_split_id_group4.pickle  
  -- important_readme.md
```
Then `$datadir` in the following script should be set as `/data/`.

### ATS with noise=4. 

Uniform Sampling:  
```
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --noise 4 --data_dir $datadir
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --noise 4 --data_dir $datadir --train 0
```
ATS:
```
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --sampling_method ATS --noise 4 --data_dir $datadir
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --sampling_method ATS --noise 4 --data_dir $datadir --train 0
```

### ATS with full budgets
Uniform Sampling:
```
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --data_dir $datadir
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --data_dir $datadir --train 0
```

ATS:
```
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --sampling_method ATS --data_dir $datadir
python3 main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=17 --sampling_method ATS --data_dir $datadir --train 0
```

For ATS, if you need to use ![1](http://latex.codecogs.com/svg.latex?\theta_0) for calculating the loss as the input of the scheduler instead of ![1](http://latex.codecogs.com/svg.latex?\theta), you can add `--simple_loss` after the script above.  






