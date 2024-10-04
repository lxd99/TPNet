# [NeurIPS 2024] Improving Temporal Link Prediction via Temporal Walk Matrix Projection

## Environments

- python=3.9.18
- pytorch=2.0.1
- numpy=1.24.3
- pandas=1.5.3
- scikit-learn==1.3.0
- wandb=0.16.3
- tqdm
- tabulate


## Benchmark Datasets and Preprocessing

Thirteen datasets are used in our experiments, including Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl.,  US Legis., UN Trade, UN Vote, and Contact. The first four datasets are bipartite, and the others only contain nodes with a single type.

The used original dynamic graph datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 

Please first download them and put them in ```DG_data``` folder. Before preprocessing datasets, we need to create necessary directories by the following command.

 

```
mkdir processed_data logs saved_results saved_models wandb
```

Then we can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:

```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```


## Executing Scripts
#### Model Training
* Example of training *TPNet* on *Wikipedia* dataset:
```{bash}
python train_link_prediction.py --prefix std  --dataset_name wikipedia --model_name TPNet  --num_runs 5 --gpu 0 --use_random_projection
```
* If you want to use the best model configurations to train *TPNet* on *Wikipedia* dataset, run
```{bash}
python train_link_prediction.py --prefix std  --dataset_name wikipedia --model_name TPNet  --num_runs 5 --gpu 0 --use_random_projection --load_best_configs
```
#### Model Evaluation
Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.
* Example of evaluating *TPNet* with *random* negative sampling strategy on *Wikipedia* dataset:
```{bash}
python evaluate_link_prediction.py --prefix std --dataset_name wikipedia --model_name TPNet --num_runs 5 --gpu 0 --use_random_projection --negative_sample_strategy random
```
* If you want to use the best model configurations to evaluate *TPNET* with *random* negative sampling strategy on *Wikipedia* dataset, run
```{bash}
python evaluate_link_prediction.py --prefix std --dataset_name wikipedia --model_name TPNet --num_runs 5 --gpu 0 --use_random_projection --load_best_configs --negative_sample_strategy random
```
## Useful File
You can refer to the `demo_on_matrix_updating.ipynb` file for more information about the updating functions of different temporal walk matrices.

## Acknowledgments
We are grateful to the authors of [DyGLib](https://github.com/yule-BUAA/DyGLib), [PINT](https://github.com/AaltoPML/PINT), and[NAT](https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network) for making their project codes publicly available.
