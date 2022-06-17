# rl_policy:

## to deploy experiments

### automated creation the exp folders and sub folders

Traditionally, deep rl is sensitive to various hyperparamers of the traing like the algorithm hyperparametes and env hyperparameters. Hence, we are required to deploy multiple trainings in parallel, i.e. a grid search over all possible combination of hyperparameters. 

Thus, to deploy such a trainig, creaate a yaml file, titled `exp_name.yaml` with all the env params listed. Here for each choice of value for the hyperparameters, you are required to have a `<start>`, `<next>`, and `<stop>` tags. Checkout the example job conf's available [here](./trng_job_confs/).


For example, to create the exp folders for all the jobs present as yamls inside the folder `rl_policy/trng_job_confs/`, run

    python3 rl_policy/deploy_trainings/main.py --input_job_folderpath rl_policy/trng_job_confs/ --output_exp_path experiments/

as of 17/6/22, this command creates all the necessary folder and subfolders inside the output path (in this example `experiments/`, which should exist in the first place), ready to be deployed for training. (job deployment automation is to be added soon)

### to start a training

To start a single training, simply run a command similar to

    python3 rl_policy/train.py --single_trng_path experiments/17Jun1/000/

For now, you may have to manually run this for each training under a experiment, but will be automated soon.

### to test a trained policy


