# CS294-112 HW 1: Imitation Learning

## Setup

make new directories for different sections:

```bash
mkdir expert_data
mkdir bc
mkdir dagger
```

## Section 2
#### S2Q1: Generate rollouts for expert's policies:

```bash
python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts 100
python run_expert.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 --num_rollouts 100
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 100
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 100
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts 100
python run_expert.py experts/Walker2d-v2.pkl Walker2d-v2 --num_rollouts 100
```

#### S2Q2: Train task `Hopper-v2` and `Ant-v2` on behavioral cloning:

```bash
# generate training data
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 100 --output_rollout
python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts 100 --output_rollout
# train BC
python hw1.py Hopper-v2 --data_file expert_data/Hopper-v2_100.pkl --bc --epochs 30
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 30
# test trained BC model
python hw1.py Hopper-v2 --data_file expert_data/Hopper-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
```

#### S2Q3: Hyperparameter - Epochs

```bash
# generate training data
python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts 100 --output_rollout
# train and test BC with epochs = [1, 3, 5, 10, 20, 30]
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 1; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 3; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 5; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 10; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 20; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --epochs 30; python hw1.py Ant-v2 --data_file expert_data/Ant-v2_100.pkl --bc --num_rollouts 20
```

## Section 3

```bash
# generate training data
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts 100
# train and test DAgger with iteration = [1, 2, 3, 4, 5, 7, 10]
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 1; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 2; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 3; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 4; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 5; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 7; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --dagger --iteration 10; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --dagger --num_rollouts 20
```

## Section 4

```bash
mkdir alternative_bc
mkdir alternative_dagger

# train BC
python hw1.py Hopper-v2 --data_file expert_data/Hopper-v2_100.pkl --alternative bc --epochs 30
# test 
python hw1.py Hopper-v2 --data_file expert_data/Hopper-v2_100.pkl --alternative bc --num_rollouts 20

# train and test DAgger
python hw1.py Reacher-v2 --expert_policy_file experts/Reacher-v2.pkl --data_file expert_data/Reacher-v2_100.pkl --alternative dagger --iteration 10; python hw1.py Reacher-v2 --data_file expert_data/Reacher-v2_100.pkl --alternative dagger --num_rollouts 20

```



## Plot

plot figures for 2.3 and 3.2:

```
python plot.py
```

