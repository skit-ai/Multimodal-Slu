# Downstream Tasks

## General rule of thumb
* Downstream tasks are to be run with upstream models, check the list of upstream models available [here](https://github.com/s3prl/s3prl/tree/master/upstream#upstream-models), or through the following:
```python
import torch
print(torch.hub.list('s3prl/s3prl'))
```
* The setup of all tasks are very simple, open up the config files upder each downstream directory (`downstream/*/config.yaml`) and download the required data.
* Run training with this command: `python run_downstream.py -m train -u baseline -d example -n NameOfExp`
* Downstream tasks can be specified with the argument `-d`, for example `-d phone_linear`.
* Upstream models can be specified with the argument `-u`, for example `-u tera`.
* Upstream models can be fine-tuned with downstream models with the argument `-f`, for example `python run_downstream.py -m train -u mockingjay -d phone_linear -n NameOfExp -f`.

## Adding new downstream tasks
* Please see this slide for a detailed tutorial: [link](https://docs.google.com/presentation/d/1QRau3NyuHM6KXa8j6Jnw_deFH6Y7s9_kO-_VjeS5LZs/edit?usp=sharing)
