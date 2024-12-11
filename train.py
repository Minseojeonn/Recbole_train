from logging import getLogger
from fire import Fire
from pathlib import Path
import os.path 
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN, BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


def main(model_name='LightGCN', dataset_name='ml-100k'):
    # configurations initialization
    config = Config(model=model_name, dataset=dataset_name)
    config["metrics"].append("GAUC")
    config["topk"] = [10, 20, 40]
    config["reproducibility"] = False
    config["show_progress"] = False
    config["worker"] = 8
    # init random seed
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model_dict = {
        'LightGCN': LightGCN,
        'BPR': BPR,
    }
    model = model_dict[model_name](config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    result_values = []
    for key in best_valid_result:
        result_values.append(best_valid_result[key])
    for key in test_result:
        result_values.append(test_result[key])
    
    file_name = f'./experiments/{model_name}_{dataset_name}.csv'
    if os.path.isfile(file_name):
        with open(file_name, 'a') as f:
            f.write(','.join(map(str, result_values)) + '\n')
    else:
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(','.join(best_valid_result.keys()) + ',' + ','.join(test_result.keys()) + '\n')
            f.write(','.join(map(str, result_values)) + '\n')
       
    
if __name__ == '__main__':
    Fire(main)