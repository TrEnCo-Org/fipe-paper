from copy import deepcopy
from datetime import datetime
from pathlib import Path

import logging
import warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np
import joblib as jl

import fipe
from fipe import FIPEPruner
from fipe import FeatureEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_dataset(
    dataset: int|str,
    openml: bool = True
):
    if openml:
        dataset = fetch_openml(data_id=dataset)
        X, y = dataset.data, dataset.target
        name = dataset.details['name']
    else:
        dataset_path = Path(dataset)
        data = pd.read_csv(dataset_path)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        name = dataset_path.stem
    return X, y, name

def encode_dataset(
    X: pd.DataFrame,
    y: pd.Series
):
    data = deepcopy(X)
    encoder = FeatureEncoder()
    encoder.fit(data)
    X = encoder.X.values

    y = y.astype('category').cat.codes
    y = np.array(y.values)

    return X, y

def run(
    dataset: int|str,
    openml: bool = True,
    logs=Path('__file__').parent/'logs',
    pickle_file=Path('__file__').parent/'prune.pkl',
    log_to_console=False,
    time_limit=600,
    threads=32
):
    logging.basicConfig(level=logging.DEBUG)
    logger.info(f'Running dataset {dataset}')
    X, y, name = load_dataset(dataset, openml)
    name = name.lower().replace(' ', '-')
    name = name.replace('_', '-')
    logger.info(f'Dataset name: {name} has been loaded. Encoding features...')
    X, y = encode_dataset(X, y)
    logger.info('Features have been encoded. Running the pruner...')
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    logger.info(f'creating logs directory...')
    # Log file for gurobi:
    if not isinstance(logs, Path):
        logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    gurobi_logs = logs / 'gurobi' / f'{dataset}'
    gurobi_logs.mkdir(parents=True, exist_ok=True)
    gurobi_log_file = gurobi_logs / f'{timestamp}.log'
    logger.info(f'log file: {gurobi_log_file}')
    logger.info('Splitting the dataset into training and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    logger.info('Training the random forest...')
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)
    logger.info('Random forest has been trained. Pruning the random forest...')
    w = np.ones(len(rf))
    
    pruner = FIPEPruner(rf, w)
    logger.info('Pruner has been created. Adding constraints...')
    pruner.build()
    pruner.add_constraints(X_train)
    logger.info('Constraints have been added. Setting gurobi parameters...')
    pruner.set_gurobi_parameter('LogToConsole', int(log_to_console))
    pruner.set_gurobi_parameter('LogFile', str(gurobi_log_file))
    pruner.set_gurobi_parameter('TimeLimit', time_limit)
    pruner.set_gurobi_parameter('OutputFlag', 1)
    pruner.set_gurobi_parameter('Threads', threads)
    logger.info('Gurobi parameters have been set. Pruning the random forest...')

    pruner.prune()
    logger.info('Random forest has been pruned. Evaluating the pruned model...')
    
    u = pruner.active
    
    if sum(u) == 0:
        warnings.warn('No trees were selected')
        return
    # Evaluate the pruned model
    # on the test set
    y_pred = fipe.predict(rf, X_test, w)
    proba_active = fipe.predict_proba(rf, X_test, w*u)
    max_proba_active = proba_active.max(axis=1)
    fidelity = 0
    for i in range(len(X_test)):
        fidelity += (max_proba_active[i] == proba_active[i, y_pred[i]])

    fidelity /= len(X_test)

    m = len(rf)
    # Sort the estimators by their fidelity
    # and select the top 10% of the estimators
    # to be used in the final model
    y_pred = fipe.predict(rf, X_train, w)
    fidelities = np.stack([
        (fipe.predict([e], X_train, [1])
         == y_pred).mean()
        for e in rf.estimators_
    ])
    idx = np.argsort(fidelities)[::-1]

    y_pred = fipe.predict(rf, X_test, w)
    fidelities = np.stack([
        (fipe.predict(
            rf.estimators_[:i],
            X_test,
            w[idx[:i]]
        ) == y_pred).mean()
        for i in range(1, m)
    ])

    pickle_data = {
        'dataset_id': dataset,
        'dataset': name,
        'n-trees': int(sum(u)),
        'timestamp': timestamp,
        'fidelity': fidelity,
        'greedy-fidelities': fidelities,
    }
    
    pickle_old = (jl.load(pickle_file)
                  if pickle_file.exists()
                  else [])
    pickle_data = pickle_old + [pickle_data]
    jl.dump(pickle_data, pickle_file)

if __name__ == '__main__':
    datasets = [
        44, # spambase
        4135, # Amazon_employee_access
        40982, # steel-plates-fault
        41703, # MIP-2016-classification
        43098, # Students_scores
        43672, # Heart-Disease-Dataset-(Comprehensive)
        45036, # default-of-credit-card-clients
        45058, # credit-g
        45068, # Adult
        45578, # California-Housing-Classification
    ]
    pickle_file = Path('__file__').parent/'prune.pkl'
    
    jl.Parallel(n_jobs=len(datasets))(
        jl.delayed(run)(
            dataset,
            pickle_file=pickle_file,
            time_limit=1000,
        )
        for dataset in datasets
    )
