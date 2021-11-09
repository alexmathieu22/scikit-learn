# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:41:15 2021

@author: Alex
"""


from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import wandb
import time




if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    from utils_v3 import DATASET_MAPPING, Wandb_Config
    
    @dataclass
    class Dataset_Config:
        name: str = "bike" #Dataset name defaults to "bike"
        n_jobs: int = 1 #Number of jobs for parallel processing.

    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Dataset_Config, "data")
    parser.add_arguments(Wandb_Config, "wandb")
    args, unknown = parser.parse_known_args()
    print(args)
    
    # Init W&B run
    if args.wandb.wandb:
        wandb.init(project=args.wandb.wandb_project, entity=args.wandb.wandb_entity)
        wandb.run.name = f"{args.data.name} - {wandb.run.name}"
        for main_arg in vars(args):
            if not main_arg == "wandb":
                wandb.config[main_arg] = getattr(args, main_arg).__dict__
    
    if wandb.wandb:
        wandb.run.summary["n_jobs"] = args.data.n_jobs

    if args.data.name == "kaggle_houses":
        X, y, features, _ = DATASET_MAPPING["kaggle_houses"](scaler=None)
    elif args.data.name == "bike":
        X, y, features, _ = DATASET_MAPPING["bike"](scaler=None)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #-------------------- RANDOM FOREST CLASSIFIER --------------------#
    print("Training Random Forest regressors...")
    
    #10*100 trees
    rf_list = []
    total = 0
    for i in range(10):
        start = time.time()
        clf_rf = RandomForestRegressor(n_jobs=args.data.n_jobs)
        clf_rf.fit(X_train, y_train)
        end = time.time()
        total += (end - start)
        print(f"Number of RFs trained: {i+1}")
        
        rf_list.append(clf_rf) 

    if wandb.wandb:
        #log time
        wandb.run.summary["RFs - train"] = total



    #----------------------------- FoF --------------------------------#
    print("Training Forest of Forests...")

    #1000 trees
    start = time.time()
    clf_fof = RandomForestRegressor(n_estimators=1000, n_jobs=args.data.n_jobs)
    clf_fof.fit(X_train, y_train)
    end = time.time()

    if wandb.wandb:
        #log time
        wandb.run.summary["FoF - train"] = (end - start)



    #--------------------------- TEST ---------------------------------#
    print("Testing Random Forest regressors...")

    start = time.time()
    all_rmse = np.array([])
    for rf in rf_list:
        y_pred = rf.predict(X_test)
        all_rmse = np.append(all_rmse, (np.mean((y_pred - y_test)**2))**(1/2))
    end = time.time()
    
    if wandb.wandb:
        wandb.run.summary["RFs - test"] = (end - start)
        wandb.run.summary["RFs - RMSE_Mean"] = np.mean(all_rmse)
        wandb.run.summary["RFs - RMSE_STD"] = np.std(all_rmse)
        
    
    print("Testing Forest of Forests...")
    #1000 trees
    start = time.time()
    _ = clf_fof.predict(X_test)
    
    
    all_rmse = np.array([])
    for i in range(0,10):
        y_pred = np.zeros(clf_fof.estimators_[0].latest_predictions.shape)
        for estimator in clf_fof.estimators_[100*i:100*(i+1)]:
            y_pred += estimator.latest_predictions
        y_pred /= 100
        all_rmse = np.append(all_rmse, (np.mean((y_pred - y_test)**2))**(1/2))
    end = time.time()
    
    if wandb.wandb:
        wandb.run.summary["FOF - test"] = (end - start)
        wandb.run.summary["FOF - RMSE_Mean"] = np.mean(all_rmse)
        wandb.run.summary["FOF - RMSE_STD"] = np.std(all_rmse)
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    