""" 
Utility functions for regression experiments 

Contains get_data functions, Scaler class, evaluate_aggregate, evaluate_ensemble functions and mapping dictionnaries.

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
# Sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from math import ceil, sqrt

# Graphic Libraries
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.io as pio
# pio.templates.default = "simple_white"
# pio.renderers.default = 'browser'

# Deep Learning
import torch

from dataclasses import dataclass, asdict, field

import sys, os

# # add uxai path relative to utils_v2 location
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from uxai.samplers import UniformSampler
# from uxai.ensembles import Ensemble
# from uxai.explainers import EG_Explainer, SHAP_Explainer
    

class Scaler(object):
    """ Object used to scale features and regression target """

    def __init__(self, scaler_type, task):
        """Initialize the feature scaler and target scalers.

        Args:
            scaler_type (string): Scaler_type name (e.g. MinMax, Standard).
            task (string): Type of task (e.g. Regression).

        Raises:
            ValueError: The scaler type isn't supported.
        """
        if scaler_type == "MinMax":
            self.scaler_x = MinMaxScaler()
            if task == "regression":
                self.scaler_y = MinMaxScaler()

        elif scaler_type == "Standard":
            self.scaler_x = StandardScaler()
            if task == "regression":
                self.scaler_y = StandardScaler()

        else:
            raise ValueError("Scaler must be MinMax or Standard")

    def fit_transform(self, X, y):
        """Fit and transform the unprocessed data which is still stored as 
            Numpy arrays or DataFrames

        Args:
            X (Numpy array/DataFrame): Array of all X values.
            y (Numpy array/DataFrame): Array of all y values.

        Returns:
            np.array/dataframe, np.array/dataframe: Fit and transformed X and y arrays.
        """

        return (
            self.scaler_x.fit_transform(X),
            self.scaler_y.fit_transform(y.reshape((-1, 1))).ravel(),
        )

    def invscale_target(self, y_tensor):
        """Scale back the regression target in a way that is 
            compatible with Pytorch

        Args:
            y_tensor (Tensor): Regression target.

        Returns:
            Tensor: Scaled regression target.
        """
        if type(y_tensor) in [float, np.float32, np.float64, np.ndarray]:
            if type(self.scaler_y) == MinMaxScaler:
                return y_tensor / float(self.scaler_y.scale_[0])
            elif type(self.scaler_y) == StandardScaler:
                return y_tensor * float(self.scaler_y.scale_[0])
        elif type(y_tensor) == torch.Tensor:
            if type(self.scaler_y) == MinMaxScaler:
                return y_tensor.cpu() / self.scaler_y.scale_[0]
            elif type(self.scaler_y) == StandardScaler:
                return y_tensor.cpu() * self.scaler_y.scale_[0]


# Boolean feature
class bool_value_mapper(object):
    """ Organise feature values as 1->true or 0->false """

    def __init__(self):
        self.values = ["False", "True"]

    # map 0->False  1->True
    def __call__(self, x):
        return self.values[round(x)]


# Ordinal encoding of categorical features
class cat_value_mapper(object):
    """ organise categorical features  int_value->'string_value' """

    def __init__(self, categories_in_order):
        self.cats = categories_in_order

    # x takes values 0, 1, 2 ,3  return the category
    def __call__(self, x):
        return self.cats[round(x)]


# Numerical features x in [xmin, xmax]
class numerical_value_mapper(object):
    """ organise feature values in quantiles  value->{low, medium, high}"""

    def __init__(self, num_feature_values):
        self.quantiles = np.quantile(num_feature_values, [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        return self.quantifiers[
            np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
        ]


# Numerical features but with lots of zeros x in {0} U [xmin, xmax]
class sparse_numerical_value_mapper(object):
    """ organise feature values in quantiles but treat 0-values differently
    """

    def __init__(self, num_feature_values):
        idx = np.where(num_feature_values != 0)[0]
        self.quantiles = np.quantile(
            num_feature_values[idx], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x == 0:
            return int(x)
        else:
            return self.quantifiers[
                np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
            ]


class Features(object):
    """ An abstraction of the concept of a feature. Useful when doing
    feature importance plots"""

    def __init__(self, X, feature_names, feature_types):
        self.names = feature_names
        self.types = feature_types
        # map feature values to interpretable text
        self.maps = []
        for i, feature_type in enumerate(self.types):
            # If its a list then the feature is categorical
            if type(feature_type) == list:
                self.maps.append(cat_value_mapper(feature_type))
            elif feature_type == "num":
                self.maps.append(numerical_value_mapper(X[:, i]))
            elif feature_type == "sparse_num":
                self.maps.append(sparse_numerical_value_mapper(X[:, i]))
            elif feature_type == "bool":
                self.maps.append(bool_value_mapper())
            elif feature_type == "num_int":
                self.maps.append(lambda x: round(x))
            else:
                raise ValueError("Wrong feature type")
                
    def map_values(self, x):
        """ Map values of x into interpretable text """
        return [self.maps[i](x[i]) for i in range(len(x))]

    def __len__(self):
        return len(self.names)


data_dir = "../../datasets/"



def get_data_bike(scaler="MinMax", verbose=False):
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "datasets", "Bike-Sharing/hour.csv")
    )
    df.drop(columns=["dteday", "casual", "registered", "instant"], inplace=True)

    # Remove correlated features
    df.drop(columns=["atemp", "season"], inplace=True)

    # Rescale temp to Celcius
    df["temp"] = 41 * df["temp"]

    # Month count starts at 0
    df["mnth"] -= 1

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)

    # Scale all features
    feature_names = list(df.columns[:-1])

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Generate Features object
    feature_types = [
        ["2011", "2012"],
        [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October","November", "December",
        ],
        "num_int",
        "bool",
        ["Sunday", "Monday", "Thuesday", "Wednesday", "Thursday",
         "Friday", "Saturday"],
        "bool",
        "num_int",
        "num",
        "num",
        "num",
    ]

    features = Features(X, feature_names, feature_types)

    # Scale data
    if scaler is not None:
        scaler = Scaler(scaler, "regression")
        X, y = scaler.fit_transform(X, y)

    return X, y, features, scaler



# def get_data_student():
#     #Here, please change the string for the folder in which your csv file is located.
#     df = pd.read_csv("datasets/student/student-mat.csv", delimiter=";")

#     df.drop(columns=["nursery", "romantic", "Mjob", "Fjob", "reason", "guardian",
#                      "G1", "G2"], inplace=True)

#     #Tranformation of data in an appropriate format
#     cat_features = ["school", "sex", "address", "famsize", "Pstatus",
#                     "schoolsup", "famsup", "paid", "activities",
#                     "higher", "internet"]
#     nume_features = ["age", "traveltime", "Medu", "Fedu", "studytime",
#                      "failures", "famrel",
#                      "freetime", "goout", "Dalc", "Walc", "health", "absences"]

#     # Processed data
#     X = np.zeros((len(df), len(df.columns) - 1))
#     X[:, :len(nume_features)] = df[nume_features]
#     encoder = OneHotEncoder(sparse = False)
#     bob =  encoder.fit_transform(df[cat_features])


#     # Scale all features
#     features = list(df.columns[:-1])

#     X = df.to_numpy()[:, :-1]
#     y = df.to_numpy()[:, -1]
#     scaler_x = MinMaxScaler()
#     scaler_y = MinMaxScaler()
#     X = scaler_x.fit_transform(X)
#     y = scaler_y.fit_transform(y.reshape((-1, 1))).ravel()

#     x_train, x_test, y_train, y_test = train_test_split(X, y,
#                                                         test_size=0.15, random_state=42)
#     return x_train, x_test, y_train, y_test, features, scaler_x, scaler_y



# def get_data_boston(scaler="MinMax", verbose=False):
#     data = load_boston()

#     X, y, features = data["data"], data["target"], data["feature_names"]

#     # Remove CHAS feature
#     X = np.delete(X, 3, 1)
#     features = np.delete(features, 3, 0)

#     # Print the range of the target
#     if verbose:
#         print("Loading Boston Housing\n")
#         print(f"Target Range [{np.min(y):.2f}, {np.max(y):.2f}]\n")

#     # Scale data
#     scaler = Scaler(scaler, "regression")
#     X, y = scaler.fit_transform(X, y)

#     return X, y, features, scaler



def get_data_houses(scaler="MinMax", verbose=False):
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "datasets", "kaggle_houses", "train.csv"
        )
    )

    # dropping categorical features
    df.drop(
        labels=[
            "Id", "MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
            "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
            "Condition1", "Condition2", "BldgType", "HouseStyle", "MSZoning",
            "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
            "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",
            "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
            "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
            "MiscFeature", "SaleType", "SaleCondition", "CentralAir","PavedDrive"
        ],
        axis=1,
        inplace=True,
    )

    # shuffle the data
    df = df.sample(frac=1, random_state=42)

    # Replace missing values by the mean
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    df["MasVnrArea"] = imp.fit_transform(df[["MasVnrArea"]])
    df["GarageYrBlt"] = imp.fit_transform(df[["GarageYrBlt"]])

    # dropping LotFrontage because it is missing 259/1460 values which is a lot (GarageYrBlt: 81 and MasVnrArea: 8 is reasonable)
    df.drop(labels=["LotFrontage"], axis=1, inplace=True)

    # correlation
    df.drop(labels=["GarageCars"], axis=1, inplace=True)

    # dropping GarageYrBlt because it is highly correlated (0.84791) with YearBuilt.
    df.drop(columns=["GarageYrBlt"], inplace=True)

    # TotalBsmtSF: Total square feet of basement area and 1stFlrSF: First Floor square feet are highly correlated (0.829292)
    df.drop(columns=["TotalBsmtSF"], inplace=True)

    # GrLivArea: Above grade (ground) living area square feet and TotRmsAbvGrd: Total rooms above
    # grade (does not include bathrooms) are highly correlated (0.827874)
    df.drop(columns=["TotRmsAbvGrd"], inplace=True)

    # All features under 0.1: BsmtFinSF2, LowQualFinSF, BsmtHalfBath, 3SsnPorch, PoolArea, MiscVal, MoSold, YrSold
    # Almost no values == (almost all values have a value of 0)

    # BsmtFinSF2 almost no values
    df.drop(columns=["BsmtFinSF2"], inplace=True)

    # BsmtHalfBath had a really low (2nd lowest) with the target (-0.0121889), almost no values
    df.drop(columns=["BsmtHalfBath"], inplace=True)

    # MoSold low correlation with target -0.0298991, I
    # removed year because it was the lowest correlation of all features with the target
    df.drop(columns=["MoSold"], inplace=True)

    # LowQualFinSF: Low quality finished square feet (all floors), almost no values
    df.drop(columns=["LowQualFinSF"], inplace=True)

    # MiscVal almost no values
    df.drop(columns=["MiscVal"], inplace=True)

    # Pool area almost no values
    df.drop(columns=["PoolArea"], inplace=True)

    # 3SsnPorch almost no values
    df.drop(columns=["3SsnPorch"], inplace=True)

    # MR under 1 (hasse graph)
    df.drop(columns=["FullBath"], inplace=True)
    df.drop(columns=["HalfBath"], inplace=True)

    # Too similar to first and second floor areas
    df.drop(columns=["GrLivArea"], inplace=True)

    # #basement adding both values?
    # df.insert(0, "BsmtScore", df["BsmtUnfSF"])

    # df.loc[(df.BsmtUnfSF == 0) & (df.BsmtFinSF1 != 0), "BsmtScore"] = df.BsmtFinSF1
    # df.loc[df.BsmtFinSF1 == 0, "BsmtScore"] = 0
    # df.loc[(df.BsmtScore != 0) & (df.BsmtScore != df.BsmtFinSF1), "BsmtScore"] = (df.BsmtFinSF1/df.BsmtUnfSF)

    # df.drop(columns=["BsmtUnfSF", "BsmtFinSF1"], inplace=True)

    # #basement binary?
    # df.insert(0, "BsmtUnfBinary", df["BsmtUnfSF"])
    # df.loc[df.BsmtUnfSF > 0, "BsmtUnfBinary"] = 1
    # df.loc[df.BsmtUnfSF == 0, "BsmtUnfBinary"] = 0

    # df.drop(columns=["BsmtUnfSF", "BsmtFinSF1"], inplace=True)

    # Remove outliers
    df = df[df["SalePrice"] < 500000]
    df = df[df["SalePrice"] > 50000]

    # Scale all features
    feature_names = list(df.columns[:-1])

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Generate Features object
    feature_types = [
        "num",
        "num_int",
        "num_int",
        "num",
        "num",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "num",
        "sparse_num",
        "num_int",
        "num_int",
        "num_int",
        "num_int",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "num_int",
    ]

    features = Features(X, feature_names, feature_types)

    # Scale data
    if scaler is not None:
        scaler = Scaler(scaler, "regression")
        X, y = scaler.fit_transform(X, y)

    return X, y, features, scaler



def get_data_california_housing(scaler="MinMax", verbose=False):
    data = fetch_california_housing()

    X, y, feature_names = data["data"], data["target"], data["feature_names"]

    # Remove outlier
    keep_bool = X[:, 5] < 1000
    X = X[keep_bool]
    y = y[keep_bool]
    del keep_bool

    # Take log of right-skewed features
    for i in [2, 3, 5]:
        X[:, i] = np.log10(X[:, i])
        feature_names[i] = f"log_{feature_names[i]}"

    # Add additionnal location feature
    def closest_point(location):
        # Biggest cities in 1990
        # Los Angeles, San Francisco, San Diego, San Jose
        biggest_cities = [
            (34.052235, -118.243683),
            (37.773972, -122.431297),
            (32.715736, -117.161087),
            (37.352390, -121.953079),
        ]
        closest_location = None
        for city_x, city_y in biggest_cities:
            distance = ((city_x - location[0]) ** 2 + (city_y - location[1]) ** 2) ** (
                1 / 2
            )
            if closest_location is None:
                closest_location = distance
            elif distance < closest_location:
                closest_location = distance
        return closest_location

    X = np.column_stack((X, [closest_point(x[-2:]) for x in X]))
    feature_names.append('ClosestBigCityDist')

    # Generate Features object
    feature_types = ["num", "num", "num", "num",\
                     "num", "num", "num", "num", "num"]
    
    features = Features(X, feature_names, feature_types)
    
    # Scale data
    if scaler is not None:
        scaler = Scaler(scaler, "regression")
        X, y = scaler.fit_transform(X, y)

    return X, y, features, scaler



def get_data_adults(scaler="MinMax", verbose=False):

    # load train
    raw_data_1 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.data'), 
                                                     delimiter=', ', dtype=str)
    # load test
    raw_data_2 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.test'),
                                      delimiter=', ', dtype=str, skip_header=1)

    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 
                     'educational-num', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Shuffle train/test
    df = pd.DataFrame(np.vstack((raw_data_1, raw_data_2)), columns=feature_names)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, 
                    "hours-per-week": np.int64, "capital-gain": np.int64, 
                    "capital-loss": np.int64 })

    # Reduce number of categories
    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 
                                   'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 
                                   'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 
                                   'Self-emp-inc': 'Self-Employed'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 
                                    'Craft-repair': 'Blue-Collar',
                                    'Exec-managerial':'White-Collar',
                                    'Farming-fishing':'Blue-Collar',
                                    'Handlers-cleaners':'Blue-Collar',
                                    'Machine-op-inspct':'Blue-Collar',
                                    'Other-service':'Service',
                                    'Priv-house-serv':'Service',
                                    'Prof-specialty':'Professional',
                                    'Protective-serv':'Service',
                                    'Tech-support':'Service',
                                    'Transport-moving':'Blue-Collar',
                                    'Unknown':'Other/Unknown',
                                    'Armed-Forces':'Other/Unknown',
                                    '?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 
                                        'Married-AF-spouse': 'Married', 
                                        'Married-spouse-absent':'Married',
                                        'Never-married':'Single'}})

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                   '11th':'School', '10th':'School', 
                                   '7th-8th':'School', '9th':'School',
                                   '12th':'School', '5th-6th':'School', 
                                   '1st-4th':'School', 'Preschool':'School'}})

    # Put numeric before categoric and remove fnlwgt-country
    df = df[['age', 'educational-num', 'capital-gain', 'capital-loss',
             'hours-per-week', 'gender', 'workclass','education', 'marital-status', 
             'occupation', 'relationship', 'race', 'native-country', 'income']]


    df = df.rename(columns={'educational-num': 'educational_num',
                            'marital-status': 'marital_status', 
                            'hours-per-week': 'hours_per_week', 
                            'capital-gain': 'capital_gain', 
                            'capital-loss': 'capital_loss'})

    df = shuffle(df, random_state=42)
    feature_names = df.columns[:-1]
    
    # Make a column transformer for ordinal encoder
    encoder = ColumnTransformer(transformers=
                      [('scaler', StandardScaler(), df.columns[:5]),
                       ('encoder', OrdinalEncoder(), df.columns[5:-1])
                      ])
    X = encoder.fit_transform(df.iloc[:, :-1])
    y = df["income"].to_numpy()
    
    # Generate Features object
    feature_types = ["num", "num", "sparse_num", "sparse_num", "num"] + \
                     [list(l) for l in encoder.transformers_[1][1].categories_]
    
    features = Features(X, feature_names, feature_types)
    
    return X, y, features
    

def evaluate_aggregate(models, valid_loader, scaler):
    """ 
    Evaluate the aggregated predictor on train and test sets
    """

    n_examples = len(valid_loader.dataset)
    error = 0

    with torch.no_grad():
        for (x, y) in valid_loader:
            y_pred = models(x)
            y_pred = y_pred.cpu()

            # Aggregate the ensemble
            y_pred = torch.mean(y_pred, dim=0)

            if models.hparams.task == "regression":
                error += (y_pred - y).pow(2).sum(dim=0).item()
            else:
                pred = (y_pred >= 0.5).int()
                error += (pred != y).float().sum(dim=0).item()

    error /= n_examples

    if models.hparams.task == "regression":
        # Take RMSE
        error = scaler.invscale_target(sqrt(error))

    return error


def evaluate_ensemble(models, data_loaders, scaler):
    """ 
    Evaluate each model of the ensemble on train and test sets
    """

    errors = torch.zeros(models.hparams.size_ensemble + 1, len(data_loaders))

    with torch.no_grad():
        for i, data_loader in enumerate(data_loaders):

            n_examples = len(data_loader.dataset)
            for (x, y) in data_loader:
                y_pred = models(x)
                y_pred = y_pred.cpu()

                # Aggregate the ensemble
                mean_pred = torch.mean(y_pred, dim=0)

                if models.hparams.task == "regression":
                    errors[:-1, [i]] += (y_pred - y).pow(2).sum(dim=1)
                    errors[-1, i] += (mean_pred - y).pow(2).sum(dim=0).item()
                else:
                    pred = (y_pred >= 0.5).int()
                    errors[:-1, [i]] += (pred != y).float().sum(dim=1)
                    mean_pred = (mean_pred >= 0.5).int()
                    errors[-1, i] += (mean_pred != y).float().sum(dim=0).item()

            errors[:, i] /= n_examples

    if models.hparams.task == "regression":
        # Take RMSE
        errors = scaler.invscale_target(torch.sqrt(errors))

    return errors

# Mappings for the different datasets to customize the code

DATASET_MAPPING = {
    "bike": get_data_bike,
    # "boston": get_data_boston,
     "kaggle_houses" : get_data_houses,
     "california" : get_data_california_housing,
     "adult_income" : get_data_adults
}

THRESHOLDS_MAPPING = {
    "kaggle_houses": 28090,
    "bike": 46.71,
    "california": 0.6865
}

CONFIDENCE_MAPPING = {
    "bike" : 52.85,
    "kaggle_houses" : 29316.15
}

TASK_MAPPING = {
    "kaggle_houses": "regression",
    "bike": "regression",
    "california": "regression"
}


COALLITION_MAPPING = {
    # Socio-Economic  vs  Location
    "california" : np.array([[0], [1], [2], [3], [4] ,[5], [6, 7, 8]], dtype=object),
    # Time/Day    vs    Meteo
    "bike": np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]], dtype=object),
    # Quality   vs   Area   vs   Content   vs   Basement   vs   other
    "kaggle_houses": np.array(
        [
            [1, 2, 3, 4],
            [0, 5, 8, 9, 14, 15, 16, 17, 18],
            [11, 12, 13],
            [6, 7, 10],
            [19],
        ],
        dtype=object,
    ),
}


@dataclass
class Wandb_Config:
    wandb: bool = False  # Use wandb logging
    wandb_project: str = "XUncertainty - FoF"  # Which wandb project to use
    wandb_entity: str = "alexandremathieu"  # Which wandb entity to use


@dataclass
class Data_Config:
    name: str = "bike"  # Name of dataset "bike", "california", "boston"
    batch_size: int = 100  # Mini batch size
    scaler: str = "MinMax"  # Scaler used for features and target
    data_seed: int = 0  # Seed for toy data


@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    cross_valid: str = "Shuffle" # Type of cross-valid "Shuffle" "K-fold"
    split_seed: int = 1 # Seed for the train/valid splits reproducability
    

@dataclass
class Explain_Config():
    explainer: str = "SHAP" # Type fo explainer "EG", "SHAP", "OWEN"
    instance: int = 1 # Instance to explain
    MC_seed: int = 42 # Seed for Monte Carlo estimations
    MC_samples: int = 10000 # Number of Monte Carlo samples
    top_ranks: int = 2 # Max number of ranks to plot in Hasse diagram
    sparsify: bool = False # Sparsify the Hasse diagram by ignoring features


if __name__ == "__main__":
    get_data_adults()
