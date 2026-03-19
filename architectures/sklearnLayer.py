import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def generate_statistical_pipeline(model_name, seed):
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, random_state=seed)
    elif model_name == "svr":
        from sklearn.svm import SVR
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=seed)
    elif model_name == "gb":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, random_state=seed)
    elif model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])