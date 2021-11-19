from lightgbm import LGBMRegressor
import pandas as pd

def main(data):
    model = LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=30,
        max_depth=10,
        learning_rate=0.003,
        n_estimators=500,
        reg_alpha=0.1,
        silent=False
    )
    model.fit(data['X'], data['label'])
    model.predict(data['test'])

if __name__ == '__main__':
    data = pd.read('data.csv')
    main(data)