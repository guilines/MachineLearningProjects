import numpy as np
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))




def get_df(symbol,initDate=None,finalDate=None):
    ''' Returns a data frame containing all stock values of a company
    in a period'''

    conn = sqlite3.connect('stock_data.sqlite')
    cur=conn.cursor()

    cur.execute(''' SELECT id FROM Companies WHERE symbol = ? ''',
                (symbol,)
    )
    try:
        symbolId = cur.fetchone()[0]
    except:
        print 'This symbol: {} is not available on this database. Please,\
        run the getData.py to achieve data'.format(symbol)

#    if finalDate is None:
#        finalDate = initDate
#    cur.execute(''' SELECT * FROM Period WHERE date >= ? AND date <= ? ''',
#                (initDate,finalDate)
#    )
#    dateIdLst = cur.fetchall()
#    if len(dateIdLst):
#        print 'This period: {} to {} is not avalible on this database. Please,
#        run the getData.py to achieve this data'.format(initDate,finalDate)
#


    disk_engine = create_engine('sqlite:///stock_data.sqlite')
    cmd = 'SELECT   Companies.name,\
                    Companies.symbol,\
                    Period.date,\
                    Data.open,\
                    Data.close,\
                    Data.low,\
                    Data.high,\
                    Data.volume,\
                    Data.adj_close\
                    FROM Data\
          LEFT JOIN Companies ON Data.symbol_id = Companies.id\
          LEFT JOIN Period ON Data.period_id = Period.id\
          WHERE symbol_id IS {}\
          ORDER BY date'.format(symbolId)
    df = pd.read_sql_query(cmd,disk_engine)
    return df


def evaluation_metric(y_true,y_predict):
    ''' Returns the performance score between
        true and predicted values, using r2_score'''
    return r2_score(y_true,y_predict)

from sklearn.metrics import make_scorer, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit


def decisionTree_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
    decision tree regressor trained on the input data [X, y]. """
                    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=23)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth' : range(1,11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(r2_score)

    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def svr_model(X, y):

    params_dist = {  'svr__C' : np.logspace(-3,2,6), 
                    'svr__kernel' : ['sigmoid','linear','poly','rbf']}#, 
#                    'svr__degree' : range(1,4),
#                    'svr__gamma' : np.logspace(-3,2,6)}

    steps = [('scaler',StandardScaler()),('svr', SVR())]
    pipeline = Pipeline(steps)

  
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(pipeline,param_grid=params_dist,scoring=scoring_fnc,n_jobs=4)
    grid = grid.fit(X, y)

    return grid.best_estimator_

from sklearn.ensemble import RandomForestRegressor 
def ensemble_model(X,y):


    regressor = RandomForestRegressor(random_state=23) 
    scoring_fnc = make_scorer(r2_score)
    regressor = regressor.fit(X, y)

    return regressor


def split_data(X,y):
    return train_test_split(X,y,test_size=0.2,random_state=23)



if __name__ == '__main__':
#    df = get_df('AAPL')
    df = get_df('YHOO')
    cName = df['name'][0]
    cSymbol = df['symbol'][0]
    adj_close_prices = df['adj_close']
    features = df.drop(df.columns[[0,1,2,8]], axis=1)
    features = features.drop('close', axis=1)

    print 'This dataset has {} data points with {} variables each.'.format(*features.shape)

    print 'Statistics for {} dataset:'.format(cName)
    print 'Minimum price: ${:,.2f}'.format(np.min(adj_close_prices))
    print 'Maximum price: ${:,.2f}'.format(np.max(adj_close_prices))
    print 'Average price: ${:,.2f}'.format(np.average(adj_close_prices))
    print 'Standart deviation of prices: ${:,.2f}'.format(np.std(adj_close_prices))
    print '-------------------------------------------------------------------------'


    X_train,X_test,y_train,y_test = split_data(features,adj_close_prices)
    
    
    ### -- DecisionTree Regressor
    reg = decisionTree_model(X_train, y_train)
    y_pred = reg.predict(X_test)
    print "Classifier:\n{}".format(reg.get_params)
    print "DTs - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
    print '-------------------------------------------------------------------------'

    
    ### -- SVR Regressor
    reg = svr_model(X_train, y_train)
    y_pred = reg.predict(X_test)
    print "Classifier:\n{}".format(reg.get_params)
    print "SVC - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
    print '-------------------------------------------------------------------------'

    ### -- RandomForest Regressor
    reg = ensemble_model(X_train, y_train)
    y_pred = reg.predict(X_test)
    print "Classifier:\n{}".format(reg.get_params)
    print "RandomForestRegressor - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
    print '-------------------------------------------------------------------------'
