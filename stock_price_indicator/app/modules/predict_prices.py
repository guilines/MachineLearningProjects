import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import sqlite3
from sqlalchemy import create_engine
import scipy
import matplotlib.pyplot as plt

#SKLEARN libs
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

#KERAS libs
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(sum(map(ord, "distributions")))

class PredStockPrices:
    def __init__(self,db_path):
        self.symbol = ''
        self.period = None
        self.database = db_path
        self.df = None
        self.method = None
        self.X = None
        self.y = None
        self.reg = None

    def load_data(self,symbol,initDate=None,finalDate=None):
        ''' Loads data available in the database given the company symbol
        into the class'''

        conn = sqlite3.connect(self.database)
        cur=conn.cursor()

        cur.execute(''' SELECT id FROM Companies WHERE symbol = ? ''',
                    (symbol,)
        )
        try:
            symbolId = cur.fetchone()[0]
        except:
            print 'This symbol: {} is not available on this database. Please,\
            run the getData.py to achieve data'.format(symbol)

        disk_engine = create_engine('sqlite:///{}'.format(self.database))
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
        self.df = df

    def get_df(self):
        ''' Returns a data frame containing all stock values of a company
        in a period'''
        return self.df

    def set_method(self,method):
        ''' Selects the method that will be used and starts the predictor'''
        self._prepare_data()
        if method == 'Ensemble':
            self.ensemble_model()
        elif method == 'Support Vector Machine':
            self.machineVec_model()
        elif method == 'Deep Learning':
            self.deepL_model()
        elif method == 'Gaussian':
            self.gaussian_model()
        elif method == 'K-Nearest Neighbor':
            self.KNN_model()
        elif method == 'Neural Network':
            self.neuralNet_model()
        else:
            print 'Wrong method: {}'.format(method)

    def get_method(self):
        ''' Get method being used'''
        return self.method

    def _prepare_data(self):
        ''' Local function to reshape the data to be used'''
        data_len = len(self.df['adj_close'])
        dates = np.arange(data_len)
        self.X = np.reshape(dates,(-1,1))
        self.y = np.reshape(self.df['adj_close'],-1)


    def ensemble_model(self,X=None,y=None):
        ''' Creates an ensemble gradient boosting regressor'''
        self.method = 'Gradient Boosting  Regressor'
        X_train,X_test,y_train,y_test = self.split_data()
        regressor = GradientBoostingRegressor(random_state=23) 
        regressor = regressor.fit(X_train, y_train)
        self.reg = regressor

    def machineVec_model(self,X=None,y=None):
        ''' Creates a support vector machine regressor, and uses
        pipelines to determine the best regressor'''
        self.method = 'SVR - Support Vector Regressor'
        X_train,X_test,y_train,y_test = self.split_data()

        regressor = self._find_SVR_best_estimator(X_train, y_train)
        self.reg = regressor

    def deepL_model(self,X=None,y=None):
        ''' Creates a sequential deep learning regressor'''
        self.method = 'Deep Learning: Sequential'
        X_train,X_test,y_train,y_test = self.split_data()

        regressor = Sequential()
        regressor.add(Dense(8, input_dim=1, activation='relu'))
        regressor.add(Dense(1))
        regressor.compile(loss='mean_squared_error', optimizer='adam')
        regressor.fit(X_train, y_train, nb_epoch=200, batch_size=2, verbose=0)
        self.reg = regressor

    def gaussian_model(self,X=None,y=None):
        ''' Creates a gaussian regressor'''
        self.method = 'Gaussian Process Regressor'
        X_train,X_test,y_train,y_test = self.split_data()

        regressor = GaussianProcessRegressor(random_state=23)
        regressor = regressor.fit(X_train, y_train)
        self.reg = regressor


    def KNN_model(self,X=None,y=None):
        ''' Creates a K-Nearest Neighbor regressor'''
        self.method = 'K-Nearest Neighbors Regressor'
        X_train,X_test,y_train,y_test = self.split_data()

        regressor = KNeighborsRegressor()
        regressor = regressor.fit(X_train, y_train)
        self.reg = regressor

    def neuralNet_model(self,X=None,y=None):
        ''' Creates a Neural Network - Multi-Layer Perceptron regressor'''
        self.method = 'Multi-layer Perceptron Regressor'
        X_train,X_test,y_train,y_test = self.split_data()

        regressor = MLPRegressor()
        regressor = regressor.fit(X_train, y_train)
        self.reg = regressor

    def _find_SVR_best_estimator(self,X, y):
        '''Determine the best parameters to SVR RBF kernel '''

        params_dist = { 'svr__C'      : np.logspace(-3,2,6), 
                        'svr__kernel' : ['rbf'], 
                        'svr__degree' : np.logspace(-1,3,6),
                        'svr__gamma'  : np.logspace(-3,2,6)}

        steps = [('scaler',StandardScaler()),('svr', SVR())]
        pipeline = Pipeline(steps)

  
        scoring_fnc = make_scorer(r2_score)
        grid = GridSearchCV(pipeline,param_grid=params_dist,scoring=scoring_fnc,n_jobs=4)
        grid = grid.fit(X, y)

        return grid.best_estimator_



    def split_data(self,X=None,y=None):
        ''' Splits data into Training and Testing'''
        if X is None and y is None:
            return train_test_split(self.X,self.y,test_size=0.2,random_state=23)
        elif X is None and y is not None:
            return train_test_split(self.X,y,test_size=0.2,random_state=23)
        elif X is not None and y is None:
            return train_test_split(X,self.y,test_size=0.2,random_state=23)
        else:
            return None

    def get_pred(self,pred=None):
        ''' Predicts data '''
        if pred is None:
            return np.reshape(self.reg.predict(self.X),(-1,))
        else:
            return np.reshape(self.reg.predict(pred),(-1,))
    
    def get_r2Score(self,y_predict,y_true=None):
        ''' Returns the performance score between
            true and predicted values, using r2_score'''
        if y_true is None:
            y_true = self.y
        return r2_score(y_true,y_predict)

    def get_params(self):
        ''' Returns the params of the regressors, if available'''
        try:
            params = '{}'.format(self.reg.get_params())
        except:
            params = None
        return params


#--------------------------------------------------------------------
## Running code
#The code below is used for debug and command-line execution. It is
#not stable.
#--------------------------------------------------------------------

def get_df(symbol,initDate=None,finalDate=None):
    ''' Returns a data frame containing all stock values of a company
    in a period'''

    conn = sqlite3.connect('modules/database/stock_data.sqlite')
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


    disk_engine = create_engine('sqlite:///modules/database/stock_data.sqlite')
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


def svr_model(X, y):

    params_dist = {  'svr__C' : [15.3], 
                    'svr__kernel' : ['sigmoid','linear','poly','rbf']}#, 
#                    'svr__degree' : range(1,4)}#,
#                    'svr__gamma' : np.logspace(-3,2,6)}

    steps = [('scaler',StandardScaler()),('svr', SVR())]
    pipeline = Pipeline(steps)

  
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(pipeline,param_grid=params_dist,scoring=scoring_fnc,n_jobs=4)
    grid = grid.fit(X, y)

    return grid.best_estimator_

def ensemble_model(X,y):


    regressor = RandomForestRegressor(random_state=23) 
    scoring_fnc = make_scorer(r2_score)
    regressor = regressor.fit(X, y)

    return regressor


def split_data(X,y):
    return train_test_split(X,y,test_size=0.2,random_state=23)

def PRED_TEST():
    df = get_df('ERJ')
    return df['date'].values, df['close'].values, df['adj_close'].values

    adj_close_prices = np.reshape(df['adj_close'],-1)

    data_len = len(df['adj_close'])
    dates = np.arange(data_len)
    dates = np.reshape(dates,(-1,1))
    X_train,X_test,y_train,y_test = split_data(dates,adj_close_prices)
    priceReg = ensemble_model(X_train,y_train)
    y_pred = priceReg.predict(dates)

    return df

if __name__ == '__main__':
#    df = get_df('AAPL')
    df = get_df('ERJ')
    cName = df['name'][0]
    cSymbol = df['symbol'][0]
    adj_close_prices = df['adj_close']
    dates_len = np.arange(len(df['date']))
    features = df.drop(df.columns[[0,1,2,8]], axis=1)
#    features = features.drop('close', axis=1)


    print 'This dataset has {} data points with {} variables each.'.format(*features.shape)

    print 'Statistics for {} dataset:'.format(cName)
    print 'Minimum price: ${:,.2f}'.format(np.min(adj_close_prices))
    print 'Maximum price: ${:,.2f}'.format(np.max(adj_close_prices))
    print 'Average price: ${:,.2f}'.format(np.average(adj_close_prices))
    print 'Standart deviation of prices: ${:,.2f}'.format(np.std(adj_close_prices))
    print '-------------------------------------------------------------------------'

    ### -- Plots
    enable_plots = False
    if enable_plots:
        fNames = list(features)
        correlations = features.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(fNames),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(list(fNames))
        ax.set_yticklabels(list(fNames))
        plt.title('Correlation')
        scatter_matrix(features)
        plt.show()


    X_train,X_test,y_train,y_test = split_data(features,adj_close_prices)
    
    
    ### -- DecisionTree Regressor
    use_DTs = False
    if use_DTs:
        reg = decisionTree_model(X_train, y_train)
        y_pred = reg.predict(X_test)
        print "Classifier:\n{}".format(reg.get_params)
        print "DTs - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
        print '-------------------------------------------------------------------------'

    ### -- SVR Regressor
    use_SVR = False
    if use_SVR:
        reg = svr_model(X_train, y_train)
        y_pred = reg.predict(X_test)
        print "Classifier:\n{}".format(reg.get_params)
        print "SVC - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
        print '-------------------------------------------------------------------------'

    ### -- RandomForest Regressor
    use_RFR = False
    if use_RFR:
        reg = ensemble_model(X_train, y_train)
        y_pred = reg.predict(X_test)
        print "Classifier:\n{}".format(reg.get_params)
        print "RandomForestRegressor - R2_Score: {}".format(evaluation_metric(y_test,y_pred))
        print '-------------------------------------------------------------------------'


    if True:
         
        adj_close_prices = np.reshape(df['adj_close'],-1)
        volume = np.reshape(df['volume'],-1)
        close_price = np.reshape(df['close'],-1)
        open_price = np.reshape(df['open'],-1)

        data_len = len(df['adj_close'])
        dates = np.arange(data_len)
        dates = np.reshape(dates,(-1,1))

        #Predicting each one
            ##Predicting Volume

        X_train,X_test,y_train,y_test = split_data(dates,volume)
        volReg = ensemble_model(X_train,y_train)
        
        X_train,X_test,y_train,y_test = split_data(dates,close_price)
        closeReg = ensemble_model(X_train,y_train)
        
        X_train,X_test,y_train,y_test = split_data(dates,open_price)
        openReg = ensemble_model(X_train,y_train)
        
        X_train,X_test,y_train,y_test = split_data(dates,adj_close_prices)
        priceReg = ensemble_model(X_train,y_train)

        plt.figure()
        plt.subplot(411)
        plt.scatter(dates,volume,color='black',label='Data')
        plt.plot(dates,volReg.predict(dates),color='green',label='Pred')
        plt.xlabel('Period[days]')
        plt.ylabel('Volume')
        plt.xlim(xmin=0)
        plt.semilogy()
        plt.legend()
        plt.grid()

        
        plt.subplot(412)
        plt.scatter(dates,close_price,color='black',label='Data')
        plt.plot(dates,closeReg.predict(dates),color='blue',label='Pred')
        plt.xlabel('Period[days]')
        plt.ylabel('Close Price')
        plt.xlim(xmin=0)
        plt.legend()
        plt.grid()

        
        plt.subplot(413)
        plt.scatter(dates,open_price,color='black',label='Data')
        plt.plot(dates,openReg.predict(dates),color='red',label='Pred')
        plt.xlabel('Period[days]')
        plt.ylabel('Open Price')
        plt.xlim(xmin=0)
        plt.legend()
        plt.grid()

#        plt.figure()
        plt.subplot(414)
        plt.scatter(dates,adj_close_prices,color='black',label='Data')
        plt.plot(dates,priceReg.predict(dates),color='yellow',label='Pred')
        plt.xlabel('Period[days]')
        plt.ylabel('Adjustment Close Price')
        plt.xlim(xmin=0)
        plt.legend()
        plt.grid()

        plt.figure()
        dates = np.asarray(range(data_len-2,data_len+5))
        dates = np.reshape(dates,(-1,1))
#        tenDays = np.asarray(range(dates[-10],dates[-1]+11))
#        tenDays = np.reshape(tenDays,(-1,1))
#        print tenDays
#        plt.scatter(dates[-10:],adj_close_prices[-10:],color='black',label='Adj Close Price Data')
        plt.plot(dates,priceReg.predict(dates),color='black',label='Adj Close Price Pred')
        plt.xlabel('Period[days]')
        plt.ylabel('Price')
        plt.title('Last 10 days and next 10 days')
        plt.xlim(xmin=0)
        plt.legend()
        plt.grid()

        y_pred = priceReg.predict(dates)
        print y_pred[-10:]

        plt.show()
