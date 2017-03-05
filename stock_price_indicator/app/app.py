import web
import simplejson as json
import numpy as np
import pandas
from datetime import datetime, timedelta

import modules.get_data as getData
import modules.predict_prices as predPrices


DATABASE_PATH = 'modules/database/stock_data.sqlite'
g_predObj = predPrices.PredStockPrices(DATABASE_PATH)



urls = ('/', 'messages')
render = web.template.render('templates/')

app = web.application(urls, globals())

req = web.form.Form(
                web.form.Textbox('', class_='textfield', id='textfield'),
                )

class messages:
    def GET(self):
        reqs = req()
        return render.index(reqs, "Your text goes here.")
        
    def POST(self):
        reqs = req()
        reqs.validates()
        if not reqs.value['operation']:
            print 'Failed to get operation'
            return False
        else:
            print 'Requested operation: {}'.format(reqs.value['operation'])
            if reqs.value['operation'] == 'reset':
                reqs =  _reset(reqs)
            elif reqs.value['operation'] == 'update_values_chart':
                reqs =  _update_values_chart(reqs)
            elif reqs.value['operation'] == 'set_method':
                reqs =  _set_method(reqs)
            elif reqs.value['operation'] == 'new_pred':
                reqs =  _new_pred(reqs)

        #s = reqs.value['textfield']
        return json.dumps(reqs)

def _reset(reqs):
    g_predObj = None
    g_predObj = predPrices.PredStockPrices(DATABASE_PATH)
    return True

    
    
    
def _pred_test(reqs):
    print 'Pred Test'
    #reqs["symbol"] = 'APPL'
    symbol = 'YHOO'
    predObj = predPrices.PredStockPrices(DATABASE_PATH)
    predObj.load_data(symbol)
    df = predObj.get_df()
    results = _get_company_values(df)

    res = {}
    res['chart1'] = results
    return res

def _update_values_chart(reqs):
    if 'period' in reqs.value:
        if not getData.updateData(DATABASE_PATH,reqs.value['symbol'],reqs.value['period']):
            return False

    predObj = predPrices.PredStockPrices(DATABASE_PATH)
    predObj.load_data(reqs.value['symbol'])
    df = predObj.get_df()
    return _get_company_values(df)

def _get_company_values(df):
    result = {}
    result['date'] =        df['date'].values.tolist()
    result['close_price'] = df['close'].values.tolist()
    result['open_price'] =  df['open'].values.tolist()
    result['adj_price'] =   df['adj_close'].values.tolist()
    result['volume'] =      df['volume'].values.tolist()
    result['company_name'] = df['name'][0]
    result['values'] = ['Date','Close Prices','Open Prices', 'Adj Close Price']

    result['vStr'] = [] 
    result['vStr'].append('Statistics for <font color="green">{}</font> dataset:'.format(result['company_name']))
    result['vStr'].append('Minimum price:<font color="green"> ${:,.2f}</font>'.format(np.min(result['adj_price'])))
    result['vStr'].append('Maximum price:<font color="green"> ${:,.2f}</font>'.format(np.max(result['adj_price'])))
    result['vStr'].append('Average price:<font color="green"> ${:,.2f}</font>'.format(np.average(result['adj_price'])))
    result['vStr'].append('Standart deviation of prices:<font color="green"> ${:,.2f}</font>'.format(np.std(result['adj_price'])))



    return result

def _set_method(reqs):
    global g_predObj
    #predObj = predPrices.PredStockPrices()
    g_predObj.load_data(reqs.value['symbol'])
    g_predObj.set_method(reqs.value['method'])
    y_pred = g_predObj.get_pred()
    r2_score = g_predObj.get_r2Score(y_pred)
    method = g_predObj.get_method()
    params = g_predObj.get_params()
    df = g_predObj.get_df()

    result = {}
    result['adj_price'] =   df['adj_close'].values.tolist()
    result['date'] =        df['date'].values.tolist()
    result['pred_values'] =   y_pred.tolist()
    result['company_name'] = df['name'][0]

    if r2_score > 0.7:
        color = 'green'
    else:
        color = 'red'

    result['vStr'] = [] 
    result['vStr'].append('Method used:<font color="green"> {}</font>'.format(method))
    result['vStr'].append('R2 Score:<font color="{}"> {}</font>'.format(color,r2_score))
    if params is not None:
        result['vStr'].append('Regressor:<font color="{}"> {}</font>'.format(color,params))

    #print result['vStr']
    #print y_pred
    #print y_pred.shape
    
    return result

def _new_pred(reqs):
    df = g_predObj.get_df()
    lenDate = len(df['date'])
    days = int(reqs.value['period'].split()[0])
    period = range(lenDate-1,lenDate+days)
    period = np.asarray(period)
    period = np.reshape(period,(-1,1))
    y_pred = g_predObj.get_pred(period)

    result = {}
    result['date'] = []
    result['date'].append('{}'.format(datetime.now().date()))
    for i in range(1,days+1):
        result['date'].append('{}'.format((datetime.now() + timedelta(days=i)).date()))
    result['pred_values'] =   y_pred.tolist()
    result['company_name'] = df['name'][0]

    
    return result



if __name__ == '__main__':
    app.run()

