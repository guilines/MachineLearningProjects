import numpy as np
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))




def get_value(symbol,initDate=None,finalDate=None):
    ''' Returns a list containing all stock values of a company
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



df = get_value('AAPL')
print df.head()
sns.joinplot(data=df)

