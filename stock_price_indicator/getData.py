import sqlite3
import yahoo_finance as yhoo

#Create SQL Table

def create_tables(cur):
    ''' Create tables if they does not exist'''

    cur.execute(''' CREATE TABLE IF NOT EXISTS Companies (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, 
            symbol TEXT UNIQUE,
            name TEXT UNIQUE
            )''')
    
    
    cur.execute(''' CREATE TABLE IF NOT EXISTS Period (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            date DATE UNIQUE
            )''')
    
    cur.execute(''' CREATE TABLE IF NOT EXISTS Data (
            symbol_id INTEGER,
            period_id INTEGER,
            open INTEGER,
            close INTEGER,
            low INTEGER,
            high INTEGER,
            volume INTEGER,
            adj_close INTEGER,
            PRIMARY KEY (symbol_id,period_id)
            )''')
    

#--------------------------------------------------------------------
## Running code
#--------------------------------------------------------------------
if __name__ == '__main__':
    conn = sqlite3.connect('stock_data.sqlite')
    cur = conn.cursor()
    
    create_tables(cur)
    
    cSymbol = raw_input('Enter company symbol: ')
    obj = yhoo.Share(cSymbol)
   
    
    cName = obj.get_name()
    if cName is None:
        print 'Wrong symbol: {}'.format(cSymbol)
        exit()
   
    print '{} - {}'.format(cSymbol,cName)
    cYear = raw_input('Enter a year <YYYY>: ')
    initDate = cYear+'-01-01'
    finalDate = cYear+'-12-31'

    try:
        all_values = obj.get_historical(initDate,finalDate)
    
    except:
        print '{} is not avalibale for {}.'.format(cName,cYear)
        exit()
    

    cur.execute(''' INSERT OR IGNORE INTO Companies (symbol,name)
            VALUES (?,?)''', (cSymbol,cName))
    cur.execute('SELECT id FROM Companies WHERE symbol = ?', (cSymbol,))
    symbol_id = cur.fetchone()[0]

    for values in all_values:
        cur.execute(''' INSERT OR IGNORE INTO Period (date)
                    VALUES (?)''', (values['Date'],))
        cur.execute('SELECT id FROM Period WHERE date = ?', (values['Date'],))
        period_id = cur.fetchone()[0]

        cur.execute(''' INSERT OR IGNORE INTO Data 
                    (symbol_id,period_id,open,close,low,
                    high,volume,adj_close)
                    VALUES (?,?,?,?,?,?,?,?)''', (symbol_id,
                    period_id,values['Open'],values['Close'],
                    values['Low'],values['High'],
                    values['Volume'],values['Adj_Close']))

        conn.commit()

    print 'Done'

