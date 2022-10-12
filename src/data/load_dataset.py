import pandas as pd
import pytz


def load_data(path):
    '''
        ARGS: path to the local .csv file
        Load data and search for the Date_Time column to index the dataframe by a datetime

    '''

    data = pd.read_csv(path, sep=None, engine='python',encoding = 'utf-8-sig',parse_dates= True)
    data.dropna(axis="columns", how="any", inplace=True)
    try:

        data["Date_Time"] = pd.to_datetime(data["Date_Time"],dayfirst=True)
        data.set_index("Date_Time", inplace=True)
        chile = pytz.timezone("Chile/Continental")
        data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
        print('1')
        return data
    except:
        try:
            data['Date_Time'] = pd.to_datetime(data["Date_Time"],dayfirst=True)
            data.set_index("Date_Time", inplace=True)
            chile = pytz.timezone("Chile/Continental")
            data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
            print('2')
            return data
        except:
            try: 
                data['Date_Time'] = data['Date'] + ' ' + data['Time']
                data.drop(['Date','Time'],axis=1,inplace=True)
                data['Date_Time'] = pd.to_datetime(data["Date_Time"],dayfirst=True)
                data.set_index("Date_Time", inplace=True)
                chile = pytz.timezone("Chile/Continental")
                data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
                print('3')
                return data
            except:
                print('4')
                return data
