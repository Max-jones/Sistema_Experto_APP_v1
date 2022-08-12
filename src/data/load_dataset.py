import pandas as pd


def load_data(path):
    '''
        ARGS: path to the local .csv file
        Load data and search for the Date_Time column to index the dataframe by a datetime

    '''

    data = pd.read_csv(path, sep=None, engine='python')
    try:

        data["Date_Time"] = pd.to_datetime(data["Date_Time"])
        data.set_index("Date_Time", inplace=True)
        chile = pytz.timezone("Chile/Continental")
        data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
        return data
    except:
        try:
            data['Date_Time'] = pd.to_datetime(data["Date_Time"])
            data.set_index("Date_Time", inplace=True)
            chile = pytz.timezone("Chile/Continental")
            data.index = data.index.tz_localize(pytz.utc).tz_convert(chile)
            return data
        except:
            return data
