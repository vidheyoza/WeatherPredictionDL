import pandas as pd
import numpy as np


# Returns data in (X, y) format, with data of every day.
def data_from_dates():

    # Array of lat and long, used for referencing
    lat = [220, 223, 226, 229, 233, 236, 239]
    lon = [25, 28, 31, 34, 38, 41, 44, 47, 50, 53, 56, 59, 63, 66, 69, 72, 75, 78, 81]

    # Extract dates for which to create dataset
    dates = pd.read_csv('dataset/dates.csv', index_col=0, header=None)
    dates = np.array(dates.iloc[:,0].values)

    dates = np.array(dates)

    images = []  # array returned as X
    out = []  # array returned as y
    # create dataset from CSV file of each day
    for date in dates:
        (month,day,year) = date.split('/')
        date = date.replace('/', '-')

        # Take data for July to October (highest chances of precipitation)
        if int(month) < 7 or int(month) > 10:
            continue

        d = pd.read_csv('dataset/date-csv-headers/' + str(date) + '.csv', index_col=False)
        d = d.drop(['LAT', 'LONG', 'SOLAR'], axis=1)  # drop unnecessary columns
        d = d.iloc[:, :].values
        d = np.array(d)

        l = []
        m = []
        i = 0
        for la in lat:
            x = []
            y = []
            for lo in lon:
                x.append(d[i])
                y.append([d[i][3]])
                i += 1
            x = np.array(x)
            y = np.array(y)
            l.append(x)
            m.append(y)
        l = np.array(l)
        m = np.array(m)
        images.append(l)
        out.append(m)
    images = np.array(images)
    out = np.array(out)

    # Fix dimension of images for convolution operations
    images = np.pad(images, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    out = np.pad(out, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)

    # Standard scaling
    images[:, :, :, 0] = (np.array(images[:, :, :, 0]) - np.mean(images[:, :, :, 0])) / (1 + np.std(images[:, :, :, 0]))
    images[:, :, :, 1] = (np.array(images[:, :, :, 1]) - np.mean(images[:, :, :, 1])) / (1 + np.std(images[:, :, :, 1]))
    images[:, :, :, 2] = (np.array(images[:, :, :, 2]) - np.mean(images[:, :, :, 2])) / (1 + np.std(images[:, :, :, 2]))
    images[:, :, :, 3] = (np.array(images[:, :, :, 3]) - np.mean(images[:, :, :, 3])) / (1 + np.std(images[:, :, :, 3]))
    images[:, :, :, 4] = (np.array(images[:, :, :, 4]) - np.mean(images[:, :, :, 4])) / (1 + np.std(images[:, :, :, 4]))
    images[:, :, :, 5] = (np.array(images[:, :, :, 5]) - np.mean(images[:, :, :, 5])) / (1 + np.std(images[:, :, :, 5]))

    out[:, :, :, 0] = (np.array(out[:, :, :, 0]) - np.mean(out[:, :, :, 0])) / (1 + np.std(out[:, :, :, 0]))

    return images, out


# For each day in dataset, creates window backwards from t=0 of previous day images for X (for RNN), and
# corresponding forecast day forwards from t=0 for Y
def dataset_maker(window, forecast_day):
    images, out = data_from_dates()

    x = []
    y = []
    for i in range(window, len(images)-forecast_day):
        X = images[i-window:i, :]
        Y = out[i+forecast_day]
        x.append(X)
        y.append(Y)

    x = np.array(x)
    y = np.array(y)

    return x, y
