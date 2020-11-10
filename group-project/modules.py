from datetime import datetime, timedelta

def date2days(date):
    d1 = datetime(2020, 4, 12)
    month = int(date[:2])
    day = int(date[3:5])
    year = int(date[6:])
    d2 = datetime(year, month, day)
    return (d2-d1).days

def days2date(num):
    d1 = datetime(2020, 4, 12)
    d2 = d1 + timedelta(days=num)
    date = d2.strftime('%m-%d-%Y')
    return date