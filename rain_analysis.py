import pandas as pd
import numpy as np
import seaborn as sns
import calmap
import calendar

def yearly_summary(ds):
    
    data = ds.reset_index().rename(columns = {'day':'date', 'rainfall':'rain'}, inplace = False)

    # remove the bad samples
    data = data[data['rain'] > -500]

    # Assign the "day" to every date entry
    data['day'] = data['date']#.apply(lambda x: x.date())

    # Mark the month for each entry so we can look at monthly patterns
    data['month'] = data['date'].apply(lambda x: x.month)

    # If there's any rain at all, mark that!
    data['raining'] = data['rain'] > 0.0

    # Create a daily data set distinguishing between rainy and dry days:
    daily_data = data.groupby('day')['raining'].any()
    daily_data = pd.DataFrame(daily_data).reset_index()

    # Group by month for display - monthly data set for plots.
    daily_data['month'] = daily_data['day'].apply(lambda x: x.month)
    monthly= daily_data.groupby('month')['raining'].value_counts().rename('Days').reset_index() ## we do the rename because we have a duplicate
    monthly.rename(columns={'raining':'Rainy'}, inplace=True)
    monthly.replace({"Rainy": {True: "Wet", False:"Dry"}}, inplace=True)    
    monthly['month_name'] = monthly['month'].apply(lambda x: calendar.month_abbr[x])

    # Get aggregate stats for each day in the dataset on rain in general - for heatmaps.
    rainy_days = data.groupby(['day']).agg({
            "rain": [("rain", lambda x: (x > 2.0).any()),
                     ("rain_amount", "sum")]
            })
    # clean up the aggregated data to a more easily analysed set:
    rainy_days.reset_index(drop=False, inplace=True) # remove the 'day' as the index
    rainy_days.rename(columns={"":"date"}, inplace=True) # The old index column didn't have a name - add "date" as name
    rainy_days.columns = rainy_days.columns.droplevel(level=0) # The aggregation left us with a multi-index
                                                               # Remove the top level of this index.
    rainy_days['rain'] = rainy_days['rain'].astype(bool)       # Change the "rain" column to True/False values

    # Add the number of rainy hours per day to the rainy_days dataset.
    temp = data.groupby(["day"])['raining'].any()
    temp = temp.groupby(level=[0]).sum().reset_index()
    #temp.rename(columns={'raining': 'hours_raining'}, inplace=True)
    #temp['day'] = temp['day'].apply(lambda x: x.date())
    rainy_days = rainy_days.merge(temp, left_on='date', right_on='day', how='left')
    rainy_days.drop('day', axis=1, inplace=True)
    
    return rainy_days, monthly

def get_seasonal_stats(df):
    rainy_days = df

    # Classify into long rain (LR) or short rain (SR) (assuming LR period is March-April-May and SR Period is Oct-Nov-Dec)
    rainy_days['LR'] = rainy_days['date'].apply(lambda x: x.month>= 3 and x.month<= 5)
    rainy_days['SR'] = rainy_days['date'].apply(lambda x: x.month>= 10 and x.month<= 12)
    rainy_days['Week'] = rainy_days['date'].apply(lambda x: x.isocalendar()[1]) # get week of month

    # Get Summary dictionaries for rainy days with >2mm
    rainy_days_dict = {
        "Annual": rainy_days['rain'].sum(),
        "LR": rainy_days.query('LR ==True')['rain'].sum(),
        "SR": rainy_days.query('SR ==True')['rain'].sum()
    }

    # Get weekly summary
    weekly_summary = rainy_days.groupby(["Week", "LR", "SR"])['rain_amount'].sum().reset_index()
    weekly_summary["above_10"] = weekly_summary['rain_amount'].apply(lambda x: x>10)

    weekly_rfall_dict = {
        "Annual": round(rainy_days['rain_amount'].sum()/(len(rainy_days.Week.unique())), 0),
        "LR": round(rainy_days.query('LR ==True')['rain_amount'].sum()/(len(rainy_days.query('LR ==True').Week.unique())), 0),
        "SR": round(rainy_days.query('SR ==True')['rain_amount'].sum()/(len(rainy_days.query('LR ==True').Week.unique())), 0)
    }

    weekly_rfall_above_10 = {
            "Annual": weekly_summary.above_10[weekly_summary.above_10 == True].count(),
            "LR": weekly_summary.query('LR ==True').above_10[weekly_summary.above_10 == True].count(),
            "SR": weekly_summary.query('SR ==True').above_10[weekly_summary.above_10 == True].count()
        }

    total_rainfall = {
            "Annual": round(rainy_days.rain_amount.sum(),2),
            "LR": round(rainy_days.query('LR ==True').rain_amount.sum(),2),
            "SR": round(rainy_days.query('SR ==True').rain_amount.sum(),2)
        }
    merged_df = rainy_days_dict | weekly_rfall_dict

    #Summarize dictionaries
    df1 = pd.DataFrame([weekly_rfall_dict])
    df2 = pd.DataFrame([rainy_days_dict])
    df3 = pd. DataFrame([weekly_rfall_above_10])
    df4 = pd. DataFrame([total_rainfall])

    frames = [df4, df2, df1,df3]

    result = pd.concat(frames).reset_index().drop('index', axis=1, inplace= False)
    df_summ = pd.DataFrame({'Variable':['Rainfall amount','Rainy days with >2 mm', 'Average Weekly rainfall (mm)', 'Number of weeks >10 mm'],
                               'Annual':list(result.Annual), 
                               'Long Rain':list(result.LR),
                              'Short Rain':list(result.SR)})
    return df_summ

    
    