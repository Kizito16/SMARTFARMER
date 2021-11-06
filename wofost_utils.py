from pathlib import Path
import datetime as dt
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pcse.fileinput import CABOFileReader, YAMLCropDataProvider, CABOWeatherDataProvider,YAMLAgroManagementReader, ExcelWeatherDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.models import Wofost71_WLP_FD, Wofost71_PP

import ipywidgets.widgets as widgets
from ipywidgets import interact, interactive, fixed


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

data_dir = os.path.join(os.getcwd(), "data")

agromanagement_contents = """
Version: 1.0
AgroManagement:
- {year:d}-01-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: '{variety}'
        crop_start_date: {crop_start_date}
        crop_start_type: sowing
        crop_end_date: {crop_end_date}
        crop_end_type: harvest
        max_duration: 150
    TimedEvents: null
    StateEvents: null
"""
WOFOST_PARAMETERS = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST',
                'TWRT', 'TRA', 'RD', 'SM']
LABELS = ["Development stage [-]", "LAI [m2/m2]",
                 "Total Biomass [kg/ha]",
                 "Total Storage Organ Weight [kg/ha]",
                 "Total Leaves Weight [kg/ha]",
                 "Total Stems Weight [kg/ha]",
                 "Total Root Weight [kg/ha]",
                 "Transpiration rate [cm/d]",
                 "Rooting depth [cm]",
                 "Soil moisture [cm3/cm3]"]
WOFOST_LABELS = dict(zip(WOFOST_PARAMETERS, LABELS))

def set_up_wofost(crop_start_date, crop_end_date,
                  meteofile, crop, variety, soil,
                  wav=10, co2=360, rdmsol=100.):
    cropdata = YAMLCropDataProvider(os.path.join(data_dir, "crop"))
    cropdata.set_active_crop(crop, variety)
    soilfile = os.path.join(data_dir, 'soil', soil)
    soildata = CABOFileReader(soilfile)
    soildata['RDMSOL'] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)
    with open("temporal.amgt", 'w') as fp:
        fp.write(agromanagement_contents.format(year=crop_start_date.year,
                        crop=crop, variety=variety, crop_start_date=crop_start_date,
                        crop_end_date=crop_end_date))
    agromanagement = YAMLAgroManagementReader("temporal.amgt")

    meteo = os.path.join(data_dir, 'meteo', meteofile)
    wdp = ExcelWeatherDataProvider(meteo)
    return parameters, agromanagement, wdp

def run_wofost(parameters, agromanagement, wdp, potential=False):
    if potential:
        wofsim = Wofost71_PP(parameters, wdp, agromanagement)
    else:        
        wofsim = Wofost71_WLP_FD(parameters, wdp, agromanagement)
    print(parameters)
    wofsim.run_till_terminate()
    df_results = pd.DataFrame(wofsim.get_output())
    df_results = df_results.set_index("day")
    return df_results, wofsim

def change_sowing_date(start_sowing, end_sowing, meteo, crop, variety, soil,
                        n_days=10):
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, squeeze=True,
                           figsize=(16,16))
    axs = axs.flatten()
    sowing_date = start_sowing
    
    while sowing_date < end_sowing:
        parameters, agromanagement, wdp = set_up_wofost(
                sowing_date, sowing_date + dt.timedelta(days=150),
                meteo, crop, variety, soil)
        
        df_results, simulator = run_wofost(parameters, agromanagement, wdp,
                                           potential=False)
        sowing_date += dt.timedelta(days=n_days)
        for j, p in enumerate(WOFOST_PARAMETERS):
            axs[j].plot_date(df_results.index, df_results[p], '-')
            axs[j].set_ylabel(WOFOST_LABELS[p], fontsize=8)
    # fig.autofmt_xdate()

    plt.gcf().autofmt_xdate()
    plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')
    plt.xlim(start_sowing, None)
    axs[8].set_xlabel("Time [d]")
    axs[9].set_xlabel("Time [d]")
    
def change_sowing_slider():
    interact(change_sowing_date,
            start_sowing=widgets.DatePicker(value=dt.date(2020, 8, 15),
            description="Earliest possible sowing date"),
            end_sowing=widgets.DatePicker(value=dt.date(2020, 10, 12),
            description="Latest possible sowing date"),
            meteo=fixed('chwele.xls'),
            n_days = widgets.IntSlider(min=1, max=20, value=10),
            crop=fixed('soybean'),
            variety=fixed('Soybean_901'),
            soil=fixed('M04.AWC'),
            mgmt=fixed("soybean_calendar.agro"))

def wofost_parameter_sweep_func(crop_start_date = dt.date(2020, 9, 15),
                    crop_end_date = dt.date(2021, 1, 12),
                  span=40.0, tdwi=20., tsum1=750., tsum2=859.,
                  tsumem=70,rgrlai=0.05,cvo=0.05, cvl=0.05,
                  meteo='chwele.xls', crop = 'soybean',
                variety = 'Soybean_901', soil = 'M04.AWC',
                  wav=10, co2=360, rdmsol=100., potential=False):
    cropdata = YAMLCropDataProvider(os.path.join(data_dir, "crop"))
    cropdata.set_active_crop(crop, variety)
    soilfile = os.path.join(data_dir, 'soil', soil)
    soildata = CABOFileReader(soilfile)
    soildata['RDMSOL'] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)
    for p, v in zip(["SPAN", "TSUM1", "TSUM2", "TSUMEM", "TDWI", "RGRLAI", "CVO", "CVL"],
                    [span, tsum1, tsum2, tsumem, tdwi, rgrlai, cvo, cvl]):
        parameters.set_override(p, v, check=True) 
    with open("temporal.amgt", 'w') as fp:
        fp.write(agromanagement_contents.format(year=crop_start_date.year,
                        crop=crop, variety=variety, crop_start_date=crop_start_date,
                        crop_end_date=crop_end_date))
    agromanagement = YAMLAgroManagementReader("temporal.amgt")

    meteo = os.path.join(data_dir, 'meteo', meteo)
    wdp = ExcelWeatherDataProvider(meteo)
    df_results, simulator = run_wofost(parameters, agromanagement, wdp,
                                           potential=potential)
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, squeeze=True,
                           figsize=(16,16))
    axs = axs.flatten()
    for j, p in enumerate(WOFOST_PARAMETERS):
        axs[j].plot_date(df_results.index, df_results[p], '-')
        axs[j].set_ylabel(WOFOST_LABELS[p], fontsize=8)

    plt.gcf().autofmt_xdate()
    plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')
    axs[8].set_xlabel("Time [d]")
    axs[9].set_xlabel("Time [d]")

    key = f"span_{span}-tdwi_{tdwi}-tsum1_{tsum1}-tsum2_{tsum2}-tsumem_{tsumem}"
    key += f"-rgrlai_{rgrlai}-wav_{wav}-cvo_{cvo}-cvl_{cvl}"

    if potential:
        key += "-POT.csv"
    else:
        key += "-LIM.csv"
    
    df_results.to_csv(key, encoding="utf-8", index=False)
    
def wofost_parameter_sweep():
    widgets.interact_manual(wofost_parameter_sweep_func,
                  crop_start_date=widgets.fixed(dt.date(2020, 9, 15)),
                  crop_end_date=widgets.fixed(dt.date(2021, 1, 12)),
                  span=widgets.FloatSlider(value=40.0, min=20, max=50),
                  cvo = widgets.FloatSlider(value=0.72, min=0.1, max=0.9, step=0.02),
                  cvl = widgets.FloatSlider(value=0.72, min=0.1, max=0.9, step=0.02),
                  tdwi=widgets.FloatSlider(value=20.0,min=5, max=50),
                  tsum1=widgets.FloatSlider(value=750.0,min=100, max=1500),
                  tsum2=widgets.FloatSlider(value=859.0,min=100, max=1500),
                  tsumem=widgets.FloatSlider(value=70,min=10, max=200),
                  rgrlai=widgets.FloatSlider(value=0.05, min=0.001, max=0.3, step=0.01),
                  meteo=widgets.fixed("chwele.xls"),
                  crop=widgets.fixed('soybean'),
                  variety=widgets.fixed('Soybean_901'),
                  soil=widgets.fixed('M04.AWC'),
                  wav=widgets.FloatSlider(value=5, min=0, max=100),
                  co2=widgets.fixed(400),
                  rdmsol=widgets.fixed(100.),
                  potential=widgets.Checkbox(value=False, description='Potential mode',
                  icon='check'))