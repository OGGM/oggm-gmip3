import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import os
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import numpy as np
import xarray as xr
import pandas as pd

# Local imports
from oggm.core import flowline
from oggm import cfg, utils, workflow, tasks
import oggm_gmip3

prepro_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/match_geod_pergla'

def test_fetch_files():

    fp = oggm_gmip3.fetch_isimip_file('tas', 'gfdl-esm4', 'historical')
    assert os.path.exists(fp)

    fp = oggm_gmip3.fetch_isimip_file('pr', 'gfdl-esm4', 'historical')
    assert os.path.exists(fp)

    fp = oggm_gmip3.fetch_isimip_file('tas', 'ukesm1-0-ll', 'ssp585')
    assert os.path.exists(fp)

    fp = oggm_gmip3.fetch_isimip_file('pr', 'ukesm1-0-ll', 'ssp585')
    assert os.path.exists(fp)

class TestGmipRuns:

    def test_process_isimip(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['hydro_month_sh'] = 1

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897', 'RGI60-03.04384'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=160,
                                                  prepro_rgi_version='62')

        gcm = 'ukesm1-0-ll'
        ssp = 'ssp585'
        fsuff = f'{gcm}_{ssp}'
        workflow.execute_entity_task(oggm_gmip3.process_isimip_data, gdirs,
                                     filesuffix=fsuff, gcm=gcm, ssp=ssp)

        with xr.open_dataset(gdirs[0].get_filepath('gcm_data', filesuffix=fsuff)) as ds:
            ds = ds.load()

        assert ds['time.year'][0] == 1980
        assert ds['time.year'][-1] == 2100
        assert ds['time.month'][0] == 1
        assert ds['time.month'][-1] == 12
        assert (ds.temp.sel(time=slice('2080', '2099')).mean() >
                ds.temp.sel(time=slice('1980', '1999')).mean() + 5)  # !!!

        gcm = 'gfdl-esm4'
        ssp = 'historical'
        fsuff = f'{gcm}_{ssp}'
        workflow.execute_entity_task(oggm_gmip3.process_isimip_data, gdirs,
                                     filesuffix=fsuff, gcm=gcm, ssp=ssp)

        with xr.open_dataset(gdirs[0].get_filepath('gcm_data', filesuffix=fsuff)) as ds:
            ds = ds.load()
        assert ds['time.year'][0] == 1850
        assert ds['time.year'][-1] == 2014
        assert ds['time.month'][0] == 1
        assert ds['time.month'][-1] == 12

    def test_random_isimip_series(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['hydro_month_sh'] = 1

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897', 'RGI60-16.02207'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=160,
                                                  prepro_rgi_version='62')

        gcm = 'ipsl-cm6a-lr'
        ssp = 'ssp585'
        period = '2081-2100'
        fsuff = f'{gcm}_{ssp}'
        ofsuff = f'{gcm}_{ssp}_{period}'
        workflow.execute_entity_task(oggm_gmip3.process_isimip_data, gdirs,
                                     filesuffix=fsuff, gcm=gcm, ssp=ssp)

        workflow.execute_entity_task(oggm_gmip3.random_isimip_series, gdirs,
                                     period=period,
                                     input_filesuffix=fsuff,
                                     output_filesuffix=ofsuff)

        # HEF
        gdir = gdirs[0]
        fref = ('https://raw.githubusercontent.com/GlacierMIP/GlacierMIP3/main/shuffling/test_shuffling/'
                'test_RGI60-11.00897_ipsl-cm6a-lr_ssp585_tasAdjust_shuffled.csv')

        with xr.open_dataset(gdir.get_filepath('gcm_data', filesuffix=ofsuff)) as ds:
            ds = ds.load()
            ds = ds.resample(time='AS').mean()

        df = pd.read_csv(utils.file_downloader(fref), index_col=0)
        df['ours'] = ds.temp.data + 273.15

        assert df[['2081-2100', 'ours']].loc[:20].corr().iloc[0, 1] > 0.99
        assert df[['2081-2100', 'ours']].loc[-20:].corr().iloc[0, 1] > 0.99
        dfm = df[['2081-2100', 'ours']].mean()
        # Thats the bias correction
        assert_allclose(dfm.iloc[0], dfm.iloc[1], atol=1.5)

        # Shallap
        gdir = gdirs[1]
        fref = ('https://raw.githubusercontent.com/GlacierMIP/GlacierMIP3/main/shuffling/test_shuffling/'
                'test_RGI60-16.02207_ipsl-cm6a-lr_ssp585_tasAdjust_shuffled.csv')

        with xr.open_dataset(gdir.get_filepath('gcm_data', filesuffix=ofsuff)) as ds:
            ds = ds.load()
            ds = ds.resample(time='AS').mean()

        df = pd.read_csv(utils.file_downloader(fref), index_col=0)
        df['ours'] = ds.temp.data + 273.15

        assert df[['2081-2100', 'ours']].loc[:20].corr().iloc[0, 1] > 0.99
        assert df[['2081-2100', 'ours']].loc[-20:].corr().iloc[0, 1] > 0.99
        dfm = df[['2081-2100', 'ours']].mean()
        # Thats the bias correction
        assert_allclose(dfm.iloc[0], dfm.iloc[1], atol=5)

    def test_run_isimip(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir
        cfg.PARAMS['hydro_month_nh'] = 1
        cfg.PARAMS['hydro_month_sh'] = 1

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897', 'RGI60-16.02207'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=160,
                                                  prepro_rgi_version='62')

        nyears = 500
        gcm = 'ipsl-cm6a-lr'
        ssp = 'ssp585'
        period = '1995-2014'
        fsuff = f'{gcm}_{ssp}'
        ofsuff = f'{gcm}_{ssp}_{period}'
        workflow.execute_entity_task(oggm_gmip3.process_isimip_data, gdirs,
                                     filesuffix=fsuff, gcm=gcm, ssp=ssp)

        workflow.execute_entity_task(oggm_gmip3.run_isimip_climate, gdirs,
                                     nyears=nyears,
                                     period=period,
                                     climate_input_filesuffix=fsuff,
                                     output_filesuffix=ofsuff)

        # The above should be equivalent as below
        workflow.execute_entity_task(oggm_gmip3.random_isimip_series, gdirs,
                                     period=period,
                                     input_filesuffix=fsuff,
                                     output_filesuffix=ofsuff)

        workflow.execute_entity_task(flowline.run_from_climate_data, gdirs,
                                     climate_filename='gcm_data',
                                     ys=0, ye=nyears,
                                     climate_input_filesuffix=ofsuff,
                                     output_filesuffix=ofsuff + '_alt')

        # HEF
        gdir = gdirs[0]
        fref = ('https://raw.githubusercontent.com/GlacierMIP/GlacierMIP3/main/shuffling/test_shuffling/'
                'test_RGI60-11.00897_ipsl-cm6a-lr_ssp585_tasAdjust_shuffled.csv')

        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=ofsuff+'_alt')) as ds:
            ds_alt = ds.load()
        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=ofsuff)) as ds:
            ds = ds.load()

        mbts = ds.volume_m3.to_series()
        mbts = mbts.iloc[1:].values - mbts.iloc[:-1]
        mbts /= ds.area_m2.to_series().iloc[:-1]

        df = pd.read_csv(utils.file_downloader(fref), index_col=0).loc[:nyears]
        df['ours'] = mbts
        # These correlate quite a lot
        assert df.iloc[-50:][[period, 'ours']].corr().iloc[0, 1] < -0.8

        mbts = ds.volume_m3.to_series()
        mbts_alt = ds_alt.volume_m3.to_series()
        assert_allclose(mbts, mbts_alt)
