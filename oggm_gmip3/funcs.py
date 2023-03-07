# Builtins
import logging

# External libs
import numpy as np
import pandas as pd
import xarray as xr
import cftime

# Locals
from oggm import entity_task
from oggm import utils, cfg
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   RandomMassBalance)

from oggm.shop.gcm_climate import process_gcm_data
from oggm.core.flowline import FileModel, flowline_model_run
from oggm.exceptions import InvalidParamsError


# Module logger
log = logging.getLogger(__name__)

isimip_url = 'https://cluster.klima.uni-bremen.de/~lschuster/isimip3b/'

shuffle_file = 'https://raw.githubusercontent.com/GlacierMIP/GlacierMIP3/main/shuffling/shuffled_years_GlacierMIP3.csv'


def fetch_isimip_file(var, gcm, ssp):

    if var not in ['tas', 'pr']:
        raise InvalidParamsError(f'var {var} not correct')

    if gcm not in ['gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0', 'ukesm1-0-ll']:
        raise InvalidParamsError(f'gcb {gcm} not correct')

    if ssp not in ['historical', 'ssp126', 'ssp370', 'ssp585']:
        raise InvalidParamsError(f'gcb {gcm} not correct')

    rea = 'r1i1p1f2' if gcm == 'ukesm1-0-ll' else 'r1i1p1f1'
    tperiod = '1850_2014' if ssp == 'historical' else '2015_2100'

    fp = isimip_url + f'/isimip3b_{var}Adjust_monthly/{gcm}_{rea}_w5e5_{ssp}_{var}Adjust_global_monthly_{tperiod}.nc'
    return utils.file_downloader(fp)


@entity_task(log, writes=['gcm_data'])
def process_isimip_data(gdir, filesuffix='', gcm='', ssp='',
                        year_range=('1981', '2010'), **kwargs):
    """Read, process and store the isimip climate data for this glacier.

    It stores the data in a format that can be used by the OGGM mass balance
    model and in the glacier directory.

    Parameters
    ----------
    """

    # Glacier location
    glon = gdir.cenlon
    glat = gdir.cenlat

    # Get the path of GCM temperature & precipitation data
    fpath_temp = fetch_isimip_file('tas', gcm, ssp)
    fpath_precip = fetch_isimip_file('pr', gcm, ssp)

    # Read the GCM files
    with xr.open_dataset(fpath_temp, use_cftime=True) as tempds, \
            xr.open_dataset(fpath_precip, use_cftime=True) as precipds:

        # Check longitude conventions
        assert tempds.lon.min() < -170
        assert precipds.lon.min() < -170

        # Take the closest to the glacier
        temp = tempds.tasAdjust.sel(lat=glat, lon=glon, method='nearest')
        precip = precipds.prAdjust.sel(lat=glat, lon=glon, method='nearest')

    if ssp != 'historical':
        # For this we need to add the timeseries upfront

        # Get the path of GCM temperature & precipitation data
        fpath_temp = fetch_isimip_file('tas', gcm, 'historical')
        fpath_precip = fetch_isimip_file('pr', gcm, 'historical')

        # Read the GCM files
        with xr.open_dataset(fpath_temp, use_cftime=True) as tempds, \
                xr.open_dataset(fpath_precip, use_cftime=True) as precipds:

            # Check longitude conventions
            assert tempds.lon.min() < -170
            assert precipds.lon.min() < -170

            # Take the closest to the glacier
            selt = slice(str(int(year_range[0]) - 1), '2014')
            _temp = tempds.tasAdjust.sel(time=selt)
            _temp = _temp.sel(lat=glat, lon=glon, method='nearest')
            _precip = precipds.prAdjust.sel(time=selt)
            _precip = _precip.sel(lat=glat, lon=glon, method='nearest')

        temp = xr.concat([_temp, temp], dim='time')
        precip = xr.concat([_precip, precip], dim='time')

    # Convert kg m-2 s-1 to mm mth-1 => 1 kg m-2 = 1 mm !!!
    assert 'kg m-2 s-1' in precip.units, 'Precip units not understood'

    ny, r = divmod(len(temp), 12)
    assert r == 0
    dimo = [cfg.DAYS_IN_MONTH[m - 1] for m in temp['time.month']]
    precip = precip * dimo * (60 * 60 * 24)

    process_gcm_data(gdir, filesuffix=filesuffix, prcp=precip, temp=temp,
                     source=filesuffix, year_range=year_range,
                     **kwargs)

@entity_task(log, writes=['gcm_data'])
def random_isimip_series(gdir, period='', input_filesuffix='', output_filesuffix=''):
    """Create the imposed timeseries according to the gmip protocol.

    This may be useful for some models, but "old" OGGM will use a custom
    random model for optimisation.

    Parameters
    ----------
    """

    df = pd.read_csv(utils.file_downloader(shuffle_file), index_col=0)
    s = df[period]

    fp = gdir.get_filepath('gcm_data', filesuffix=input_filesuffix)
    with xr.open_dataset(fp) as ds:
        ds = ds.load()

    temp = []
    prcp = []
    optim = dict()
    for y in s.values.astype(str):
        if y not in optim:
            tds = ds.sel(time=y)
            optim[y] = {'temp': tds.temp.data,
                        'prcp': tds.prcp.data}
        temp = np.append(temp, optim[y]['temp'])
        prcp = np.append(prcp, optim[y]['prcp'])

    t = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * len(s))
    t = cftime.num2date(np.append([0], t[:-1]), 'days since 0000-01-01 00:00:00',
                        calendar='noleap')

    gdir.write_monthly_climate_file(t, prcp, temp,
                                    float(ds.ref_hgt),
                                    ds.ref_pix_lon, ds.ref_pix_lat,
                                    calendar='noleap',
                                    file_name='gcm_data',
                                    source=ds.climate_source,
                                    filesuffix=output_filesuffix)


@entity_task(log)
def run_isimip_climate(gdir, period='',
                       nyears=None,
                       climate_filename='gcm_data',
                       climate_input_filesuffix='',
                       output_filesuffix='',
                       **kwargs):
    """Runs the random isimip mass-balance model for a given number of years.

    Parameters
    ----------
    """

    df = pd.read_csv(utils.file_downloader(shuffle_file), index_col=0)

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     prescribe_years=df[period],
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix)

    if nyears is not None:
        ye = df.index[nyears]
    else:
        ye = df.index[-1] + 1

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=df.index[0], ye=ye,
                              store_monthly_step=False,
                              store_model_geometry=False,
                              **kwargs)
