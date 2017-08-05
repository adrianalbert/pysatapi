# satapi

This package contains functionality to download satellite imagery from common providers that make their data available (either commercially or for free) via online APIs. The emphasis is on acquiring data for analysis and experimentation, and for training machine learning algorithms.

Currently, the commercial satellite data providers supported are:

* [Google Maps Static API](https://developers.google.com/maps/documentation/static-maps/). Example usage can be found in [this notebook](examples/Google-Maps-Static-API-Example.ipynb)
* [Planet Labs](https://www.planet.com/docs/reference/). Example usage can be found in [this notebook](examples/Test-custom-Planet-Labs-API-client.ipynb)

## Installation

The `satapi` package is available on `pypi` and can be easily installed via `pip`. For this, type the following comand in a terminal:
```bash
pip install satapi
```

Alternatively, you can obtain the latest `master` branch on GitHub and install manually:
```bash
git clone https://github.com/adrianalbert/satapi.git
cd satapi/
python setup.py install
```


