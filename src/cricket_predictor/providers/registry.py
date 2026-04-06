from cricket_predictor.config.settings import Settings
from cricket_predictor.providers.base import LiveDataProvider
from cricket_predictor.providers.http_provider import HttpLiveDataProvider
from cricket_predictor.providers.ipl_csv_provider import IplCsvDataProvider
from cricket_predictor.providers.mock_provider import MockLiveDataProvider


def build_live_provider(settings: Settings) -> LiveDataProvider:
    if settings.live_provider == "http" and settings.live_provider_base_url:
        return HttpLiveDataProvider(settings.live_provider_base_url)
    if settings.live_provider == "ipl_csv" and settings.ipl_csv_data_dir:
        return IplCsvDataProvider(settings.ipl_csv_data_dir)
    return MockLiveDataProvider()
