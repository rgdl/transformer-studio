from itertools import product

import requests
import pytest

DOMAIN = "transformer-studio.com"


def test_http() -> None:
    _test_url(f"http://{DOMAIN}")


def test_https() -> None:
    _test_url(f"https://{DOMAIN}")


def test_www_and_http() -> None:
    _test_url(f"http://www.{DOMAIN}")


def test_www_and_https() -> None:
    _test_url(f"https://www.{DOMAIN}")


def _test_url(url: str) -> None:
    headers = {"user-agent": "test"}

    with requests.Session() as session:
        response = session.get(url, headers=headers)

    assert response.status_code == 200, f"Failed for {url}"
