"""
WorldQuant Brain API Wrapper
Credentials are loaded from a .env file (WQ_EMAIL / WQ_PASSWORD).
Session cookies are persisted to avoid re-login (and captcha) on every worker spawn.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
BASE = "https://api.worldquantbrain.com"
DEFAULT_COOKIE_PATH = Path("db/wq_cookies.json")


class WQClient:
    def __init__(
        self,
        env_file: str = ".env",
        cookie_path: str | Path | None = None,
    ):
        """
        Reads WQ_EMAIL and WQ_PASSWORD from env_file (default .env).
        Already-set environment variables take precedence.
        Reuses cookies from cookie_path when still valid.
        """
        load_dotenv(env_file, override=False)
        email = os.environ.get("WQ_EMAIL")
        password = os.environ.get("WQ_PASSWORD")
        if not email or not password:
            raise OSError("WQ_EMAIL and WQ_PASSWORD must be set in the environment or in .env")
        self._email = email
        self._password = password
        self._cookie_path = Path(
            cookie_path or os.environ.get("WQ_COOKIE_PATH") or DEFAULT_COOKIE_PATH
        )
        self.session = requests.Session()
        self.session.auth = (self._email, self._password)
        if self._load_cookies() and self._session_alive():
            logger.info("Reusing saved WQ session from %s", self._cookie_path)
        else:
            self.login()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self):
        self.session.auth = (self._email, self._password)
        r = self.session.post(f"{BASE}/authentication")
        if r.status_code not in (200, 201):
            body = {}
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text}
            if "inquiry" in body:
                input(
                    f"Complete biometric auth at {r.url}/persona?inquiry={body['inquiry']}"
                    " then press Enter..."
                )
                self.session.post(f"{r.url}/persona", json=body)
            elif "captcha" in body:
                raise RuntimeError(
                    "WQ login requires captcha (too many auth attempts). "
                    "Sign in once at https://platform.worldquantbrain.com, wait a few "
                    f"minutes, then restart the API server so it can save cookies to "
                    f"{self._cookie_path}. Improve workers will reuse that session."
                )
            else:
                raise RuntimeError(f"Login failed: {body}")
        self._save_cookies()
        logger.info("Logged in to WorldQuant Brain")

    def _session_alive(self) -> bool:
        try:
            r = self.session.get(f"{BASE}/authentication", timeout=30)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _load_cookies(self) -> bool:
        if not self._cookie_path.exists():
            return False
        try:
            raw = json.loads(self._cookie_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read cookie jar %s: %s", self._cookie_path, exc)
            return False
        if not isinstance(raw, list):
            return False
        for c in raw:
            if not isinstance(c, dict) or "name" not in c or "value" not in c:
                continue
            self.session.cookies.set(
                c["name"],
                c["value"],
                domain=c.get("domain") or None,
                path=c.get("path") or "/",
            )
        return True

    def _save_cookies(self) -> None:
        try:
            self._cookie_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path,
                }
                for c in self.session.cookies
            ]
            self._cookie_path.write_text(json.dumps(payload))
        except OSError as exc:
            logger.warning("Could not save cookie jar %s: %s", self._cookie_path, exc)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_operators(self) -> list:
        """
        List all available operators.

        Returns:
        [{
            'name': 'add',
            'category': 'Arithmetic',
            'scope': ['REGULAR'],
            'definition': 'add(x, y, filter = false), x + y',
            'description': 'Adds two or more inputs element wise. Set filter=true to treat NaNs as 0 before summing.',
            'documentation': '/operators/add', 'level': 'ALL'
        }, ...]
        """
        return self._get(f"{BASE}/operators").json()

    def get_data_fields(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        dataset_id: str = "",
        search: str = "",
    ) -> list:
        """
        Paginate through all available data fields.

        Returns:
        [{
            'id': 'year_three_amortization_expense_intangibles',
            'description': 'Amortization expense for finite-lived intangible assets in the third year.',
            'dataset': {'id': 'fundamental2', 'name': 'Report Footnotes'},
            'category': {'id': 'fundamental', 'name': 'Fundamental'},
            'subcategory': {'id': 'fundamental-footnotes', 'name': 'Footnotes'},
            'region': 'USA', 'delay': 1, 'universe': 'TOP3000',
            'type': 'MATRIX', 'dateCoverage': 1.0, 'coverage': 0.3561,
            'userCount': 0, 'alphaCount': 0, 'themes': [], 'dateCreated': '2026-03-01'
        }, ...]
        """
        params = {
            "instrumentType": instrument_type,
            "region": region,
            "universe": universe,
            "delay": delay,
            "limit": 50,
            "offset": 0,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id
        if search:
            params["search"] = search

        results = []
        while True:
            body = self._get(f"{BASE}/data-fields", params=params).json()
            batch = body["results"]
            results.extend(batch)
            if not batch or len(results) >= body["count"]:
                break
            params["offset"] += 50
        return results

    def get_data_sets(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
    ) -> list:
        """
        List available datasets.

        Returns:
        [{
            'id': 'univ1', 'name': 'Universe Dataset',
            'description': 'No dataset description',
            'category': {'id': 'pv', 'name': 'Price Volume'},
            'subcategory': {'id': 'pv-price-volume', 'name': 'Price Volume'},
            'region': 'USA', 'delay': 1, 'universe': 'TOP2000',
            'dateCoverage': 1.0, 'coverage': 1.0, 'valueScore': 3.0,
            'userCount': 0, 'alphaCount': 0, 'fieldCount': 6,
            'themes': [], 'dateUpdated': '2026-03-01', 'researchPapers': []
        }, ...]
        """
        params = {
            "instrumentType": instrument_type,
            "region": region,
            "universe": universe,
            "delay": delay,
            "limit": 50,
            "offset": 0,
        }
        results = []
        while True:
            r = self._get(f"{BASE}/data-sets", params=params).json()
            results.extend(r.get("results", []))
            if len(results) >= r.get("count", 0):
                break
            params["offset"] += 50
        return results

    def get_universes(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
    ) -> list[str]:
        """List universes available for a region (from OPTIONS /simulations).

        Returns:
        ['TOP2000', 'TOP3000', ...]
        """
        settings = self._options(f"{BASE}/simulations")["actions"]["POST"]["settings"]["children"]
        for setting in settings.values():
            if setting.get("label") != "Universe":
                continue
            region_universes = setting["choices"]["instrumentType"][instrument_type]["region"][
                region
            ]
            return [u["value"] if isinstance(u, dict) else u for u in region_universes]
        return []

    # ------------------------------------------------------------------
    # Simulations
    # ------------------------------------------------------------------

    def simulate(
        self,
        code: str,
        region: str = "USA",
        universe: str = "TOP3000",
        neutralization: str = "SUBINDUSTRY",
        decay: int = 6,
        truncation: float = 0.08,
        delay: int = 1,
        pasteurization: str = "ON",
        nan_handling: str = "OFF",
        language: str = "FASTEXPR",
        wait: bool = True,
        poll_interval: float = 60.0,
    ) -> str | None:
        """
        POST a simulation. If wait=True, blocks until complete and returns alpha_id.
        If wait=False, returns the simulation progress URL.
        """
        payload = {
            "type": "REGULAR",
            "regular": code,
            "settings": {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization.upper(),
                "truncation": truncation,
                "pasteurization": pasteurization.upper(),
                "nanHandling": nan_handling.upper(),
                "unitHandling": "VERIFY",
                "language": language,
                "visualization": False,
            },
        }
        r = self._post(f"{BASE}/simulations", json=payload)
        progress_url = r.headers.get("Location")
        if not progress_url:
            body = r.json() if r.content else {}
            if body.get("detail") == "CONCURRENT_SIMULATION_LIMIT_EXCEEDED":
                wait_s = float(r.headers.get("Retry-After", 15))
                logger.warning("Concurrent sim limit — retrying in %.0fs", wait_s)
                time.sleep(wait_s)
                return self.simulate(
                    code=code,
                    region=region,
                    universe=universe,
                    neutralization=neutralization,
                    decay=decay,
                    truncation=truncation,
                    delay=delay,
                    pasteurization=pasteurization,
                    nan_handling=nan_handling,
                    language=language,
                    wait=wait,
                    poll_interval=poll_interval,
                )
            raise RuntimeError(f"No Location header: {r.content}")
        logger.info(f"Simulation queued: {progress_url}")

        if not wait:
            return progress_url

        return self._poll_simulation(progress_url, poll_interval)

    def _poll_simulation(self, progress_url: str, interval: float = 30.0) -> str:
        """Block until simulation finishes; return alpha_id."""
        while True:
            r = self._get(progress_url)

            body = r.json()
            if "alpha" in body:
                logger.info(f"Simulation complete: alpha {body['alpha']}")
                return body["alpha"]
            if "message" in body and "progress" not in body:
                raise RuntimeError(f"Simulation failed: {body['message']}")

            time.sleep(interval)

    # ------------------------------------------------------------------
    # Alphas
    # ------------------------------------------------------------------

    def get_alpha(self, alpha_id: str) -> dict:
        """
        Fetch alpha details.

        Returns:
        {
            'id': 'xxx', 'type': 'REGULAR', 'author': 'xxx',
            'settings': {'instrumentType': 'EQUITY', 'region': 'USA', 'universe': 'TOP3000', 'delay': 1, 'decay': 6, 'neutralization': 'SUBINDUSTRY', 'truncation': 0.08, 'pasteurization': 'ON', 'unitHandling': 'VERIFY', 'nanHandling': 'OFF', 'maxTrade': 'OFF', 'maxPosition': 'OFF', 'language': 'FASTEXPR', 'visualization': False, 'startDate': '2019-01-01', 'endDate': '2023-12-31'},
            'regular': {'code': 'rank(close)', 'description': None, 'operatorCount': 1},
            'dateCreated': '2026-06-10T12:18:06-04:00', 'dateSubmitted': None, 'dateModified': '2026-06-10T12:18:15-04:00',
            'name': 'test_rank_close', 'favorite': False, 'hidden': False, 'color': None, 'category': None, 'tags': ['test'], 'classifications': [{'id': 'DATA_USAGE:SINGLE_DATA_SET', 'name': 'Single Data Set Alpha'}], 'grade': 'INFERIOR', 'stage': 'IS', 'status': 'UNSUBMITTED',
            'is': {
                'pnl': 465873, 'bookSize': 20000000, 'longCount': 1536, 'shortCount': 1535, 'turnover': 0.0159, 'returns': 0.0094, 'drawdown': 0.5345, 'margin': 0.001184, 'sharpe': 0.07, 'fitness': 0.02, 'startDate': '2019-01-01',
                'checks': [
                    {'name': 'LOW_SHARPE', 'result': 'FAIL', 'limit': 1.25, 'value': 0.07},
                    {'name': 'LOW_FITNESS', 'result': 'FAIL', 'limit': 1.0, 'value': 0.02},
                    {'name': 'LOW_TURNOVER', 'result': 'PASS', 'limit': 0.01, 'value': 0.0159},
                    {'name': 'HIGH_TURNOVER', 'result': 'PASS', 'limit': 0.7, 'value': 0.0159},
                    {'name': 'CONCENTRATED_WEIGHT', 'result': 'PASS'},
                    {'name': 'LOW_SUB_UNIVERSE_SHARPE', 'result': 'FAIL', 'limit': 0.03, 'value': -0.14},
                    {'name': 'SELF_CORRELATION', 'result': 'PENDING'},
                    {'name': 'MATCHES_COMPETITION', 'result': 'PASS', 'competitions': [{'id': 'challenge', 'name': 'Challenge'}]}
                ]
            },
            'os': None, 'train': None, 'test': None, 'prod': None, 'competitions': None,
            'themes': None, 'pyramids': None, 'pyramidThemes': None, 'team': None, 'origin': 'PLATFORM'}
        """
        return self._get(f"{BASE}/alphas/{alpha_id}").json()

    def get_alpha_check(self, alpha_id: str) -> dict:
        """Fetch IS checks (incl. SELF_CORRELATION, PROD_CORRELATION).

        Returns:
        {'is': {'checks':
            [
            {'name': 'LOW_SHARPE', 'result': 'FAIL', 'limit': 1.25, 'value': 0.07},
            {'name': 'LOW_FITNESS', 'result': 'FAIL', 'limit': 1.0, 'value': 0.02},
            {'name': 'LOW_TURNOVER', 'result': 'PASS', 'limit': 0.01, 'value': 0.0159},
            {'name': 'HIGH_TURNOVER', 'result': 'PASS', 'limit': 0.7, 'value': 0.0159},
            {'name': 'CONCENTRATED_WEIGHT', 'result': 'PASS'},
            {'name': 'LOW_SUB_UNIVERSE_SHARPE', 'result': 'FAIL', 'limit': 0.03, 'value': -0.14},
            {'name': 'SELF_CORRELATION', 'result': 'PENDING'},
            {'name': 'MATCHES_COMPETITION', 'result': 'PASS',
            'competitions': [{'id': 'challenge', 'name': 'Challenge'}]}]]}
        }
        """
        while True:
            r = self._get(f"{BASE}/alphas/{alpha_id}/check")
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                time.sleep(float(retry_after))
                continue
            return r.json()

    def submit_alpha(self, alpha_id: str) -> requests.Response:
        return self._post(f"{BASE}/alphas/{alpha_id}/submit")

    def submit_and_poll(self, alpha_id: str, interval: float = 5.0) -> dict:
        """POST submit then poll until WQ returns a final verdict."""
        r = self.submit_alpha(alpha_id)
        if r.status_code == 403:
            r.raise_for_status()
        if r.status_code not in (200, 201, 202, 204, 503):
            r.raise_for_status()
        return self.poll_submit(alpha_id, interval=interval)

    def poll_submit(self, alpha_id: str, interval: float = 5.0) -> dict:
        """Block until submission resolves; return final submit response."""
        while True:
            r = self.session.get(f"{BASE}/alphas/{alpha_id}/submit")
            if r.status_code == 401:
                logger.warning("Session expired, re-logging in...")
                self.login()
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 5)))
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(5)
                continue
            if r.status_code == 404:
                return {"status": "already_submitted"}
            if r.status_code == 403:
                r.raise_for_status()
            r.raise_for_status()
            if r.content:
                return r.json()
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _options(self, url, **kwargs) -> dict:
        r = self.session.options(url, **kwargs)
        self._check_auth(r)
        return r.json()

    def _get(self, url, **kwargs) -> requests.Response:
        backoff = 30
        while True:
            try:
                r = self.session.get(url, **kwargs)
            except requests.exceptions.ConnectionError as exc:
                logger.warning("GET connection error (%s), retrying in %ds", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 5  # reset on successful connect
            if r.status_code == 401:
                logger.warning("Session expired, re-logging in...")
                self.login()
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 5))
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                logger.warning("Server error %d, retrying in 5s", r.status_code)
                time.sleep(5)
                continue
            r.raise_for_status()
            return r

    def _post(self, url, **kwargs) -> requests.Response:
        backoff = 5
        while True:
            try:
                r = self.session.post(url, **kwargs)
            except requests.exceptions.ConnectionError as exc:
                logger.warning("POST connection error (%s), retrying in %ds", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            self._check_auth(r)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 15))
                logger.warning("POST rate limited, retrying in %.0fs", wait)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                logger.warning("POST server error %d, retrying in 5s", r.status_code)
                time.sleep(5)
                continue
            return r

    def _patch(self, url, **kwargs) -> requests.Response:
        r = self.session.patch(url, **kwargs)
        self._check_auth(r)
        return r

    def _check_auth(self, r: requests.Response):
        if r.status_code == 401:
            logger.warning("Session expired, re-logging in...")
            self.login()
            r.raise_for_status()
