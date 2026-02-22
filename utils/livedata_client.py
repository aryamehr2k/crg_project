from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time
import urllib.request
from typing import Any


API_BASE = "https://gotlivedata.io/api"
SESSION_URL = f"{API_BASE}/identity/v1/session"


@dataclass
class LiveDataCredentials:
    org_id: str
    api_key: str | None = None
    client_id: str | None = None
    client_secret: str | None = None


class LiveDataClient:
    def __init__(self, creds: LiveDataCredentials) -> None:
        self.creds = creds
        self._access_token_value: str | None = None
        self._expires_at: float | None = None

    @staticmethod
    def from_env() -> "LiveDataClient":
        org_id = os.environ.get("LIVEDATA_ORG_ID") or os.environ.get("GOTLIVEDATA_ORG_ID")
        api_key = os.environ.get("LIVEDATA_API_KEY")
        client_id = os.environ.get("LIVEDATA_CLIENT_ID")
        client_secret = os.environ.get("LIVEDATA_CLIENT_SECRET")

        if not org_id:
            raise ValueError("LIVEDATA_ORG_ID is required to use the Live Data API")

        creds = LiveDataCredentials(
            org_id=org_id,
            api_key=api_key,
            client_id=client_id,
            client_secret=client_secret,
        )
        return LiveDataClient(creds)

    def _token_active(self) -> bool:
        if not self._access_token_value or not self._expires_at:
            return False
        return (self._expires_at - time.time()) > 30

    def _login(self) -> str:
        if self.creds.api_key:
            self._access_token_value = self.creds.api_key
            self._expires_at = time.time() + 24 * 3600 * 365
            return self._access_token_value

        if not self.creds.client_id or not self.creds.client_secret:
            raise ValueError("Provide LIVEDATA_API_KEY or LIVEDATA_CLIENT_ID + LIVEDATA_CLIENT_SECRET")

        payload = {
            "grantType": "clientCredentials",
            "clientId": self.creds.client_id,
            "clientSecret": self.creds.client_secret,
        }
        resp = self._request_json(
            SESSION_URL,
            payload,
            auth_required=False,
        )

        token = resp.get("accessToken")
        expires_at = resp.get("expiresAt")
        if not token or not expires_at:
            raise RuntimeError("Live Data auth response missing accessToken or expiresAt")

        try:
            expires_dt = datetime.fromisoformat(expires_at)
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            self._expires_at = expires_dt.timestamp()
        except ValueError:
            self._expires_at = time.time() + 15 * 60

        self._access_token_value = token
        return token

    def _get_access_token(self) -> str:
        if self._token_active():
            return self._access_token_value  # type: ignore[return-value]
        return self._login()

    def _request_json(self, url: str, payload: dict[str, Any], auth_required: bool = True) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if auth_required:
            token = self._get_access_token()
            headers["authorization"] = f"Bearer {token}"

        request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")

        return json.loads(body) if body else {}

    def search_people(self, query: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_BASE}/people/v1/{self.creds.org_id}/search"
        return self._request_json(url, query, auth_required=True)

    def find_people(self, query: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_BASE}/people/v1/{self.creds.org_id}/find"
        return self._request_json(url, query, auth_required=True)
