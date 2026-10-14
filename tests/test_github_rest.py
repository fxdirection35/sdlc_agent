"""Tests for the GitHub REST client - fully offline via httpx MockTransport.

No golden task can deterministically express "a PR appears on GitHub", so the
contract lives here instead: idempotency (revision loops re-deliver to the
same PR), the race fallback, the remote-host guard, and the token travelling
in the header only.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from sdlc_agent.integrations import (
    GitHubApiError,
    GithubPullRequests,
    UnsupportedRemoteError,
    repo_slug_from_remote,
)

TOKEN = "ghp_sentinel_github_0123456789"
PR_URL = "https://github.com/example/reports/pull/7"


class Api:
    """Scripted GitHub API over MockTransport, recording every request."""

    def __init__(self, *, pulls_pages: list[list[dict[str, Any]]], **routes: Any) -> None:
        # pulls_pages: the response body for each GET /pulls, in order; the
        # last page repeats if the client polls again.
        self._pages = list(pulls_pages)
        self.requests: list[httpx.Request] = []
        self._routes = routes

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path
        if method == "GET" and path == "/repos/example/reports/pulls":
            page = self._pages.pop(0) if len(self._pages) > 1 else self._pages[0]
            return httpx.Response(200, json=page)
        if method == "GET" and path == "/repos/example/reports":
            return httpx.Response(200, json={"default_branch": self._routes.get("default", "main")})
        if method == "POST" and path == "/repos/example/reports/pulls":
            return httpx.Response(
                self._routes.get("post_status", 201),
                json={"html_url": PR_URL},
            )
        return httpx.Response(404, json={"message": "not found"})

    def client(self) -> GithubPullRequests:
        return GithubPullRequests(token=TOKEN, transport=httpx.MockTransport(self.handler))


def test_an_existing_open_pr_is_returned_without_posting() -> None:
    api = Api(pulls_pages=[[{"html_url": PR_URL}]])

    url = api.client().ensure_open(
        repo="example/reports", branch="sdlc-agent/t1", title="t", body="b"
    )

    assert url == PR_URL
    assert [request.method for request in api.requests] == ["GET"]


def test_opens_a_pr_when_none_exists() -> None:
    api = Api(pulls_pages=[[]])

    url = api.client().ensure_open(
        repo="example/reports", branch="sdlc-agent/t1", title="Add CSV export", body="summary"
    )

    assert url == PR_URL
    post = next(request for request in api.requests if request.method == "POST")
    body = post.read()
    assert b'"head":"sdlc-agent/t1"' in body
    assert b'"base":"main"' in body
    assert b'"title":"Add CSV export"' in body
    # The lookup happened first, so a re-delivery never duplicates the PR.
    assert [request.method for request in api.requests] == ["GET", "GET", "POST"]


def test_a_lost_race_falls_back_to_the_lookup() -> None:
    """422 'already exists' between our GET and POST is a race we lost, not a
    failure: resolve it through the lookup we already trust."""
    api = Api(pulls_pages=[[], [{"html_url": PR_URL}]], post_status=422)

    url = api.client().ensure_open(
        repo="example/reports", branch="sdlc-agent/t1", title="t", body="b"
    )

    assert url == PR_URL
    assert [request.method for request in api.requests] == ["GET", "GET", "POST", "GET"]


def test_api_failures_raise_with_status_but_never_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, text="server error " + "x" * 500 + f" (token {TOKEN} would be here)"
        )

    pulls = GithubPullRequests(token=TOKEN, transport=httpx.MockTransport(handler))

    with pytest.raises(GitHubApiError) as exc_info:
        pulls.ensure_open(repo="example/reports", branch="b", title="t", body="b")

    message = str(exc_info.value)
    assert "-> 500" in message
    assert len(message) < 350  # the 500-char body was truncated
    assert TOKEN not in message


def test_the_token_travels_in_the_header_only() -> None:
    api = Api(pulls_pages=[[]])

    api.client().ensure_open(repo="example/reports", branch="b", title="t", body="b")

    for request in api.requests:
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert TOKEN not in str(request.url)


def test_a_custom_default_branch_is_used_as_base() -> None:
    api = Api(pulls_pages=[[]], default="trunk")

    api.client().ensure_open(repo="example/reports", branch="b", title="t", body="b")

    post = next(request for request in api.requests if request.method == "POST")
    assert b'"base":"trunk"' in post.read()


class TestRepoSlugFromRemote:
    def test_parses_owner_name_and_strips_git(self) -> None:
        assert repo_slug_from_remote("https://github.com/example/reports.git") == "example/reports"

    def test_accepts_the_configured_host_case_insensitively(self) -> None:
        assert repo_slug_from_remote("https://GITHUB.com/example/reports.git") == "example/reports"

    def test_a_ghe_remote_needs_the_matching_host(self) -> None:
        url = "https://ghe.corp.example/team/repo.git"
        with pytest.raises(UnsupportedRemoteError):
            repo_slug_from_remote(url)
        assert repo_slug_from_remote(url, git_host="ghe.corp.example") == "team/repo"

    @pytest.mark.parametrize(
        "url",
        [
            "https://gitlab.com/example/reports.git",
            "http://github.com/example/reports.git",  # token would travel in cleartext
            "git@github.com:example/reports.git",  # ssh remotes have no API token flow
            "https://github.com/example/reports/extra",
            "https://github.com/example",
        ],
    )
    def test_everything_else_is_refused(self, url: str) -> None:
        with pytest.raises(UnsupportedRemoteError):
            repo_slug_from_remote(url)
