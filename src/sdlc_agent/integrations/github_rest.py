"""GitHub REST: opening pull requests for delivered branches.

The token travels in the ``Authorization`` header only - never in a URL, never
in an error message. Everything here is offline-testable by injecting an
httpx transport, the same seam the git client's runner provides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


class UnsupportedRemoteError(RuntimeError):
    """The remote is not the GitHub host this client is configured for."""


class GitHubApiError(RuntimeError):
    """A GitHub API call failed.

    The message carries the repo, the call, the status and a truncated
    response snippet. GitHub never echoes the Authorization header, and this
    class never includes it either - errors are audit material.
    """

    def __init__(self, repo: str, method: str, path: str, response: httpx.Response) -> None:
        snippet = response.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        super().__init__(
            f"GitHub {method} /repos/{repo}{path} -> {response.status_code}: {snippet}"
        )
        self.status_code = response.status_code


#: owner/name, the only shape a repository slug may take on GitHub.
_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_slug_from_remote(url: str, *, git_host: str = "github.com") -> str:
    """``owner/name`` from an https remote on ``git_host``, or raise.

    Only the configured host is accepted: the API token is scoped to that
    host, and a remote pointing elsewhere must not get a PR about code it
    does not host.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc.lower() != git_host.lower():
        raise UnsupportedRemoteError(f"remote {url!r} is not an https remote on {git_host}")
    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not _SLUG.fullmatch(path):
        raise UnsupportedRemoteError(f"remote {url!r} does not name an owner/repository")
    return path


@dataclass(frozen=True, slots=True)
class GithubPullRequests:
    """Open pull requests over the GitHub REST API.

    Implements the pull-request seam the deliver node declares in
    :mod:`sdlc_agent.pipeline.codeflow` (structural typing - this module
    imports nothing from the pipeline layer, keeping the dependency arrow
    pointing one way).

    Attributes:
        token: Fine-grained PAT with pull-request write on target repos.
        api_base_url: API root; point at ``https://host/api/v3`` for GitHub
            Enterprise.
        git_host: The git-remote host corresponding to the API root, used by
            :func:`repo_slug_from_remote` to match remotes to this client.
        timeout: Per-request ceiling; the API must not hang a graph node.
        transport: httpx transport seam. Tests inject a MockTransport; the
            default does real HTTP.
    """

    token: str
    api_base_url: str = "https://api.github.com"
    git_host: str = "github.com"
    timeout: float = 30.0
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.api_base_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            transport=self.transport,
            timeout=self.timeout,
        )

    def ensure_open(self, *, repo: str, branch: str, title: str, body: str) -> str:
        """Return the open PR's ``html_url`` for ``branch``, opening one if needed.

        Idempotent by construction: an existing PR is looked up before any
        POST, so a revision loop that re-delivers finds its own PR. A 422
        from a lost race (opened between the lookup and the POST) resolves
        through the same lookup instead of failing the delivery.
        """
        owner = repo.split("/", 1)[0]
        head = f"{owner}:{branch}"
        with self._client() as client:
            existing = self._find_open(client, repo, head)
            if existing is not None:
                return existing

            response = client.post(
                f"/repos/{repo}/pulls",
                json={
                    "title": title,
                    "head": branch,
                    "base": self._default_branch(client, repo),
                    "body": body,
                },
            )
            if response.status_code == 422:
                existing = self._find_open(client, repo, head)
                if existing is not None:
                    return existing
            if response.status_code != 201:
                raise GitHubApiError(repo, "POST", "/pulls", response)
            return str(response.json()["html_url"])

    def _find_open(self, client: httpx.Client, repo: str, head: str) -> str | None:
        response = client.get(f"/repos/{repo}/pulls", params={"head": head, "state": "open"})
        if response.status_code != 200:
            raise GitHubApiError(repo, "GET", "/pulls", response)
        pulls = response.json()
        if isinstance(pulls, list) and pulls:
            return str(pulls[0]["html_url"])
        return None

    def _default_branch(self, client: httpx.Client, repo: str) -> str:
        response = client.get(f"/repos/{repo}")
        if response.status_code != 200:
            raise GitHubApiError(repo, "GET", "", response)
        return str(response.json()["default_branch"])


__all__ = [
    "GitHubApiError",
    "GithubPullRequests",
    "UnsupportedRemoteError",
    "repo_slug_from_remote",
]
