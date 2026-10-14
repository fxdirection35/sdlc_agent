"""Integrations with the world outside the pipeline: git and GitHub.

Everything here implements a protocol declared in the pipeline layer (the
:class:`~sdlc_agent.pipeline.codeflow.RepoClient` seam, mirroring
:class:`~sdlc_agent.pipeline.verify.TestExecutor`), so the graph depends on
the seam and tests inject fakes - never the network.
"""

from sdlc_agent.integrations.git_subprocess import GitCommandError, SubprocessGit
from sdlc_agent.integrations.github_rest import (
    GitHubApiError,
    GithubPullRequests,
    UnsupportedRemoteError,
    repo_slug_from_remote,
)

__all__ = [
    "GitCommandError",
    "GitHubApiError",
    "GithubPullRequests",
    "SubprocessGit",
    "UnsupportedRemoteError",
    "repo_slug_from_remote",
]
