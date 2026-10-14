"""The test suite as a package, so subdirectories can share helpers.

``tests.service`` builds runs with the same scripted roles and artifact
factories as ``tests.pipeline``; a package ``__init__`` makes that import
(``from tests.pipeline.helpers import ...``) resolve the same way under pytest
as it does in a plain interpreter, instead of depending on which directory
pytest happened to prepend first.
"""
