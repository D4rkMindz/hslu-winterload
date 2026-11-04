"""The only code that knows where data lives. Domain code, pipelines and notebooks ask a repository
for named data, and table names, column filters and result paths stop here.

Modules here import no config and no storage backend, so tests can hand them a fake. Only
winterload.repository.factory reads settings.
"""
