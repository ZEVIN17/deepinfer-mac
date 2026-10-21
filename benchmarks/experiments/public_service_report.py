"""Keep local execution paths in excluded results; publish portable aliases."""

from pathlib import Path


def public_report(value, *, workspace=None, snapshot=None):
    workspace = str(Path(workspace or Path.cwd()).resolve())
    substitutions = [(workspace, "<WORKSPACE>"), (str(Path.home()), "<HOME>")]
    if snapshot:
        substitutions.insert(0, (str(snapshot), "<MODEL_SNAPSHOT>"))

    def visit(item):
        if isinstance(item, dict):
            return {visit(key): visit(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(value) for value in item]
        if isinstance(item, str):
            for actual, alias in substitutions:
                item = item.replace(actual, alias)
        return item

    return visit(value)
