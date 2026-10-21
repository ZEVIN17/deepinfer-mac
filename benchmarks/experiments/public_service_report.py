"""Keep local execution paths in excluded results; publish portable aliases."""

from pathlib import Path


def public_report(value, *, workspace=None, snapshot=None):
    workspace = str(Path(workspace or Path.cwd()).resolve())
    substitutions = [(workspace, "<WORKSPACE>"), (str(Path.home()), "<HOME>")]
    if snapshot:
        substitutions.insert(0, (str(snapshot), "<MODEL_SNAPSHOT>"))
    configuration = value.get("configuration", {}) if isinstance(value, dict) else {}
    frozen = configuration.get("frozen_source", {})
    if frozen.get("root"):
        substitutions.insert(0, (str(Path(frozen["root"]).parent), "<EXPERIMENT_ROOT>"))
    for engine in configuration.get("engines", []):
        command = engine.get("command", [])
        if command and engine.get("framework") == "omlx":
            substitutions.insert(0, (command[0], "<OMLX_PYTHON>"))
        draft = engine.get("additional_draft") or {}
        if draft.get("snapshot"):
            substitutions.insert(0, (draft["snapshot"], "<DFLASH_SNAPSHOT>"))
    for key, alias in (("omlx_dflash_resource_admission", "<DFLASH_RESOURCE_REPORT>"),
                       ("dev_selection_report", "<DEV_SELECTION_REPORT>")):
        item = configuration.get(key) or {}
        if item.get("path"):
            substitutions.insert(0, (item["path"], alias))
    dev = configuration.get("dev_selection_report") or {}
    if dev.get("raw_path"):
        substitutions.insert(0, (dev["raw_path"], "<DEV_RAW_REPORT>"))
    substitutions.sort(key=lambda pair: len(pair[0]), reverse=True)

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
