from typing import Any


def snapshot(obj) -> dict[str, Any]:
    return {col.key: getattr(obj, col.key) for col in obj.__table__.columns}


def build_create_changes(obj, exclude: set[str] | None = None) -> dict[str, list[Any]]:
    exclude = exclude or {"id", "created_ts", "last_login_ts"}
    return {
        col.key: [None, getattr(obj, col.key)]
        for col in obj.__table__.columns
        if col.key not in exclude
    }


def build_update_changes(
    before: dict[str, Any],
    after,
    exclude: set[str] | None = None,
) -> dict[str, list[Any]]:
    exclude = exclude or {"id", "created_ts", "last_login_ts"}
    changes = {}
    for col in after.__table__.columns:
        key = col.key
        if key in exclude:
            continue
        old = before.get(key)
        new = getattr(after, key)
        if old != new:
            changes[key] = [old, new]
    return changes


def build_delete_changes(
    before: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, list[Any]]:
    exclude = exclude or {"id", "created_ts", "last_login_ts"}
    return {
        key: [val, None]
        for key, val in before.items()
        if key not in exclude
    }