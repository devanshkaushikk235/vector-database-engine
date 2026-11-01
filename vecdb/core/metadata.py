from typing import Any
from vecdb.core.vector_store import VectorStore


def evaluate_operator(field_value: Any, op: str, target_val: Any) -> bool:
    """Evaluate a single operator condition against a field value.
    
    Supported operators: eq, neq, gt, gte, lt, lte, in, nin.
    """
    if op == "eq":
        return field_value == target_val
    elif op == "neq":
        return field_value != target_val
    elif op == "gt":
        return field_value is not None and field_value > target_val
    elif op == "gte":
        return field_value is not None and field_value >= target_val
    elif op == "lt":
        return field_value is not None and field_value < target_val
    elif op == "lte":
        return field_value is not None and field_value <= target_val
    elif op == "in":
        if isinstance(field_value, (list, tuple, set)):
            return any(item in target_val for item in field_value)
        return field_value in target_val if hasattr(target_val, "__contains__") else False
    elif op == "nin":
        if isinstance(field_value, (list, tuple, set)):
            return not any(item in target_val for item in field_value)
        return field_value not in target_val if hasattr(target_val, "__contains__") else True
    else:
        raise ValueError(f"Unsupported filter operator: {op}")


def evaluate_field_condition(field_value: Any, condition: Any) -> bool:
    """Evaluate a field condition, which can be direct equality or a dict of operators."""
    if isinstance(condition, dict):
        for op, target_val in condition.items():
            if not evaluate_operator(field_value, op, target_val):
                return False
        return True
    else:
        return field_value == condition


def evaluate_filter(payload: dict[str, Any], filter_query: dict[str, Any] | None) -> bool:
    """Evaluate if a payload dictionary matches a structured filter query.
    
    Supports implicit AND across top-level fields, as well as explicit '$and' and '$or'.
    """
    if not filter_query:
        return True

    # Handle logical AND / OR operators
    if "$and" in filter_query:
        and_clauses = filter_query["$and"]
        if not all(evaluate_filter(payload, clause) for clause in and_clauses):
            return False

    if "$or" in filter_query:
        or_clauses = filter_query["$or"]
        if not any(evaluate_filter(payload, clause) for clause in or_clauses):
            return False

    # Field-level conditions
    for key, condition in filter_query.items():
        if key in ("$and", "$or"):
            continue
        field_val = payload.get(key)
        if not evaluate_field_condition(field_val, condition):
            return False

    return True


def get_matching_ids(vector_store: VectorStore, filter_query: dict[str, Any] | None) -> set[str]:
    """Retrieve all active vector IDs matching a filter query from VectorStore."""
    if not filter_query:
        return set(vector_store.get_all_ids(include_tombstones=False))

    active_ids = vector_store.get_all_ids(include_tombstones=False)
    matching = set()
    for vid in active_ids:
        payload = vector_store.get_payload(vid)
        if payload is not None and evaluate_filter(payload, filter_query):
            matching.add(vid)
    return matching
