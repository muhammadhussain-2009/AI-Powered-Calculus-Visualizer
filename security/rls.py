import re
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional

class SecurityRole(str, Enum):
    ANONYMOUS_USER = "anonymous_user"
    SYSTEM_ADMIN = "system_admin"

class SecurityContext:
    """
    Encapsulates user session security context.
    Determines user role and session identifier for Role-Level Security (RLS).
    """
    def __init__(self, session_id: str, role: SecurityRole = SecurityRole.ANONYMOUS_USER, client_ip: Optional[str] = None):
        self.session_id = session_id or "anonymous"
        self.role = role
        self.client_ip = client_ip

    def is_admin(self) -> bool:
        return self.role == SecurityRole.SYSTEM_ADMIN

class RLSEngine:
    """
    Role-Level Security (RLS) Engine for database operations.
    Enforces row-level isolation so that each anonymous session can strictly access
    and write only its own session-scoped data across all database tables.
    """
    ALLOWED_TABLES = {"visualization_logs", "usage_logs", "rate_limit_audit"}

    @classmethod
    def apply_read_policy(
        cls, table: str, context: SecurityContext, sql: str, params: Optional[List[Any]] = None
    ) -> Tuple[str, List[Any]]:
        """
        Applies RLS read filter. For ANONYMOUS_USER, restricts SELECT queries to rows
        matching context.session_id.
        """
        if table not in cls.ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' is not registered under RLS policies.")

        params_list = list(params) if params else []

        if context.is_admin():
            return sql, params_list

        clause = "session_id = ?"
        upper_sql = sql.upper()

        if "WHERE" in upper_sql:
            # Append to existing WHERE clause before ORDER BY / LIMIT / GROUP BY
            split_keyword = None
            for kw in ["ORDER BY", "LIMIT", "GROUP BY"]:
                if kw in upper_sql:
                    split_keyword = kw
                    break
            
            if split_keyword:
                parts = re.split(f"(?i){split_keyword}", sql, maxsplit=1)
                scoped_sql = f"{parts[0]} AND {clause} {split_keyword}{parts[1]}"
            else:
                scoped_sql = f"{sql} AND {clause}"
        else:
            # Add WHERE clause before ORDER BY / LIMIT / GROUP BY
            split_keyword = None
            for kw in ["ORDER BY", "LIMIT", "GROUP BY"]:
                if kw in upper_sql:
                    split_keyword = kw
                    break
            
            if split_keyword:
                parts = re.split(f"(?i){split_keyword}", sql, maxsplit=1)
                scoped_sql = f"{parts[0]} WHERE {clause} {split_keyword}{parts[1]}"
            else:
                scoped_sql = f"{sql} WHERE {clause}"

        # Insert session_id parameter at index 0 if clause was injected before trailing keywords (like ORDER BY/LIMIT)
        if split_keyword:
            params_list.insert(0, context.session_id)
        else:
            params_list.append(context.session_id)
        return scoped_sql, params_list

    @classmethod
    def apply_insert_policy(
        cls, table: str, context: SecurityContext, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Applies RLS write filter. Ensures record session_id matches context.session_id
        to prevent session spoofing.
        """
        if table not in cls.ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' is not registered under RLS policies.")

        validated_record = dict(record)
        if not context.is_admin():
            validated_record["session_id"] = context.session_id

        return validated_record
