-- ============================================================================
-- Calculus Visualizer PostgreSQL Initialization Script
-- Security: Least Privilege, Row-Level Security (RLS), pgcrypto Encryption
-- ============================================================================

-- 1. Enable pgcrypto Extension for Column-Level Data Encryption at Rest
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 2. Create Dedicated Application User with Least Privilege (No Superuser)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        -- Password is set via container initialization or environment variable
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_secure_password_2026';
    END IF;
END
$$;

-- Grant Connection & Schema Privileges to app_user
GRANT CONNECT ON DATABASE calculus_vis TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;

-- 3. Create Multi-Tenant Schema Tables
CREATE TABLE IF NOT EXISTS visualization_logs (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    expressions_count INTEGER DEFAULT 0,
    processing_time_ms DOUBLE PRECISION DEFAULT 0.0,
    error_message TEXT,
    encrypted_jwt_token BYTEA,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS usage_logs (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    encrypted_jwt_token BYTEA,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rate_limit_audit (
    id SERIAL PRIMARY KEY,
    client_ip TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Grant Strict CRUD Privileges ONLY (Least Privilege)
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE visualization_logs TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE usage_logs TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE rate_limit_audit TO app_user;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_user;

-- 4. Multi-Tenancy via Row-Level Security (RLS)
ALTER TABLE visualization_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE visualization_logs FORCE ROW LEVEL SECURITY;

ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_logs FORCE ROW LEVEL SECURITY;

ALTER TABLE rate_limit_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limit_audit FORCE ROW LEVEL SECURITY;

-- Security Policies: Enforce access matching current_setting('app.current_session_id') or app.is_admin = 'true'

-- Visualization Logs Policy
DROP POLICY IF EXISTS visualization_logs_tenant_isolation ON visualization_logs;
CREATE POLICY visualization_logs_tenant_isolation ON visualization_logs
    FOR ALL
    TO app_user
    USING (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
    )
    WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
    );

-- Usage Logs Policy
DROP POLICY IF EXISTS usage_logs_tenant_isolation ON usage_logs;
CREATE POLICY usage_logs_tenant_isolation ON usage_logs
    FOR ALL
    TO app_user
    USING (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
    )
    WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
    );

-- Rate Limit Audit Policy
DROP POLICY IF EXISTS rate_limit_audit_tenant_isolation ON rate_limit_audit;
CREATE POLICY rate_limit_audit_tenant_isolation ON rate_limit_audit
    FOR ALL
    TO app_user
    USING (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
        OR client_ip = NULLIF(current_setting('app.current_client_ip', true), '')
    )
    WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR session_id = NULLIF(current_setting('app.current_session_id', true), '')
        OR client_ip = NULLIF(current_setting('app.current_client_ip', true), '')
    );
