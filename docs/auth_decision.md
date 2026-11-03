# Authentication decision — 2026-11-03

NEEDS OWNER ACTION: no provider selected/configured. Guest use remains available. No password form, custom cookie, email sending, account privilege or saved-run backend added. Absent functionality is not advertised as working.

Streamlit OIDC establishes identity, not database authorization or native email/password signup. Auth0 is documented as an OIDC option. Supabase/Firebase/Clerk need provider-specific integration, token validation and storage-policy review before selection; current prices, SMTP limits and complete signup compatibility are NOT VERIFIED here. No provider selected on assumed free-tier promises.

Defer account UI until a sandbox and lifecycle tests exist. No privilege based on public contact email. Optional S3/PostgreSQL adapters are not verified user-isolated workspaces. See [owner setup](OWNER_SETUP.md) and [official Streamlit identity documentation](https://docs.streamlit.io/develop/concepts/connections/authentication).
