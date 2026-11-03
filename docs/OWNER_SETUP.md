# Owner setup

NEEDS OWNER ACTION for optional external services; guest work is independent.

1. Review the feature PR. No automatic merge. Existing production entry point remains app/streamlit_app.py; no deployment settings changed.
2. If accounts are wanted, select a managed OIDC provider/test tenant and approve any costs first. Configure verification, recovery, deletion and consent through the provider. No tokens in chat.
3. A future approved Streamlit OIDC adapter needs redirects https://mini-project-ml-lab-rssr.streamlit.app/oauth2callback and http://localhost:8512/oauth2callback for this local port. Store client ID/secret, cookie secret and metadata URL only in deployment secret management. This is not a working signup flow.
4. Select private storage/database, user-ID authorization, row-level isolation, retention/deletion and quotas. Require two-user isolation, revocation, export/deletion tests before saved runs. Cloud local files remain ephemeral.
5. Verify deployment and cold restart after reviewed integration. Do not replace production settings merely to test the feature branch.

NOT VERIFIED: provider delivery, signup/login/recovery, RLS, persistence and hosted AURORA release. No credentials are needed to continue local UI/lesson/passport work.
