# Research interface upgrade

## Implemented presentation scope

The shared `app/research_design.py` layer changes presentation without replacing scientific or lesson implementations. It provides light/dark appearance, responsive cards, a compact hero, evidence-derived architecture availability, a nine-stage roadmap, an explicitly historical MP31 figure, and researcher/contact information. Existing seven internal navigation IDs, state keys, input controls and inference routes remain intact; user-facing labels introduce AI Studio, Model Lab and Learning Lab. No provider calls, models or training run merely to render the new Overview.

Researcher: Ahmdullah, currently pursuing M.Tech in Signal Processing and Machine Learning at IIT Guwahati. Contact: exammotivation0816@gmail.com; LinkedIn: https://www.linkedin.com/in/exammotivation/. Affiliation is personal academic background, not endorsement.

## Boundaries and remaining acceptance

- Appearance changes are session-scoped; they do not create an account or durable preferences.
- Existing chart/table components retain their own native rendering. CSS breakpoints/focus styles are implemented, but desktop/mobile browser and contrast review remain release acceptance checks. AppTest cannot prove pixel layout or screen-reader conformance.
- All seven families are derived from checkpoint evidence, not hardcoded as scientifically accepted. The six scale choices remain subject to the existing output resource cap.
- The recorded figure is not a fresh inference result. No illustrative unrelated input/output pairs are introduced.
- Authentication, saved personal history, account deletion and subscriptions are not implemented in this presentation milestone. No password/email collection form, fake login, email sending or admin-by-contact-email rule is introduced.
- Public demos and lessons remain accessible without registration. If deployment becomes private, owner settings must restore intended public access; app code cannot override a hosting access wall.

## Optional accounts: owner configuration required

Before implementing account flows, choose/configure an authorized managed identity provider and consult its current official Streamlit-compatible flow. Configure verified redirect URLs, email verification/recovery, session expiry, logout and provider rate limits. Keep credentials in Streamlit Secrets; never paste them into source or chat. Use a persistent store with per-user authorization before enabling saved private history. Verify isolation with two separate test accounts, including logout, expired sessions and unauthorized reads. Contact email is not an authorization credential. Newsletter consent, if later requested, is separate and email sending requires explicit authorization. These are requirements, not claims of a configured integration.

## Verification

Run `PYTHONPATH=src python scripts/check_research_design.py` and the existing `scripts/check_showcase.py`. The new check renders the guest Overview with network blocked, checks both appearances and existing CTA routes, and preserves the initial inference action. CI includes both checks. Review the actual PR head results before deployment or merging. Scientific/training/frozen-study acceptance remains unchanged.
