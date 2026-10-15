<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## MEDIQ architecture
- Keep shared navigation and session-only clinical inputs in MediqShell around the root Outlet so inputs survive frontend page navigation without persistence.
- Define clinical features, illustrative reference categories and report generation in the browser-safe clinical module so forms and exports share one schema.
- Never synthesize predictions or SHAP contributions in this frontend prototype; use explicit unavailable states until a trained model is connected.
- Give each leaf route its own metadata through pageMeta so each research page has a distinct title and description.
