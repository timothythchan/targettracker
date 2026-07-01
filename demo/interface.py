"""Gradio layout — Target Tracker institutional dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from demo.data_manager import (
    ingest_uploads,
    render_data_dashboard,
    render_overview_dashboard,
    render_step_header,
)
from demo.pipeline_runner import WORKFLOW_STAGES

if TYPE_CHECKING:
    import demo.app as app_mod


def assemble_interface(app: "app_mod") -> gr.Blocks:
    """Build the Blocks UI using handlers from ``demo.app``."""
    data_dir = app._PROJECT_ROOT / "data"
    default_ticker = app._TICKERS[0] if app._TICKERS else None
    default_quarter_disp = app._QUARTERS_DISPLAY[-1] if app._QUARTERS_DISPLAY else None
    overview_html = app.render_overview_html()

    with gr.Blocks(title="Target Tracker") as demo:
        step_header = gr.HTML(
            render_step_header(data_dir, cache_ready=not app._CACHE_IS_EMPTY)
        )

        if app._CACHE_IS_EMPTY:
            cache_banner = gr.Markdown(app._cache_banner_markdown())
        else:
            cache_banner = gr.Markdown(visible=False)

        # ── Overview ───────────────────────────────────────────────────
        with gr.Tab("Overview"):
            overview_panel = gr.HTML(overview_html)
            refresh_overview_btn = gr.Button("Refresh dashboard", variant="secondary", size="sm")

        # ── Data ─────────────────────────────────────────────────────────
        with gr.Tab("Data"):
            gr.Markdown(
                "Upload corpus files or place them under `data/raw/` and `data/processed/`. "
                "A transcript parquet is required to run the pipeline."
            )
            data_dashboard = gr.HTML(render_data_dashboard(data_dir))
            upload = gr.File(
                label="Upload files",
                file_count="multiple",
                file_types=[".parquet", ".csv", ".json"],
            )
            with gr.Row():
                upload_btn = gr.Button("Save uploads", variant="primary")
                refresh_data_btn = gr.Button("Refresh checklist")
            upload_status = gr.Markdown()

        # ── Pipeline ───────────────────────────────────────────────────
        with gr.Tab("Pipeline"):
            gr.Markdown(
                "Run stages in order. Provide your LLM API key before **llm** or **cache**. "
                "Use **Run all** to chain every stage."
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    with gr.Group():
                        api_key_tb = gr.Textbox(
                            label="LLM API key",
                            placeholder="OpenAI / Gemini key",
                            type="password",
                        )
                        with gr.Row():
                            stage_dd = gr.Dropdown(
                                choices=[s[0] for s in WORKFLOW_STAGES],
                                value=WORKFLOW_STAGES[0][0],
                                label="Stage",
                                scale=2,
                            )
                            extra_args_tb = gr.Textbox(
                                label="Extra options",
                                placeholder="e.g. --limit 100",
                                scale=3,
                            )
                        with gr.Row():
                            run_stage_btn = gr.Button("Run stage", variant="primary")
                            run_all_btn = gr.Button("Run all", variant="secondary")
                            refresh_status_btn = gr.Button("Refresh status")
                        with gr.Accordion("Stage reference", open=False):
                            gr.Markdown(
                                "\n".join(
                                    f"- **{sid}** — {label}"
                                    + (f" _(suggested: `{extras}`)_" if extras else "")
                                    for sid, label, extras in WORKFLOW_STAGES
                                )
                            )
                        status_md = gr.Markdown(app._refresh_status_markdown())
                with gr.Column(scale=5):
                    log_box = gr.Textbox(
                        label="Pipeline output",
                        lines=28,
                        interactive=False,
                        elem_classes=["log-panel"],
                    )

        # ── Entity Report ──────────────────────────────────────────────
        with gr.Tab("Entity Report"):
            if app._CACHE_IS_EMPTY:
                gr.Markdown(
                    '<div class="empty-state">'
                    "No analysis cache yet. Complete the <strong>cache</strong> stage on Pipeline, "
                    "or upload <code>pipeline_cache.json</code> on the Data tab."
                    "</div>"
                )

            with gr.Row(elem_classes=["entity-report-header"]):
                ticker_dd = gr.Dropdown(
                    choices=app._TICKERS,
                    value=default_ticker,
                    label="Entity",
                    allow_custom_value=False,
                    scale=2,
                )
                quarter_dd = gr.Dropdown(
                    choices=app._QUARTERS_DISPLAY,
                    value=default_quarter_disp,
                    label="Quarter",
                    allow_custom_value=False,
                    scale=2,
                )
                analyse_btn = gr.Button("Generate report", variant="primary", scale=1)

            gauge_html = gr.HTML()
            narrative_md = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Current commitments")
                    current_targets_tbl = gr.DataFrame(
                        headers=["Metric Name", "Type", "Context"],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Column():
                    gr.Markdown("#### Dropped commitments")
                    dropped_targets_tbl = gr.DataFrame(
                        headers=["Target Name", "Last Seen", "Type", "Persistence"],
                        interactive=False,
                        wrap=True,
                    )

            errors_md = gr.Markdown()

            tab1_outputs = [
                current_targets_tbl,
                dropped_targets_tbl,
                gauge_html,
                narrative_md,
                errors_md,
            ]

        # ── Watchlist ──────────────────────────────────────────────────
        with gr.Tab("Watchlist"):
            gr.Markdown(
                "Ranked entities by moving-target (MT) risk score for the selected quarter. "
                "Click a row for a summary, then open **Entity Report** for the full analysis."
            )
            with gr.Row():
                portfolio_quarter_dd = gr.Dropdown(
                    choices=app._QUARTERS_DISPLAY,
                    value=default_quarter_disp,
                    label="Quarter",
                    scale=3,
                )
                load_btn = gr.Button("Load watchlist", variant="primary", scale=1)
            portfolio_status = gr.Markdown()
            portfolio_tbl = gr.DataFrame(
                headers=[
                    "rank", "ticker", "company_name",
                    "mt_score", "n_dropped", "risk_flag",
                ],
                interactive=False,
                wrap=True,
            )
            drill_down_md = gr.Markdown()

        gr.Markdown(
            "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:8px;'>"
            "Target Tracker — research prototype. Not investment advice."
            "</p>"
        )

        # ── Events: Overview ───────────────────────────────────────────
        refresh_overview_btn.click(
            app.render_overview_html,
            outputs=[overview_panel],
        )

        # ── Events: Data tab ───────────────────────────────────────────
        def _on_upload(files):
            html, msg = ingest_uploads(files, data_dir)
            header = render_step_header(data_dir, cache_ready=not app._CACHE_IS_EMPTY)
            overview = app.render_overview_html()
            return html, msg, header, overview

        upload_btn.click(
            _on_upload,
            inputs=[upload],
            outputs=[data_dashboard, upload_status, step_header, overview_panel],
        )
        refresh_data_btn.click(
            lambda: (
                render_data_dashboard(data_dir),
                render_step_header(data_dir, cache_ready=not app._CACHE_IS_EMPTY),
                app.render_overview_html(),
            ),
            outputs=[data_dashboard, step_header, overview_panel],
        )

        # ── Events: Pipeline tab ───────────────────────────────────────
        refresh_status_btn.click(app._refresh_status_markdown, outputs=[status_md])

        _workflow_refresh_outputs = [
            ticker_dd,
            quarter_dd,
            portfolio_quarter_dd,
            status_md,
            cache_banner,
            step_header,
            data_dashboard,
            overview_panel,
        ]

        def _refresh_all():
            updates = list(app._reload_cache_choices())
            updates.append(render_step_header(data_dir, cache_ready=not app._CACHE_IS_EMPTY))
            updates.append(render_data_dashboard(data_dir))
            updates.append(app.render_overview_html())
            return updates

        run_stage_btn.click(
            app._run_workflow_stage,
            inputs=[stage_dd, extra_args_tb, api_key_tb],
            outputs=[log_box],
        ).then(_refresh_all, outputs=_workflow_refresh_outputs)

        run_all_btn.click(
            app._run_all_stages,
            inputs=[extra_args_tb, api_key_tb],
            outputs=[log_box],
        ).then(_refresh_all, outputs=_workflow_refresh_outputs)

        # ── Events: Entity Report / Watchlist ──────────────────────────
        analyse_btn.click(
            app.analyse_company,
            inputs=[ticker_dd, quarter_dd],
            outputs=tab1_outputs,
            show_progress=True,
        )
        if not app._CACHE_IS_EMPTY:
            demo.load(
                app.analyse_company,
                inputs=[ticker_dd, quarter_dd],
                outputs=tab1_outputs,
            )

        load_btn.click(
            app.load_portfolio,
            inputs=[portfolio_quarter_dd],
            outputs=[portfolio_tbl, portfolio_status],
        )
        if not app._CACHE_IS_EMPTY:
            demo.load(
                app.load_portfolio,
                inputs=[portfolio_quarter_dd],
                outputs=[portfolio_tbl, portfolio_status],
            )
        portfolio_tbl.select(
            app.drill_down_report,
            inputs=[portfolio_tbl],
            outputs=[drill_down_md],
        )

    return demo


def build_interface() -> gr.Blocks:
    import demo.app as app
    return assemble_interface(app)
