"""Gradio theme and global CSS for Target Tracker."""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter")],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
    ).set(
        body_background_fill="#eef1f6",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f4f6fa",
        block_background_fill="#ffffff",
        block_border_color="#dde3ec",
        block_label_background_fill="#ffffff",
        block_title_text_color="#0b1220",
        body_text_color="#1a2332",
        body_text_color_subdued="#5c6b7f",
        button_primary_background_fill="#0f4c81",
        button_primary_background_fill_hover="#0a3d68",
        button_primary_text_color="#ffffff",
        input_background_fill="#ffffff",
        input_border_color="#c5ced9",
    )


APP_CSS = """
:root, html, body, .gradio-container {
    color-scheme: light !important;
    background: #eef1f6 !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 20px 32px 64px !important;
}

/* ── Hero / header ─────────────────────────────────────────────── */
.app-hero {
    background: linear-gradient(120deg, #071525 0%, #0f2d4a 42%, #0f4c81 100%);
    color: #f0f4f8;
    border-radius: 18px;
    padding: 30px 34px 26px;
    margin-bottom: 22px;
    box-shadow: 0 14px 40px rgba(7, 21, 37, 0.18);
    position: relative;
    overflow: hidden;
}

.app-hero::after {
    content: "";
    position: absolute;
    top: -40%;
    right: -8%;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(56,189,248,0.14) 0%, transparent 70%);
    pointer-events: none;
}

.app-hero .brand-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}

.app-hero h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.03em;
}

.app-hero .tagline {
    color: #94a8bc !important;
    margin: 0 !important;
    font-size: 0.95rem;
    max-width: 560px;
}

.app-hero .corpus-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: rgba(56,189,248,0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.28);
}

.step-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: #b8c8d8;
}

.step-pill.done { background: rgba(52,211,153,0.16); color: #a7f3d0; border-color: rgba(52,211,153,0.32); }
.step-pill.active { background: rgba(56,189,248,0.18); color: #bae6fd; border-color: rgba(56,189,248,0.38); }
.step-pill.pending { opacity: 0.7; }

/* ── KPI grid ─────────────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.kpi-card .kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 6px;
}

.kpi-card .kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0b1220;
    line-height: 1.1;
}

.kpi-card .kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
}

.kpi-card.accent { border-color: #93c5fd; background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%); }
.kpi-card.warn { border-color: #fcd34d; background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%); }
.kpi-card.ok { border-color: #86efac; background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%); }

/* ── Panels ───────────────────────────────────────────────────── */
.panel {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

.panel-title {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 12px;
}

.panel-subtitle {
    font-size: 0.88rem;
    color: #475569;
    margin: -6px 0 14px 0;
}

.two-col {
    display: grid;
    grid-template-columns: 1fr 1.1fr;
    gap: 16px;
}

@media (max-width: 900px) {
    .two-col { grid-template-columns: 1fr; }
}

.pipeline-controls {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* ── File cards ───────────────────────────────────────────────── */
.file-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
}

.file-card {
    border: 1px solid #dde3ec;
    border-radius: 12px;
    padding: 14px 16px;
    background: #fafbfc;
}

.file-card.ready { border-color: #86efac; background: #f0fdf4; }
.file-card.missing { border-color: #fca5a5; background: #fff7f7; }
.file-card.optional-missing { border-color: #dde3ec; background: #fafbfc; }

.file-card h4 {
    margin: 0 0 6px 0;
    font-size: 0.93rem;
    color: #0b1220;
}

.file-card .meta {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge.ready { background: #dcfce7; color: #166534; }
.badge.missing { background: #fee2e2; color: #991b1b; }
.badge.optional { background: #e2e8f0; color: #475569; }

/* ── Overview / watchlist tables ──────────────────────────────── */
.mini-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}

.mini-table th {
    text-align: left;
    padding: 8px 10px;
    color: #64748b;
    font-weight: 600;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.mini-table td {
    padding: 9px 10px;
    border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
}

.mini-table tr:hover td { background: #f8fafc; }

.flag-high { color: #dc2626; font-weight: 700; }
.flag-medium { color: #d97706; font-weight: 700; }
.flag-low { color: #16a34a; font-weight: 700; }

.empty-state {
    border: 1px dashed #c5ced9;
    border-radius: 12px;
    padding: 32px 24px;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
    font-size: 0.92rem;
}

.quickstart-list {
    margin: 0;
    padding-left: 1.2rem;
    color: #334155;
    line-height: 1.75;
    font-size: 0.9rem;
}

/* ── Tabs & misc ──────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid #dde3ec !important;
    margin-bottom: 6px !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748b !important;
}

.tab-nav button.selected {
    color: #0f4c81 !important;
    border-bottom: 2px solid #0f4c81 !important;
}

.log-panel textarea {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.8rem !important;
    background: #0b1220 !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    line-height: 1.5 !important;
}

.entity-report-header {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}

footer.svelte-1ipelgc, .footer {
    opacity: 0.5;
}
"""
