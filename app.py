"""Flask entry point. Routes only; all logic lives in src/."""
import json
import logging
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, flash, redirect, render_template, request, session, url_for

from src import config
from src.insights import generate_insights
from src.kpi import category_performance, monthly_trend, overall_kpis
from src.predict import LatePredictor, PredictionError
from src.scoring import COMPONENTS, score_vendors, validate_weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vendor-app")

PALETTE = {"Good": "#2e9e6b", "Average": "#4f7cff", "Average 1": "#4f7cff", "Average 2": "#7aa0ff",
           "Watchlist": "#f2a33a", "Risky": "#e5534b"}
LAYOUT = dict(margin=dict(l=45, r=15, t=35, b=40), paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", size=12))


def to_json(fig: go.Figure) -> str:
    fig.update_layout(**LAYOUT)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def load_state() -> dict:
    """Read processed data and saved models once at startup (no retraining)."""
    orders = pd.read_csv(config.CLEAN_CSV, parse_dates=["order_date"])
    kpis = pd.read_csv(config.KPI_CSV)
    state = {
        "orders": orders, "kpis": kpis,
        "groups": kpis.set_index("vendor_id")["risk_group"].to_dict(),
        "cluster_summary": pd.read_csv(config.CLUSTER_SUMMARY_CSV),
        "importance": pd.read_csv(config.PERMUTATION_IMPORTANCE_CSV),
        "metrics": json.loads(config.METRICS_JSON.read_text()),
        "predictor": LatePredictor(),
    }
    state["categories"] = category_performance(orders)
    state["monthly"] = monthly_trend(orders)
    return state


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
    try:
        state = load_state()
    except FileNotFoundError:
        log.error("Processed data or models missing. Run `python train.py` first.")
        state = None

    def current_weights() -> dict:
        return session.get("weights", dict(config.DEFAULT_WEIGHTS))

    def scored_table(weights: dict) -> pd.DataFrame:
        scored = score_vendors(state["kpis"], weights)
        scored["risk_group"] = scored["vendor_id"].map(state["groups"])
        return scored

    @app.before_request
    def guard():
        if state is None and request.endpoint != "static":
            return render_template("error.html", message="Models not trained yet. Run `python train.py`."), 503

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", message="Something went wrong. Please try again."), 500

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            if "reset" in request.form:
                session.pop("weights", None)
            else:
                try:
                    weights = validate_weights({k: request.form.get(k, "") for k in config.DEFAULT_WEIGHTS})
                    session["weights"] = weights
                    flash("Weights applied. Scores and ranking recalculated.", "ok")
                except ValueError as exc:
                    flash(str(exc), "error")
            return redirect(url_for("index"))

        weights = current_weights()
        scored = scored_table(weights)
        cards = overall_kpis(state["orders"], scored)
        cards["avg_score"] = round(scored["vendor_score"].mean(), 1)

        top = scored.head(10).sort_values("vendor_score")
        rank_fig = go.Figure(go.Bar(x=top["vendor_score"], y=top["vendor_name"], orientation="h",
                                    marker_color="#4f7cff"))
        rank_fig.update_layout(title="Top 10 vendors by score", xaxis_title="Score (0-100)")

        dist = scored["risk_group"].value_counts()
        cluster_fig = go.Figure(go.Pie(labels=dist.index, values=dist.values, hole=0.5,
                                       marker_colors=[PALETTE.get(g, "#999") for g in dist.index]))
        cluster_fig.update_layout(title="Vendors by risk group")

        m = state["monthly"]
        trend_fig = go.Figure()
        trend_fig.add_bar(x=m["order_year_month"], y=m["orders"], name="Orders", marker_color="#c9d6ff", yaxis="y2")
        trend_fig.add_scatter(x=m["order_year_month"], y=m["late_rate"], name="Late %", line=dict(color="#e5534b"))
        trend_fig.update_layout(title="Monthly late-delivery rate and order volume",
                                yaxis=dict(title="Late %"), yaxis2=dict(overlaying="y", side="right", showgrid=False),
                                legend=dict(orientation="h", y=-0.25))

        c = state["categories"].sort_values("late_rate")
        cat_fig = go.Figure(go.Bar(x=c["late_rate"], y=c["category"], orientation="h", marker_color="#f2a33a"))
        cat_fig.update_layout(title="Late-delivery rate by category (%)")

        problem = scored.sort_values(["vendor_score"]).head(5)
        insights = generate_insights(scored, state["categories"], m, state["importance"])
        return render_template(
            "index.html", cards=cards, weights=weights, table=scored.to_dict("records"), problem=problem.to_dict("records"),
            insights=insights, active="home",
            charts={"rank": to_json(rank_fig), "cluster": to_json(cluster_fig),
                    "trend": to_json(trend_fig), "category": to_json(cat_fig)},
            cluster_summary=state["cluster_summary"].to_dict("records"))

    @app.route("/vendors")
    def vendors():
        scored = scored_table(current_weights())
        choices = scored.sort_values("vendor_id")[["vendor_id", "vendor_name"]].to_dict("records")
        vendor_id = request.args.get("vendor_id", choices[0]["vendor_id"]).upper()
        row = scored[scored["vendor_id"] == vendor_id]
        if row.empty:
            flash("Vendor not found.", "error")
            return redirect(url_for("vendors"))
        v = row.iloc[0].to_dict()
        orders = state["orders"][state["orders"]["vendor_id"] == vendor_id]

        monthly = orders.groupby("order_year_month").agg(
            late=("is_late", lambda s: s.mean() * 100), delay=("delay_days_positive", "mean"),
            n=("order_id", "count")).reset_index()
        trend_fig = go.Figure(go.Scatter(x=monthly["order_year_month"], y=monthly["late"], mode="lines+markers",
                                         line=dict(color="#e5534b")))
        trend_fig.update_layout(title="Monthly late-delivery rate (%)")

        comp = list(COMPONENTS)
        vals = [v[f"{c}_score"] for c in comp]
        avg = [scored[f"{c}_score"].mean() for c in comp]
        comp_fig = go.Figure()
        comp_fig.add_bar(x=[c.title() for c in comp], y=vals, name=v["vendor_name"], marker_color="#4f7cff")
        comp_fig.add_bar(x=[c.title() for c in comp], y=avg, name="Vendor average", marker_color="#c9d6ff")
        comp_fig.update_layout(title="Component scores vs average", barmode="group", legend=dict(orientation="h", y=-0.25))

        hist_fig = go.Figure(go.Histogram(x=orders["delivery_delay_days"], marker_color="#f2a33a", nbinsx=25))
        hist_fig.update_layout(title="Delivery delay distribution (days; negative = early)")

        return render_template("vendors.html", choices=choices, v=v, active="vendors", vendor_id=vendor_id,
                               charts={"trend": to_json(trend_fig), "comp": to_json(comp_fig), "hist": to_json(hist_fig)},
                               group_color=PALETTE.get(v["risk_group"], "#999"))

    @app.route("/prediction", methods=["GET", "POST"])
    def prediction():
        predictor = state["predictor"]
        form = request.form.to_dict() if request.method == "POST" else {}
        result = None
        if request.method == "POST":
            try:
                result = predictor.predict(form)
            except PredictionError as exc:
                flash(str(exc), "error")
            except Exception:
                log.exception("prediction failed")
                flash("Could not score this order. Please check the inputs and try again.", "error")
        auc = state["metrics"]["final"]["at_operating_threshold"]["roc_auc"]
        return render_template("prediction.html", predictor=predictor, form=form, result=result, active="prediction",
                               defaults=predictor.market_price, leads=predictor.median_lead, auc=auc,
                               threshold=int(predictor.threshold * 100))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
