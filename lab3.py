# lab3_charts.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Lab 3 — Chart Essentials",page_icon="3️⃣", layout="wide")

# Data helpers
def make_time_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    x = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.sin(np.linspace(0, 6 * math.pi, n)) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"date": x, "value": y})

def make_category_data(seed=1):
    rng = np.random.RandomState(seed)
    cats = list("ABC")
    return pd.DataFrame(
        {"category": np.repeat(cats, 1), "value": rng.randint(10, 100, len(cats))}
    )

def make_points_data(n=600, seed=2):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x": rng.randn(n),
        "y": rng.randn(n) * 0.8 + 0.2 * rng.randn(n),
        "group": rng.choice(list("ABC"), size=n, p=[0.4, 0.35, 0.25])
    })

def make_heat_data(nx=24, ny=12, seed=3):
    rng = np.random.RandomState(seed)
    xs = np.arange(nx)
    ys = np.arange(ny)
    data = [{"x": int(i), "y": int(j), "z": float(rng.rand())} for i in xs for j in ys]
    return pd.DataFrame(data)

# Data
ts = make_time_data()
cats = make_category_data()
pts = make_points_data()
heat = make_heat_data()

st.title("Lab 3 — Chart Essentials")

tab_native, tab_sampler, tab_altair, tab_why = st.tabs(
    ["Native Charts", "Library Sampler", "Altair Reference", "Why Altair?"]
)


# 1) built-in Streamlit charts
with tab_native:
    st.subheader("Quick native charts")
    st.write("Great for a first look; limited customization.")
    st.code(
        "st.line_chart(ts.set_index('date'))\n"
        "st.bar_chart(cats.set_index('category'))",
        language="python",
    )
    st.line_chart(ts.set_index("date"), use_container_width=True)
    st.bar_chart(cats.set_index("category"), use_container_width=True)

# 2) Library Sampler
with tab_sampler:
    st.subheader("Common plotting libraries in Streamlit (sampler)")
    st.caption("Install extras as needed: Plotly, Bokeh.")

    col1, col2, col3 = st.columns(3)

    # Matplotlib
    with col1:
        st.markdown("**Matplotlib / Seaborn**  \n`st.pyplot(fig)`")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(pts["x"], bins=30)
            ax.set_title("Histogram (Matplotlib)")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Matplotlib not available: {e}")

    # Plotly
    with col2:
        st.markdown("**Plotly**  \n`st.plotly_chart(fig)`")
        try:
            import plotly.express as px
            fig = px.scatter(pts, x="x", y="y", color="group", title="Scatter (Plotly)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly not available: {e}")

    # Bokeh
    with col3:
        st.markdown("**Bokeh**  \n`st.bokeh_chart(fig)`")
        try:
            from bokeh.plotting import figure
            p = figure(height=300, width=400, title="Line (Bokeh)")
            p.line(x=np.arange(len(ts)), y=ts["value"])
            st.bokeh_chart(p, use_container_width=True)
        except Exception as e:
            st.warning(f"Bokeh not available: {e}")

    st.divider()
    st.markdown(
        "Other options: **PyDeck** for maps (`st.pydeck_chart`) and **Graphviz** for DAGs "
        "(`st.graphviz_chart`). We’ll standardize on **Altair** below."
    )

# 3) Altair Reference
with tab_altair:
    st.subheader("Altair Reference")

    # Time series (line) + rolling mean + last-point highlight
    st.markdown("**Time series + rolling mean + last-point highlight**")
    ts2 = ts.assign(roll=lambda d: d["value"].rolling(14, min_periods=1).mean())
    line = (
        alt.Chart(ts2)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
            tooltip=["date:T", "value:Q"],
        )
    )
    roll = (
        alt.Chart(ts2)
        .mark_line(strokeDash=[4, 3])
        .encode(x="date:T", y="roll:Q", tooltip=["date:T", "roll:Q"])
    )
    last = (
        alt.Chart(ts2.tail(1))
        .mark_point(size=100)
        .encode(x="date:T", y="value:Q", tooltip=["date:T", "value:Q"])
    )
    st.altair_chart((line + roll + last).properties(height=300), use_container_width=True)

    # Area charts: stacked & stream
    st.markdown("**Area charts: stacked & stream**")
    month = pd.date_range("2024-01-01", periods=6, freq="MS")
    stack_df = pd.DataFrame({
        "month": np.repeat(month, 3),
        "group": list("ABC") * 6,
        "value": np.random.RandomState(42).rand(18) * 100,
    })
    area_stacked = (
        alt.Chart(stack_df)
        .mark_area()
        .encode(
            x="month:T",
            y=alt.Y("value:Q", stack="zero"),
            color="group:N",
            tooltip=["month:T", "group:N", alt.Tooltip("value:Q", format=".1f")],
        )
        .properties(height=260)
    )
    area_stream = (
        alt.Chart(stack_df)
        .mark_area()
        .encode(x="month:T", y=alt.Y("value:Q", stack="center"), color="group:N")
        .properties(height=260)
    )
    st.altair_chart(area_stacked, use_container_width=True)
    st.altair_chart(area_stream, use_container_width=True)

    # Bar charts: grouped / stacked / normalized / horizontal
    st.markdown("**Bar charts: grouped, stacked, normalized, horizontal**")
    grouped = (
        pts.assign(count=1)
        .groupby(["group"])
        .agg(avg_x=("x", "mean"), n=("count", "sum"))
        .reset_index()
    )
    bars_grouped = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("group:N", title="Group"),
            y=alt.Y("avg_x:Q", title="Avg X"),
            tooltip=["group", alt.Tooltip("avg_x:Q", format=".2f")],
        )
        .properties(height=220)
    )
    stacked = (
        alt.Chart(stack_df)
        .mark_bar()
        .encode(
            x="month:T",
            y=alt.Y("value:Q", stack="zero"),
            color="group:N",
            tooltip=["month:T", "group:N", alt.Tooltip("value:Q", format=".1f")],
        )
        .properties(height=220)
    )
    normalized = stacked.encode(y=alt.Y("value:Q", stack="normalize", title="Share"))
    barh = (
        alt.Chart(cats)
        .mark_bar()
        .encode(y="category:N", x="value:Q", tooltip=["category", "value"])
        .properties(height=160)
    )
    st.altair_chart(bars_grouped, use_container_width=True)
    st.altair_chart(stacked, use_container_width=True)
    st.altair_chart(normalized, use_container_width=True)
    st.altair_chart(barh, use_container_width=True)

    # Lollipop chart (category vs value)
    st.markdown("**Lollipop chart (category vs value)**")
    lollipop_rule = (
        alt.Chart(cats)
        .mark_rule()
        .encode(y="category:N", x=alt.value(0), x2="value:Q")
    )
    lollipop_dot = (
        alt.Chart(cats)
        .mark_circle(size=140)
        .encode(y="category:N", x="value:Q", tooltip=["category", "value"])
    )
    st.altair_chart((lollipop_rule + lollipop_dot).properties(height=180), use_container_width=True)

    # Pie & Donut charts
    st.markdown("**Pie & Donut**")
    col_pie, col_donut = st.columns(2)
    with col_pie:
        pie = (
            alt.Chart(cats)
            .mark_arc()
            .encode(
                theta=alt.Theta("value:Q", stack=True),
                color="category:N",
                tooltip=["category", "value"],
            )
            .properties(height=260)
        )
        st.altair_chart(pie, use_container_width=True)
    with col_donut:
        donut = (
            alt.Chart(cats)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("value:Q", stack=True),
                color="category:N",
                tooltip=["category", "value"],
            )
            .properties(height=260)
        )
        st.altair_chart(donut, use_container_width=True)

    # Distribution: Histogram, Box, Density curve
    st.markdown("**Distribution: histogram, box, density**")
    hist = (
        alt.Chart(pts)
        .mark_bar()
        .encode(
            x=alt.X("x:Q", bin=alt.Bin(maxbins=40), title="x"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=220)
    )
    box = alt.Chart(pts).mark_boxplot().encode(y=alt.Y("x:Q", title="x")).properties(height=160)
    density = (
        alt.Chart(pts)
        .transform_density("x", as_=["x", "density"])
        .mark_area(opacity=0.5)
        .encode(x="x:Q", y="density:Q")
        .properties(height=180)
    )
    st.altair_chart(hist, use_container_width=True)
    st.altair_chart(box, use_container_width=True)
    st.altair_chart(density, use_container_width=True)

    # Relationship: scatter + regression trend
    st.markdown("**Relationship: scatter + regression trend**")
    scatter = (
        alt.Chart(pts)
        .mark_point(opacity=0.7)
        .encode(x="x:Q", y="y:Q", color="group:N", tooltip=["x:Q", "y:Q", "group:N"])
        .properties(height=300)
    )
    trend = (
        alt.Chart(pts)
        .transform_regression("x", "y")
        .mark_line()
        .encode(x="x:Q", y="y:Q")
        .properties(height=300)
    )
    st.altair_chart(scatter + trend, use_container_width=True)

    # Heatmap
    st.markdown("**Heatmap**")
    heatmap = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("x:O", title="X"),
            y=alt.Y("y:O", title="Y"),
            color=alt.Color("z:Q", title="Intensity"),
            tooltip=["x:O", "y:O", alt.Tooltip("z:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(heatmap, use_container_width=True)

    # Error bars (mean ± stdev) with mean points
    st.markdown("**Error bars (mean ± stdev) with mean points**")
    err = (
        alt.Chart(pts)
        .mark_errorbar(extent="stdev")
        .encode(x="group:N", y="y:Q")
        .properties(height=240)
    )
    mean_pts = (
        alt.Chart(pts)
        .mark_point(color="black")
        .encode(x="group:N", y="mean(y):Q", tooltip=[alt.Tooltip("mean(y):Q", title="Mean")])
        .properties(height=240)
    )
    st.altair_chart(err + mean_pts, use_container_width=True)

    # Small multiples (faceted)
    st.markdown("**Small multiples (facet by group)**")
    base_facet = (
        alt.Chart(pts)
        .mark_point(opacity=0.6)
        .encode(x="x:Q", y="y:Q")
        .properties(width=150, height=150)
    )
    facet_chart = base_facet.facet(column="group:N")
    st.altair_chart(facet_chart, use_container_width=True)

# 4) WHY ALTAIR?
with tab_why:
    st.subheader("Why we’re going with Altair")
    st.markdown(
        """
- **Declarative grammar**: focus on *what* to encode (x/y/color/size), not drawing commands.
- **Concise, readable code**: ideal for teaching and copy-paste references.
- **Good defaults**: axes, legends, tooltips look right out of the box.
- **Transforms**: aggregate/bin/window/calculate/regression without extra Python.
- **Composability**: layer, facet, and build multi-panel visuals quickly.
        """
    )
    st.markdown("**Covered here:** core charts, stacked/stream area, pie/donut, lollipop, density, heatmap, error bars, facets.")
    st.info("Need 3D, candlestick/treemap/sunburst/sankey? Use Plotly. Need 3D maps/large point clouds? Use PyDeck.")
