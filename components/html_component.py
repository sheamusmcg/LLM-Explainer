"""Bridge between Streamlit and D3.js visualizations.

Renders JavaScript visualization components inside Streamlit via st.components.v1.html().
Data flows one-directionally: Python serializes data as JSON into the HTML template,
and D3.js reads it on load to render the visualization.
"""

import json
import os
import streamlit as st
import streamlit.components.v1 as components

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


@st.cache_data
def _read_file(path):
    """Read and cache a static file."""
    with open(path, "r") as f:
        return f.read()


def render_component(component_name, props, height=600):
    """Render a D3.js visualization component in Streamlit.

    Args:
        component_name: Name of the JS file in static/js/ (without .js extension).
        props: Dict of data to pass to the JS component (serialized as JSON).
        height: Height of the iframe in pixels.
    """
    d3_js = _read_file(os.path.join(_STATIC_DIR, "js", "lib", "d3.v7.min.js"))
    utils_js = _read_file(os.path.join(_STATIC_DIR, "js", "utils.js"))
    component_js = _read_file(os.path.join(_STATIC_DIR, "js", f"{component_name}.js"))
    css = _read_file(os.path.join(_STATIC_DIR, "css", "components.css"))

    # Serialize props, handling numpy arrays by converting to lists
    props_json = json.dumps(props, default=_json_serializer)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
<div id="viz-container"></div>
<script>{d3_js}</script>
<script id="component-data" type="application/json">{props_json}</script>
<script>{utils_js}</script>
<script>{component_js}</script>
</body>
</html>"""

    components.html(html, height=height, scrolling=True)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
