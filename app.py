"""Flask web application for the Zero of Functions (ZOF) solver."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

from flask import Flask, flash, render_template, request

import methods

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-to-a-secure-key"

METHOD_OPTIONS = {
    "bisection": "Bisection Method",
    "regula_falsi": "Regula Falsi Method",
    "secant": "Secant Method",
    "newton_raphson": "Newton-Raphson Method",
    "fixed_point": "Fixed Point Iteration",
    "modified_secant": "Modified Secant Method",
}


def parse_float(field: str, label: str, required: bool = True) -> float | None:
    value = request.form.get(field, "").strip()
    if not value:
        if required:
            raise ValueError(f"{label} is required.")
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc


@app.route("/", methods=["GET", "POST"])
def index():
    result_context: Dict[str, object] = {
        "method_options": METHOD_OPTIONS,
        "form_values": request.form if request.method == "POST" else {},
        "result": None,
    }

    if request.method == "POST":
        method_key = request.form.get("method", "").strip()
        if method_key not in METHOD_OPTIONS:
            flash("Please select a numerical method.", "error")
            return render_template("index.html", **result_context)

        try:
            tolerance = parse_float("tolerance", "Tolerance")
            max_iterations = int(parse_float("max_iterations", "Maximum Iterations"))
            if max_iterations <= 0:
                raise ValueError("Maximum iterations must be positive.")
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("index.html", **result_context)

        try:
            if method_key in {"bisection", "regula_falsi"}:
                func = methods.parse_function(request.form.get("equation", ""))
                lower = parse_float("lower_bound", "Lower bound a")
                upper = parse_float("upper_bound", "Upper bound b")
                if method_key == "bisection":
                    result = methods.bisection(func, lower, upper, tolerance, max_iterations)
                else:
                    result = methods.regula_falsi(func, lower, upper, tolerance, max_iterations)

            elif method_key == "secant":
                func = methods.parse_function(request.form.get("equation", ""))
                x0 = parse_float("x0", "First guess x0")
                x1 = parse_float("x1", "Second guess x1")
                result = methods.secant(func, x0, x1, tolerance, max_iterations)

            elif method_key == "newton_raphson":
                func = methods.parse_function(request.form.get("equation", ""))
                derivative = methods.parse_function(request.form.get("derivative", ""))
                x0 = parse_float("x0", "Initial guess x0")
                result = methods.newton_raphson(func, derivative, x0, tolerance, max_iterations)

            elif method_key == "fixed_point":
                g_func = methods.parse_function(request.form.get("g_equation", ""))
                x0 = parse_float("x0", "Initial guess x0")
                result = methods.fixed_point(g_func, x0, tolerance, max_iterations)

            elif method_key == "modified_secant":
                func = methods.parse_function(request.form.get("equation", ""))
                x0 = parse_float("x0", "Initial guess x0")
                delta = parse_float("delta", "Delta parameter")
                result = methods.modified_secant(func, x0, delta, tolerance, max_iterations)

            else:
                raise ValueError("Unsupported method selected.")

        except (ValueError, methods.FunctionParserError) as exc:
            flash(str(exc), "error")
            return render_template("index.html", **result_context)

        result_context["result"] = asdict(result)
        result_context["selected_method"] = method_key

    return render_template("index.html", **result_context)


if __name__ == "__main__":
    app.run(debug=True)
