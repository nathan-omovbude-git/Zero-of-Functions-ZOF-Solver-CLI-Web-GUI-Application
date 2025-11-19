"""Numerical methods utilities for the Zero of Functions (ZOF) solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import math


AllowedFunction = Callable[[float], float]


@dataclass
class MethodResult:
    """Container for an iterative method result."""

    iterations: List[Dict[str, float]]
    root: float
    error: float
    iterations_used: int


class FunctionParserError(ValueError):
    """Raised when a user-provided expression cannot be parsed or evaluated."""


def parse_function(expression: str) -> AllowedFunction:
    """Convert the user input expression into a callable function of x."""

    expression = expression.strip()
    if not expression:
        raise FunctionParserError("Expression cannot be empty.")

    allowed_names = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    allowed_names["abs"] = abs

    try:
        compiled = compile(expression, "<user_function>", "eval")
    except SyntaxError as exc:
        raise FunctionParserError(f"Invalid expression: {exc.msg}") from exc

    def func(x: float) -> float:
        local_context = {**allowed_names, "x": x}
        try:
            value = eval(compiled, {"__builtins__": {}}, local_context)
        except Exception as exc:
            raise FunctionParserError(f"Error evaluating expression at x={x}: {exc}") from exc
        if isinstance(value, complex):
            raise FunctionParserError("Expression evaluated to a complex number; real value expected.")
        return float(value)

    return func


def _build_iteration(iteration: int, xn: float, fxn: float, error: float) -> Dict[str, float]:
    return {
        "iteration": iteration,
        "xn": xn,
        "fxn": fxn,
        "error": error,
    }


def bisection(func: AllowedFunction, a: float, b: float, tol: float, max_iter: int) -> MethodResult:
    fa = func(a)
    fb = func(b)
    if fa * fb >= 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")

    iterations: List[Dict[str, float]] = []
    for iteration in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = func(c)
        error = abs(b - a) / 2
        iterations.append(_build_iteration(iteration, c, fc, error))

        if error < tol or abs(fc) < tol:
            return MethodResult(iterations, c, error, iteration)

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return MethodResult(iterations, c, error, max_iter)


def regula_falsi(func: AllowedFunction, a: float, b: float, tol: float, max_iter: int) -> MethodResult:
    fa = func(a)
    fb = func(b)
    if fa * fb >= 0:
        raise ValueError("Regula Falsi requires f(a) and f(b) to have opposite signs.")

    iterations: List[Dict[str, float]] = []
    c = a
    for iteration in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = func(c)
        error = abs(fc)
        iterations.append(_build_iteration(iteration, c, fc, error))

        if abs(fc) < tol or error < tol:
            return MethodResult(iterations, c, error, iteration)

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return MethodResult(iterations, c, error, max_iter)


def secant(func: AllowedFunction, x0: float, x1: float, tol: float, max_iter: int) -> MethodResult:
    iterations: List[Dict[str, float]] = []
    prev = x0
    curr = x1

    for iteration in range(1, max_iter + 1):
        f_prev = func(prev)
        f_curr = func(curr)
        denominator = f_curr - f_prev
        if denominator == 0:
            raise ValueError("Secant method encountered zero denominator; choose different initial guesses.")

        next_x = curr - f_curr * (curr - prev) / denominator
        error = abs(next_x - curr)
        iterations.append(_build_iteration(iteration, next_x, func(next_x), error))

        if error < tol or abs(func(next_x)) < tol:
            return MethodResult(iterations, next_x, error, iteration)

        prev, curr = curr, next_x

    return MethodResult(iterations, curr, error, max_iter)


def newton_raphson(func: AllowedFunction, derivative: AllowedFunction, x0: float, tol: float, max_iter: int) -> MethodResult:
    iterations: List[Dict[str, float]] = []
    current = x0

    for iteration in range(1, max_iter + 1):
        f_val = func(current)
        derivative_val = derivative(current)
        if derivative_val == 0:
            raise ValueError("Derivative is zero; Newton-Raphson cannot proceed.")

        next_x = current - f_val / derivative_val
        error = abs(next_x - current)
        iterations.append(_build_iteration(iteration, next_x, func(next_x), error))

        if error < tol or abs(func(next_x)) < tol:
            return MethodResult(iterations, next_x, error, iteration)

        current = next_x

    return MethodResult(iterations, current, error, max_iter)


def fixed_point(g_func: AllowedFunction, x0: float, tol: float, max_iter: int) -> MethodResult:
    iterations: List[Dict[str, float]] = []
    current = x0

    for iteration in range(1, max_iter + 1):
        next_x = g_func(current)
        error = abs(next_x - current)
        f_val = next_x - current
        iterations.append(_build_iteration(iteration, next_x, f_val, error))

        if error < tol:
            return MethodResult(iterations, next_x, error, iteration)

        current = next_x

    return MethodResult(iterations, current, error, max_iter)


def modified_secant(func: AllowedFunction, x0: float, delta: float, tol: float, max_iter: int) -> MethodResult:
    if delta == 0:
        raise ValueError("Delta must be non-zero for the modified secant method.")

    iterations: List[Dict[str, float]] = []
    current = x0

    for iteration in range(1, max_iter + 1):
        f_current = func(current)
        denominator = func(current + delta * current) - f_current
        if denominator == 0:
            raise ValueError("Modified secant encountered zero denominator; adjust delta or initial guess.")

        next_x = current - (delta * current * f_current) / denominator
        error = abs(next_x - current)
        iterations.append(_build_iteration(iteration, next_x, func(next_x), error))

        if error < tol or abs(func(next_x)) < tol:
            return MethodResult(iterations, next_x, error, iteration)

        current = next_x

    return MethodResult(iterations, current, error, max_iter)

