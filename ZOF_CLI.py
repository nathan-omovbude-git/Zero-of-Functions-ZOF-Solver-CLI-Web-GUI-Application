"""Command-line interface for the Zero of Functions (ZOF) solver."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from tabulate import tabulate

import methods


def prompt_float(message: str) -> float:
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Invalid number. Please try again.")


def prompt_int(message: str) -> int:
    while True:
        try:
            value = int(input(message))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Invalid integer. Please enter a positive integer.")


def prompt_expression(message: str) -> Callable[[float], float]:
    while True:
        expression = input(message)
        try:
            return methods.parse_function(expression)
        except methods.FunctionParserError as exc:
            print(f"Error: {exc}")


def select_method() -> str:
    menu = {
        "1": "bisection",
        "2": "regula_falsi",
        "3": "secant",
        "4": "newton_raphson",
        "5": "fixed_point",
        "6": "modified_secant",
    }

    print("\nZero of Functions (ZOF) Solver CLI")
    print("Select a method:")
    print("1. Bisection Method")
    print("2. Regula Falsi Method")
    print("3. Secant Method")
    print("4. Newton-Raphson Method")
    print("5. Fixed Point Iteration")
    print("6. Modified Secant Method")

    while True:
        choice = input("Enter choice (1-6): ").strip()
        if choice in menu:
            return menu[choice]
        print("Invalid choice. Please select a number between 1 and 6.")


def display_result(result: methods.MethodResult) -> None:
    table_data = [
        [row["iteration"], row["xn"], row["fxn"], row["error"]]
        for row in result.iterations
    ]
    headers = ["Iteration", "x_n", "f(x_n)", "Error"]
    print("\nIteration Details:")
    print(tabulate(table_data, headers=headers, floatfmt=".8f"))
    print("\nFinal Estimate:")
    print(f"Root approximation: {result.root:.10f}")
    print(f"Final error estimate: {result.error:.10f}")
    print(f"Iterations used: {result.iterations_used}")


def run_cli() -> None:
    while True:
        method_key = select_method()
        tolerance = prompt_float("Enter tolerance (e.g., 0.0001): ")
        max_iterations = prompt_int("Enter maximum iterations: ")

        try:
            if method_key in {"bisection", "regula_falsi"}:
                func = prompt_expression("Enter f(x) = 0 equation (in terms of x): ")
                lower = prompt_float("Enter lower bound a: ")
                upper = prompt_float("Enter upper bound b: ")
                if method_key == "bisection":
                    result = methods.bisection(func, lower, upper, tolerance, max_iterations)
                else:
                    result = methods.regula_falsi(func, lower, upper, tolerance, max_iterations)

            elif method_key == "secant":
                func = prompt_expression("Enter f(x) = 0 equation (in terms of x): ")
                x0 = prompt_float("Enter first initial guess x0: ")
                x1 = prompt_float("Enter second initial guess x1: ")
                result = methods.secant(func, x0, x1, tolerance, max_iterations)

            elif method_key == "newton_raphson":
                func = prompt_expression("Enter f(x) = 0 equation (in terms of x): ")
                derivative = prompt_expression("Enter derivative f'(x): ")
                x0 = prompt_float("Enter initial guess x0: ")
                result = methods.newton_raphson(func, derivative, x0, tolerance, max_iterations)

            elif method_key == "fixed_point":
                g_func = prompt_expression("Enter g(x) for fixed-point iteration: ")
                x0 = prompt_float("Enter initial guess x0: ")
                result = methods.fixed_point(g_func, x0, tolerance, max_iterations)

            elif method_key == "modified_secant":
                func = prompt_expression("Enter f(x) = 0 equation (in terms of x): ")
                x0 = prompt_float("Enter initial guess x0: ")
                delta = prompt_float("Enter delta (e.g., 0.01): ")
                result = methods.modified_secant(func, x0, delta, tolerance, max_iterations)

            else:
                raise ValueError("Unsupported method selected.")

            display_result(result)
        except ValueError as exc:
            print(f"Error: {exc}")

        continue_choice = input("\nRun another calculation? (y/n): ").strip().lower()
        if continue_choice != "y":
            print("Thank you for using the ZOF Solver CLI.")
            break


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
