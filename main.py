#--------------------------------------------------------------
# Author: Arvin Javaheripur
# Date: 2026-03-5
# License: GNU General Public License v3.0
# Description: Copyright (C) 2026 Arvin Javaheripur
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#--------------------------------------------------------------


import numpy as np
import matplotlib
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from math import e

intenv = False
try:
	__file__
except:
	intenv = True
	print("!")

if not intenv:
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	matplotlib.use("TkAgg")
    

# ==========================
# Model functions
# ==========================
def linear_func(x, m, b):
    return m * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    x_safe = np.where(x <= 0, 1e-10, x)
    return a * np.log(x_safe) + b

def format_equation(name, params=None):
    try:
        if name == "Linear":
            m, b = params
            b_sign = "+" if b >= 0 else "-"
			b = abs(b)
            return f"y = {m:.5f}x {b_sign} {b:.5f}"
        elif name == "Quadratic":
            a, b, c = params
            b_sign = "+" if b >= 0 else "-"
            c_sign = "+" if c >= 0 else "-"
			b = abs(b)
			c = abs(c)
            return f"y = {a:.5f}x² {b_sign} {b:.5f}x {c_sign} {c:.5f}"
        elif name == "Exponential":
            a, b = params
            base = e ** b
            return f"y = {a:.5f} * {base:.5f}^x\n          y = {a:.5f} * e^({b:.5f}x)"
        elif name == "Logarithmic":
            a, b = params
            b_sign = "+" if b >= 0 else "-"
			b = abs(b)
            return f"y = {a:.5f} ln(x) {b_sign} {b:.5f}"
        else:
            return "Unknown model"
    except Exception:
        return "Equation formatting error"

#=================================
# Parse data
#=================================

def parse_data(raw_data):
    valid_data = []
    invalid_count = 0
    seen = set()
    duplicate_found = False

    for line in raw_data.splitlines():
        parts = re.split("[,\\t\\s:]+", line.strip())
        if len(parts) >= 2:
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                tup = (x_val, y_val)
                if tup in seen:
                    duplicate_found = True
                seen.add(tup)
                valid_data.append(tup)
            except ValueError:
                invalid_count += 1
        else:
            invalid_count += 1

    return np.array(valid_data), invalid_count, duplicate_found

def get_data():
    while True:
        print("\nSelect data input method:")
        print(" [1] Load from a file")
        print(" [2] Enter data manually")
        data_choice = input("Enter your choice [1-2]: ").strip()

        if data_choice == "1":
            try:
                path = input("Enter file path: ").strip()
                with open(path, "r") as f:
                    raw_data = f.read()
            except Exception as e:
                print(f"[!] Error reading file: {e}")
                continue

        elif data_choice == "2":
            print("\nEnter your data as x, y pairs (space, tab, or comma separated).")
            print("Press Enter twice when done:")

            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            if not lines:
                print("[!] No data entered.")
                continue

            raw_data = "\n".join(lines)

        else:
            print("[!] Invalid choice.")
            continue

        data, invalid_count, duplicate_found = parse_data(raw_data)

        if invalid_count > 0:
            print(f"[!] ERROR: {invalid_count} invalid line(s) detected.")
            continue

        if len(data) < 2:
            print("[!] At least two data points required.")
            continue

        return data

data = get_data()
x = data[:, 0]
y = data[:, 1]

#=================================
# Manually compute linear
#=================================

try:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    Sxx = np.sum((x - x_mean) ** 2)
    Syy = np.sum((y - y_mean) ** 2)

    if Sxx == 0:
        raise ValueError("All x values identical.")

    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    r2_linear = (Sxy / np.sqrt(Sxx * Syy)) ** 2
except Exception:
    slope, intercept, r2_linear = 0, 0, -np.inf

#=================================
# Fit other models with Scipi
#=================================

models = {
    "Quadratic": (quadratic, [1, 1, 1]),
    "Exponential": (exponential, [1, 0.1]),
    "Logarithmic": (logarithmic, [1, 1])
}

results = {
    "Linear": (linear_func, (slope, intercept), r2_linear)
}

for name, (func, p0) in models.items():
    try:
        params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[name] = (func, params, r2)
    except Exception:
        results[name] = (func, None, -np.inf)


#=================================
# Rank models
#=================================

sorted_models = sorted(results.items(), key=lambda x: x[1][2], reverse=True)

r2_linear = results["Linear"][2]
r2_quad = results["Quadratic"][2]

if 0 < r2_quad - r2_linear < 0.02:
    for i, (name, data) in enumerate(sorted_models):
        if name == "Linear":
            linear_model = sorted_models.pop(i)
            sorted_models.insert(0, linear_model)
            break

#=================================
# Print results
#=================================

print("\n=== Model Results ===")
for name, (func, params, r2) in sorted_models:
    eq = format_equation(name, params) if params is not None else "Fit failed"
    print(f"\n{name} Model:")
    print(f"Equation: {eq}")
    print(f"Coefficient of Determination (R²): {round(r2, 5)}")

best_name, (best_func, best_params, best_r2) = sorted_models[0]

print("\n=== Best Model ===")
eq = format_equation(best_name, best_params) if best_params is not None else "Fit failed"
print(f"Model: {best_name}")
print(f"Equation: {eq}")
print(f"Coefficient of Determination (R²): {round(best_r2, 5)}")


#=================================
# Plot models
#=================================

def plot_model(name, func, params, r2):
    try:
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = func(x_smooth, *params)
        y_pred = func(x, *params)
        residuals = np.round(y - y_pred, 5)

        plt.figure()
        plt.suptitle(f"{name} Model Fit")

        plt.subplot(2, 1, 1)
        plt.scatter(x, y, label="Data")
        plt.plot(x_smooth, y_smooth, color="red", label="Fit")
        plt.title(f"{name} Fit (R² = {r2:.4f})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(x, residuals)
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Residuals")
        plt.xlabel("X")
        plt.ylabel("Residual")

        plt.tight_layout()
		if intenv:
			plt.show()
		else:
			plt.show(block=False)
			plt.pause(0.1)

    except Exception as e:
        print(f"[!] Error plotting model {name}: {e}")

# Plot menu

while True:
    print("\n=== Plot Models ===")
    model_options = {i + 1: name for i, (name, _) in enumerate(sorted_models)}

    for idx, name in model_options.items():
        print(f" [{idx}] {name} (R² = {results[name][2]:.5f})")
    print(" [0] Exit")

    try:
        choice = int(input("Select a model to plot: ").strip())
    except Exception:
        print("[!] Invalid input.")
        continue

    if choice == 0:
        break
    elif choice in model_options:
        name = model_options[choice]
        func, params, r2 = results[name]
        if params is None:
            print(f"[!] Cannot plot {name}: fitting failed.")
            continue
        plot_model(name, func, params, r2)
    else:
        print("[!] Invalid selection.")