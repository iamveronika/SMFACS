import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import datetime
import fractions


class DataLoader:
    @staticmethod
    def load_data(file_path='excel.xlsx', sheet_name='main'):
        return pd.read_excel(file_path, sheet_name=sheet_name)


class Plotter:
    def __init__(self, root):
        self.root = root

    def create_scrolled_window(self, parent, width=1400, height=600):
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, width=width, height=height)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill=tk.BOTH, expand=True)
        canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")

        return scrollable_frame

    def plot_data(self, data, selected_category, normalized=False, smoothed=False):
        categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone', 'naval ship',
                      'anti-aircraft warfare']
        time_series_data = {category: data[['date', category]].set_index('date') for category in categories}

        if normalized or smoothed:
            scaler = MinMaxScaler()
            time_series_data = {category: pd.DataFrame(scaler.fit_transform(time_series_data[category]),
                                                       index=time_series_data[category].index,
                                                       columns=[category]) for category in categories}

        if smoothed:
            window_size = 7  # Moving average window size
            time_series_data = {category: time_series_data[category].rolling(window=window_size, min_periods=1).mean()
                                for category in categories}

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Data Chart")

        scrollable_frame = self.create_scrolled_window(plot_window)

        if selected_category == "All categories":
            fig, axes = plt.subplots(len(categories), 1, figsize=(14, 40))
            for i, category in enumerate(categories):
                axes[i].plot(time_series_data[category], label=category, marker='o', linestyle='-')
                title_suffix = " (Normalized)" if normalized else ""
                title_suffix += " (Smoothed)" if smoothed else ""
                axes[i].set_title(f'{category.capitalize()}{title_suffix}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Count' if not normalized else 'Normalized Count')
                axes[i].grid(True)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(time_series_data[selected_category], label=selected_category, marker='o', linestyle='-')
            title_suffix = " (Normalized)" if normalized else ""
            title_suffix += " (Smoothed)" if smoothed else ""
            ax.set_title(f'{selected_category.capitalize()}{title_suffix}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Count' if not normalized else 'Normalized Count')
            ax.grid(True)
            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_trend_lines(self, data, selected_category):
        categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone', 'naval ship',
                      'anti-aircraft warfare']
        time_series_data = {category: data[['date', category]].set_index('date') for category in categories}

        scaler = MinMaxScaler()
        normalized_data = {category: pd.DataFrame(scaler.fit_transform(time_series_data[category]),
                                                  index=time_series_data[category].index,
                                                  columns=[category])
                           for category in categories}

        window_size = 7  # Moving average window size
        smoothed_data = {category: normalized_data[category].rolling(window=window_size, min_periods=1).mean()
                         for category in categories}

        models = {}
        for category in categories:
            model = LinearRegression()
            X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = smoothed_data[category].dropna().values
            model.fit(X, y)
            models[category] = model

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Trend Lines")

        scrollable_frame = self.create_scrolled_window(plot_window)

        if selected_category == "All categories":
            fig, axes = plt.subplots(len(categories), 1, figsize=(14, 40))
            for i, category in enumerate(categories):
                axes[i].plot(smoothed_data[category], label=f'{category} (Smoothed)', marker='o', linestyle='-')
                X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                axes[i].plot(smoothed_data[category].dropna().index, models[category].predict(X), color='red',
                             label='Trend Line')
                axes[i].set_title(f'{category.capitalize()} (Smoothed and Trend)')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Normalized Count')
                axes[i].legend()
                axes[i].grid(True)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(smoothed_data[selected_category], label=f'{selected_category} (Smoothed)', marker='o',
                    linestyle='-')
            X = smoothed_data[selected_category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            ax.plot(smoothed_data[selected_category].dropna().index, models[selected_category].predict(X), color='red',
                    label='Trend Line')
            ax.set_title(f'{selected_category.capitalize()} (Smoothed and Trend)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Count')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_derivatives(self, data, selected_category):
        categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone', 'naval ship',
                      'anti-aircraft warfare']
        time_series_data = {category: data[['date', category]].set_index('date') for category in categories}

        scaler = MinMaxScaler()
        normalized_data = {category: pd.DataFrame(scaler.fit_transform(time_series_data[category]),
                                                  index=time_series_data[category].index,
                                                  columns=[category])
                           for category in categories}

        window_size = 7  # Moving average window size
        smoothed_data = {category: normalized_data[category].rolling(window=window_size, min_periods=1).mean()
                         for category in categories}

        models = {}
        rmse_scores = {}
        for category in categories:
            model = LinearRegression()
            X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = smoothed_data[category].dropna().values
            model.fit(X, y)
            models[category] = model
            predictions = model.predict(X)
            rmse_scores[category] = np.sqrt(mean_squared_error(y, predictions))

        derivatives = {}
        second_derivatives = {}
        for category in categories:
            X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values
            y = smoothed_data[category].dropna().values.flatten()
            first_derivative = np.gradient(y, X)
            derivatives[category] = pd.DataFrame(first_derivative, index=smoothed_data[category].dropna().index,
                                                 columns=[category + '_first_derivative'])
            second_derivative = np.gradient(first_derivative, X)
            second_derivatives[category] = pd.DataFrame(second_derivative, index=smoothed_data[category].dropna().index,
                                                        columns=[category + '_second_derivative'])

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Derivatives")

        scrollable_frame = self.create_scrolled_window(plot_window)

        if selected_category == "All categories":
            fig, axes = plt.subplots(len(categories), 3, figsize=(14, 52))
            for i, category in enumerate(categories):
                axes[i, 0].plot(smoothed_data[category], label=f'{category} (Smoothed)', marker='o', linestyle='-')
                X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                axes[i, 0].plot(smoothed_data[category].dropna().index, models[category].predict(X), color='red',
                                label='Trend Line')
                axes[i, 0].set_title(
                    f'{category.capitalize()} (Smoothed and Trend)\nRMSE = {rmse_scores[category]:.2f}')
                axes[i, 0].set_xlabel('Date')
                axes[i, 0].set_ylabel('Normalized Count')
                axes[i, 0].legend()
                axes[i, 0].grid(True)
                axes[i, 0].xaxis.set_major_locator(plt.MaxNLocator(3))

                axes[i, 1].plot(derivatives[category], label=f'{category} First Derivative', color='blue', marker='x')
                axes[i, 1].set_title(f'{category.capitalize()} I Derivative (Trend Velocity)')
                axes[i, 1].set_xlabel('Date')
                axes[i, 1].set_ylabel('First Derivative')
                axes[i, 1].legend()
                axes[i, 1].grid(True)
                axes[i, 1].xaxis.set_major_locator(plt.MaxNLocator(3))

                axes[i, 2].plot(second_derivatives[category], label=f'{category} Second Derivative', color='green',
                                marker='s')
                axes[i, 2].set_title(f'{category.capitalize()} II Derivative (Trend Acceleration)')
                axes[i, 2].set_xlabel('Date')
                axes[i, 2].set_ylabel('Second Derivative')
                axes[i, 2].legend()
                axes[i, 2].grid(True)
                axes[i, 2].xaxis.set_major_locator(plt.MaxNLocator(3))

            fig.tight_layout()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 6))
            axes[0].plot(smoothed_data[selected_category], label=f'{selected_category} (Smoothed)', marker='o',
                         linestyle='-')
            X = smoothed_data[selected_category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            axes[0].plot(smoothed_data[selected_category].dropna().index, models[selected_category].predict(X),
                         color='red', label='Trend Line')
            axes[0].set_title(
                f'{selected_category.capitalize()} (Smoothed and Trend)\nRMSE = {rmse_scores[selected_category]:.2f}')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Normalized Count')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].xaxis.set_major_locator(plt.MaxNLocator(3))

            axes[1].plot(derivatives[selected_category], label=f'{selected_category} First Derivative', color='blue',
                         marker='x')
            axes[1].set_title(f'{selected_category.capitalize()} I Derivative (Trend Velocity)')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('First Derivative')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].xaxis.set_major_locator(plt.MaxNLocator(3))

            axes[2].plot(second_derivatives[selected_category], label=f'{selected_category} Second Derivative',
                         color='green', marker='s')
            axes[2].set_title(f'{selected_category.capitalize()} II Derivative (Trend Acceleration)')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Second Derivative')
            axes[2].legend()
            axes[2].grid(True)
            axes[2].xaxis.set_major_locator(plt.MaxNLocator(3))

            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_extrapolated_trends(self, data, selected_category):
        categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone', 'naval ship',
                      'anti-aircraft warfare']
        time_series_data = {category: data[['date', category]].set_index('date') for category in categories}

        scaler = MinMaxScaler()
        normalized_data = {category: pd.DataFrame(scaler.fit_transform(time_series_data[category]),
                                                  index=time_series_data[category].index,
                                                  columns=[category])
                           for category in categories}

        window_size = 7  # Moving average window size
        smoothed_data = {category: normalized_data[category].rolling(window=window_size, min_periods=1).mean()
                         for category in categories}

        models = {}
        for category in categories:
            model = LinearRegression()
            X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = smoothed_data[category].dropna().values
            model.fit(X, y)
            models[category] = model

        future_days = 100  # Number of days to predict
        last_date = max(data['date'])
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        future_dates_ordinal = np.array([pd.Timestamp(date).toordinal() for date in future_dates]).reshape(-1, 1)

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Forecasted Trends")

        scrollable_frame = self.create_scrolled_window(plot_window)

        if selected_category == "All categories":
            fig, axes = plt.subplots(len(categories), 1, figsize=(14, 40))
            for i, category in enumerate(categories):
                axes[i].plot(smoothed_data[category], label=f'{category} (Smoothed Data)', marker='o', linestyle='-')
                X = smoothed_data[category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                axes[i].plot(smoothed_data[category].dropna().index, models[category].predict(X), color='red',
                             label='Trend Line')

                future_predictions = models[category].predict(future_dates_ordinal)
                future_dates_index = pd.to_datetime(
                    [pd.Timestamp.fromordinal(int(date)) for date in future_dates_ordinal])
                axes[i].plot(future_dates_index, future_predictions, color='orange', linestyle='--', label='Forecast')

                axes[i].set_title(f'{category.capitalize()} (Smoothed Data, Trend, and Forecast)')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Normalized Count')
                axes[i].legend()
                axes[i].grid(True)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(smoothed_data[selected_category], label=f'{selected_category} (Smoothed Data)', marker='o',
                    linestyle='-')
            X = smoothed_data[selected_category].dropna().index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            ax.plot(smoothed_data[selected_category].dropna().index, models[selected_category].predict(X), color='red',
                    label='Trend Line')

            future_predictions = models[selected_category].predict(future_dates_ordinal)
            future_dates_index = pd.to_datetime([pd.Timestamp.fromordinal(int(date)) for date in future_dates_ordinal])
            ax.plot(future_dates_index, future_predictions, color='orange', linestyle='--', label='Forecast')

            ax.set_title(f'{selected_category.capitalize()} (Smoothed Data, Trend, and Forecast)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Count')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class Application:
    def __init__(self, root):
        self.root = root
        self.plotter = Plotter(root)
        self.setup_main_window()

    def setup_main_window(self):
        self.root.title("System for Multi-Factor Assessment of Critical Situations")
        self.root.geometry("500x300")
        self.root.configure(bg="#f0f4f8")

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10, background="#1976d2", foreground="black")
        style.map("TButton", background=[('active', '#1565c0')], foreground=[('active', 'black')])

        style.configure("TLabel", font=("Helvetica", 14), background="#f0f4f8", foreground="#0d47a1")

        tk.Label(self.root, text="\nSystem for Multi-Factor Assessment\n of Critical Situations",
                 font=("Helvetica", 18, "bold"), bg="#f0f4f8", fg="#0d47a1").pack(pady=20)
        ttk.Button(self.root, text="Start", command=self.open_second_window, style="TButton").pack(expand=True)

    def open_second_window(self):
        second_window = tk.Toplevel(self.root)
        second_window.title("System for Multi-Factor Assessment of Critical Situations")
        second_window.geometry("500x500")
        second_window.configure(bg="#e3f2fd")

        buttons_texts = [
            ("Segment data in time series format", lambda: self.open_category_selection(False, False)),
            ("Normalize Data", lambda: self.open_category_selection(True, False)),
            ("Smooth Data", lambda: self.open_category_selection(False, True)),
            ("Trend Lines", self.open_category_selection_trend_lines),
            ("Derivatives - Trend`s Velocity and Acceleration", self.open_category_selection_derivatives),
            ("Extrapolate Trends for Future Values", self.open_category_selection_extrapolated_trends),
            ("Calculate Weighted Integrated Indicator", self.open_weight_input_window)
        ]

        for text, command in buttons_texts:
            btn = ttk.Button(second_window, text=text, style="TButton", command=command)
            btn.pack(pady=10, padx=20, fill='x')

    def open_category_selection(self, normalized=False, smoothed=False):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Category Selection")
        selection_window.geometry("400x300")
        selection_window.configure(bg="#e3f2fd")

        tk.Label(selection_window, text="Select a category:", font=("Helvetica", 14), bg="#e3f2fd", fg="black").pack(
            pady=10)

        categories = ['All categories', 'aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone',
                      'naval ship', 'anti-aircraft warfare']
        selected_category = tk.StringVar(value="All categories")

        category_menu = ttk.Combobox(selection_window, textvariable=selected_category, values=categories,
                                     font=("Helvetica", 12))
        category_menu.pack(pady=10)

        ttk.Button(selection_window, text="Show Plot", style="TButton",
                   command=lambda: self.plotter.plot_data(DataLoader.load_data(), selected_category.get(), normalized,
                                                          smoothed)).pack(pady=20)

    def open_category_selection_trend_lines(self):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Category Selection")
        selection_window.geometry("400x300")
        selection_window.configure(bg="#e3f2fd")

        tk.Label(selection_window, text="Select a category:", font=("Helvetica", 14), bg="#e3f2fd", fg="black").pack(
            pady=10)

        categories = ['All categories', 'aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone',
                      'naval ship', 'anti-aircraft warfare']
        selected_category = tk.StringVar(value="All categories")

        category_menu = ttk.Combobox(selection_window, textvariable=selected_category, values=categories,
                                     font=("Helvetica", 12))
        category_menu.pack(pady=10)

        ttk.Button(selection_window, text="Show Plot", style="TButton",
                   command=lambda: self.plotter.plot_trend_lines(DataLoader.load_data(), selected_category.get())).pack(
            pady=20)

    def open_category_selection_derivatives(self):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Category Selection")
        selection_window.geometry("400x300")
        selection_window.configure(bg="#e3f2fd")

        tk.Label(selection_window, text="Select a category:", font=("Helvetica", 14), bg="#e3f2fd", fg="black").pack(
            pady=10)

        categories = ['All categories', 'aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone',
                      'naval ship', 'anti-aircraft warfare']
        selected_category = tk.StringVar(value="All categories")

        category_menu = ttk.Combobox(selection_window, textvariable=selected_category, values=categories,
                                     font=("Helvetica", 12))
        category_menu.pack(pady=10)

        ttk.Button(selection_window, text="Show Plot", style="TButton",
                   command=lambda: self.plotter.plot_derivatives(DataLoader.load_data(), selected_category.get())).pack(
            pady=20)

    def open_category_selection_extrapolated_trends(self):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Category Selection")
        selection_window.geometry("400x300")
        selection_window.configure(bg="#e3f2fd")

        tk.Label(selection_window, text="Select a category:", font=("Helvetica", 14), bg="#e3f2fd", fg="black").pack(
            pady=10)

        categories = ['All categories', 'aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone',
                      'naval ship', 'anti-aircraft warfare']
        selected_category = tk.StringVar(value="All categories")

        category_menu = ttk.Combobox(selection_window, textvariable=selected_category, values=categories,
                                     font=("Helvetica", 12))
        category_menu.pack(pady=10)

        ttk.Button(selection_window, text="Show Plot", style="TButton",
                   command=lambda: self.plotter.plot_extrapolated_trends(DataLoader.load_data(),
                                                                         selected_category.get())).pack(pady=20)

    def open_weight_input_window(self):
        weight_window = tk.Toplevel(self.root)
        weight_window.title("Enter Weights")
        weight_window.geometry("400x600")
        weight_window.configure(bg="#e3f2fd")

        tk.Label(weight_window, text="Enter weights for each category:", font=("Helvetica", 14), bg="#e3f2fd",
                 fg="black").grid(row=0, column=0, columnspan=2, pady=10)

        categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL', 'drone', 'naval ship',
                      'anti-aircraft warfare']
        entries = {}

        for i, category in enumerate(categories):
            tk.Label(weight_window, text=f"{category.capitalize()}: ", font=("Helvetica", 12), bg="#e3f2fd",
                     fg="black").grid(row=i + 1, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(weight_window, font=("Helvetica", 12), width=20)
            entry.grid(row=i + 1, column=1, padx=10, pady=5, sticky="w")
            entries[category] = entry

        ttk.Button(weight_window, text="Calculate Indicator", style="TButton",
                   command=lambda: self.calculate_integrated_score(entries)).grid(row=len(categories) + 1, column=0,
                                                                                  columnspan=2, pady=20)

    def validate_weight(self, value):
        try:
            float_value = float(value.replace(',', '.'))
            return True
        except ValueError:
            try:
                fractions.Fraction(value)
                return True
            except ValueError:
                return False

    def calculate_integrated_score(self, entries):
        weights = {}
        for category, entry in entries.items():
            value = entry.get()
            if self.validate_weight(value):
                try:
                    weights[category] = float(value.replace(',', '.'))
                except ValueError:
                    weights[category] = float(fractions.Fraction(value))
            else:
                messagebox.showerror("Invalid Input",
                                     f"Invalid weight value for {category}: {value}. Please enter a valid number.")
                return

        data = DataLoader.load_data()
        categories = list(weights.keys())

        scaler = MinMaxScaler()
        data[categories] = scaler.fit_transform(data[categories])

        weights_array = np.array([weights[category] for category in categories])

        window_size = 7
        smoothed_data = {category: data[[category]].rolling(window=window_size, min_periods=1).mean() for category in
                         categories}

        integrated_score = np.dot(data[categories], weights_array)
        data['Integrated Score'] = integrated_score

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Integrated Score")
        plot_window.geometry("800x600")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data['date'], data['Integrated Score'], marker='o', linestyle='-', label='Integrated Score')
        ax.set_title('Integrated Score Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
