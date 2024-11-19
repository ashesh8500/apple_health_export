import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess


class HealthDataProcessor:
    def __init__(self):
        self.numeric_df = None
        self.categorical_df = None
        self.data_types = None

    def load_xml_data(self, export_root):
        """Convert XML data to DataFrame and process it"""
        data = []
        for record in export_root.findall("Record"):
            record_type = record.get("type")
            start_date_str = record.get("startDate")
            value = record.get("value")

            if value is not None:
                data.append(
                    {
                        "RecordType": record_type,
                        "StartDate": pd.to_datetime(start_date_str, format="ISO8601"),
                        "Value": value,
                    }
                )

        raw_df = pd.DataFrame(data)
        self._process_dataframe(raw_df)

    def load_csv_data(self, file):
        """Load CSV data and process it"""
        raw_df = pd.read_csv(file)
        raw_df["StartDate"] = pd.to_datetime(raw_df["StartDate"], format="mixed")
        self._process_dataframe(raw_df)

    def _process_dataframe(self, df):
        """Process the raw dataframe into numeric and categorical dataframes"""
        # Add day of week
        df["DayOfWeek"] = df["StartDate"].dt.day_name()

        # Identify glucose data
        glucose_records = df[df["RecordType"].str.contains("Glucose", case=False)]

        # Determine data types for each RecordType
        self.data_types = {}
        for record_type in df["RecordType"].unique():
            values = df[df["RecordType"] == record_type]["Value"]
            try:
                pd.to_numeric(values, errors="raise")
                self.data_types[record_type] = "numeric"
            except:
                self.data_types[record_type] = "categorical"

        # Create separate numeric and categorical dataframes
        numeric_data = {}
        categorical_data = {}

        for record_type, dtype in self.data_types.items():
            temp_df = df[df["RecordType"] == record_type].copy()

            if dtype == "numeric":
                # Convert to numeric and handle duplicates
                temp_df["Value"] = pd.to_numeric(temp_df["Value"], errors="coerce")
                if "Glucose" in record_type:
                    # Keep all glucose readings without aggregation
                    temp_df.set_index("StartDate", inplace=True)
                    numeric_data[record_type] = temp_df["Value"]
                else:
                    numeric_data[record_type] = temp_df.groupby("StartDate")[
                        "Value"
                    ].mean()
            else:
                # For categorical data, take the most common value for duplicates
                categorical_data[record_type] = temp_df.groupby("StartDate")[
                    "Value"
                ].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)

        # Combine all numeric and categorical records
        if numeric_data:
            self.numeric_df = pd.DataFrame(numeric_data)
        else:
            self.numeric_df = pd.DataFrame()

        if categorical_data:
            self.categorical_df = pd.DataFrame(categorical_data)
        else:
            self.categorical_df = pd.DataFrame()

        # Add day of week to both dataframes
        if not self.numeric_df.empty:
            self.numeric_df["DayOfWeek"] = pd.DatetimeIndex(
                self.numeric_df.index
            ).day_name()
        if not self.categorical_df.empty:
            self.categorical_df["DayOfWeek"] = pd.DatetimeIndex(
                self.categorical_df.index
            ).day_name()

    def create_numeric_time_series_plot(self, record_type):
        """Create time series plot for numeric data"""
        fig = px.scatter(
            self.numeric_df,
            x=self.numeric_df.index,
            y=record_type,
            trendline="lowess",
            title=f"Time Series for {record_type}",
        )

        fig.update_layout(xaxis_title="Date", yaxis_title="Value", showlegend=True)

        return fig

    def create_categorical_frequency_plot(self, record_type):
        """Create frequency plot for categorical data"""
        value_counts = self.categorical_df[record_type].value_counts()

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Frequency Distribution for {record_type}",
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Frequency",
            xaxis={"categoryorder": "total descending"},
        )

        return fig

    def create_correlation_matrix(self):
        """Create correlation matrix for numeric data"""
        if self.numeric_df.empty:
            return None

        corr_matrix = self.numeric_df.drop(columns=["DayOfWeek"]).corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
            )
        )

        fig.update_layout(
            title="Correlation Matrix of Numeric Health Metrics", xaxis_tickangle=-45
        )

        return fig

    def calculate_agp(self, glucose_record_type):
        """Calculate AGP metrics for glucose data with 5th and 95th percentiles"""
        # Extract glucose data
        glucose_df = self.numeric_df[[glucose_record_type]].dropna().copy()
        glucose_df["Time"] = glucose_df.index.time

        # Convert time to minutes since midnight for grouping
        glucose_df["TimeMinutes"] = glucose_df.index.hour * 60 + glucose_df.index.minute

        # Group data by time and calculate percentiles
        agp_df = (
            glucose_df.groupby("TimeMinutes")[glucose_record_type]
            .agg(
                [
                    ("5th Percentile", lambda x: np.percentile(x, 5)),
                    ("10th Percentile", lambda x: np.percentile(x, 10)),
                    ("25th Percentile", lambda x: np.percentile(x, 25)),
                    ("Median", "median"),
                    ("75th Percentile", lambda x: np.percentile(x, 75)),
                    ("90th Percentile", lambda x: np.percentile(x, 90)),
                    ("95th Percentile", lambda x: np.percentile(x, 95)),
                ]
            )
            .reset_index()
        )

        # Convert TimeMinutes back to time
        agp_df["Time"] = pd.to_datetime(agp_df["TimeMinutes"], unit="m").dt.time

        return agp_df

    def create_daily_glucose_plot_for_dates(
        self, glucose_record_type, dates, target_range=(70, 180)
    ):
        """Create glucose plots for multiple dates with caching"""
        plots = []
        for date in dates:
            # Filter glucose data for the selected date
            glucose_df = self.numeric_df[[glucose_record_type]].dropna().copy()
            daily_data = glucose_df[glucose_df.index.date == date]

            if not daily_data.empty:
                # Create figure for this date
                fig = go.Figure()

                # Original glucose readings
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index,
                        y=daily_data[glucose_record_type],
                        mode="markers",
                        marker=dict(color="blue", size=4, opacity=0.6),
                        name="Glucose Readings",
                    )
                )

                # Add target range shading
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=daily_data.index.min(),
                    y0=target_range[0],
                    x1=daily_data.index.max(),
                    y1=target_range[1],
                    fillcolor="green",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )

                fig.update_layout(
                    title=f"Glucose Values on {date}",
                    xaxis_title="Time",
                    yaxis_title="Glucose Level (mg/dL)",
                    xaxis=dict(
                        tickformat="%H:%M",
                        dtick=3600000 * 2,  # Tick every 2 hours
                    ),
                    height=300,  # Smaller height for compact display
                    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
                )

                plots.append(fig)

        return plots

    def create_daily_glucose_plot_for_date(
        self, glucose_record_type, selected_date, target_range=(70, 180)
    ):
        """Create daily glucose values plot for the selected date with enhanced features"""
        # Filter glucose data for the selected date
        glucose_df = self.numeric_df[[glucose_record_type]].dropna().copy()
        daily_data = glucose_df[glucose_df.index.date == selected_date]

        if daily_data.empty:
            st.write("No glucose data available for the selected date.")
            return None

        # LOESS smoothing
        daily_data["Time"] = daily_data.index.time
        daily_data["Timestamp"] = daily_data.index
        daily_data.sort_index(inplace=True)

        # Use LOWESS from statsmodels
        import numpy as np

        # Convert time to numeric value (in seconds since midnight)
        time_numeric = daily_data["Timestamp"].map(datetime.timestamp)
        glucose_values = daily_data[glucose_record_type].values
        smoothed = lowess(glucose_values, time_numeric, frac=0.05)

        # Prepare figure
        fig = go.Figure()

        # Original glucose readings
        fig.add_trace(
            go.Scatter(
                x=daily_data["Timestamp"],
                y=daily_data[glucose_record_type],
                mode="markers",
                marker=dict(color="blue", size=4, opacity=0.6),
                name="Glucose Readings",
            )
        )

        # LOESS smoothed line
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(smoothed[:, 0], unit="s"),
                y=smoothed[:, 1],
                mode="lines",
                line=dict(color="red", width=2),
                name="LOESS Smoothed",
            )
        )

        # Target range shading
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=daily_data["Timestamp"].min(),
            y0=target_range[0],
            x1=daily_data["Timestamp"].max(),
            y1=target_range[1],
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            name="Target Range",
        )

        # Fill above target range
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(smoothed[:, 0], unit="s"),
                y=np.maximum(smoothed[:, 1], target_range[1]),
                fill="tonexty",
                fillcolor="rgba(231, 107, 243, 0.2)",
                mode="none",
                showlegend=True,
                name="Above Target",
                hoverinfo="skip",
            )
        )

        # Fill below target range
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(smoothed[:, 0], unit="s"),
                y=np.minimum(smoothed[:, 1], target_range[0]),
                fill="tonexty",
                fillcolor="rgba(255, 165, 0, 0.2)",
                mode="none",
                showlegend=True,
                name="Below Target",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=f"Daily Glucose Values on {selected_date}",
            xaxis_title="Time",
            yaxis_title="Glucose Level (mg/dL)",
            xaxis=dict(
                tickformat="%H:%M",
                dtick=3600000 * 2,  # Tick every 2 hours
            ),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def create_agp_plot(self, agp_df, target_range=(70, 140)):
        """Create enhanced AGP plot from AGP metrics dataframe"""
        # Convert 'Time' column to datetime for proper x-axis formatting
        agp_df["Time"] = pd.to_datetime(agp_df["Time"].astype(str))

        fig = go.Figure()

        # Add shaded areas between percentiles
        fig.add_trace(
            go.Scatter(
                x=agp_df["Time"],
                y=agp_df["90th Percentile"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="90th Percentile",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agp_df["Time"],
                y=agp_df["75th Percentile"],
                fill="tonexty",
                fillcolor="rgba(0, 176, 246, 0.2)",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="75th Percentile",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agp_df["Time"],
                y=agp_df["Median"],
                mode="lines",
                line=dict(color="rgb(0,100,80)", width=2),
                name="Median",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agp_df["Time"],
                y=agp_df["25th Percentile"],
                fill="tonexty",
                fillcolor="rgba(0, 176, 246, 0.2)",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="25th Percentile",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agp_df["Time"],
                y=agp_df["10th Percentile"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="10th Percentile",
            )
        )

        # Add target range shading
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=agp_df["Time"].min(),
            y0=target_range[0],
            x1=agp_df["Time"].max(),
            y1=target_range[1],
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            name="Target Range",
        )

        # Add target range lines
        fig.add_hline(y=target_range[0], line_dash="dash", line_color="red")
        fig.add_hline(y=target_range[1], line_dash="dash", line_color="red")

        # Update layout
        fig.update_layout(
            title="Ambulatory Glucose Profile (AGP)",
            xaxis_title="Time of Day",
            yaxis_title="Glucose Level (mg/dL)",
            xaxis=dict(
                tickformat="%H:%M",
                dtick=3600000 * 2,  # Tick every 2 hours
            ),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Add color only where median crosses target range boundaries
        intersect_points_high = np.where(agp_df["Median"] > target_range[1])[0]
        intersect_points_low = np.where(agp_df["Median"] < target_range[0])[0]

        if len(intersect_points_high) > 0:
            fig.add_trace(
                go.Scatter(
                    x=agp_df.iloc[intersect_points_high]["Time"],
                    y=agp_df.iloc[intersect_points_high]["Median"],
                    fill="tonexty",
                    fillcolor="rgba(231, 107, 243, 0.2)",
                    mode="none",
                    showlegend=True,
                    name="Above Target",
                    hoverinfo="skip",
                )
            )

        if len(intersect_points_low) > 0:
            fig.add_trace(
                go.Scatter(
                    x=agp_df.iloc[intersect_points_low]["Time"],
                    y=agp_df.iloc[intersect_points_low]["Median"],
                    fill="tonexty",
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    mode="none",
                    showlegend=True,
                    name="Below Target",
                    hoverinfo="skip",
                )
            )

        return fig

    def create_daily_glucose_plot(self, glucose_record_type):
        """Create daily glucose values plot"""
        glucose_df = self.numeric_df[[glucose_record_type]].dropna().copy()

        fig = px.line(
            glucose_df,
            x=glucose_df.index,
            y=glucose_record_type,
            title="Daily Glucose Values Over Time",
            labels={"x": "Date", glucose_record_type: "Glucose Level (mg/dL)"},
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Glucose Level (mg/dL)",
            hovermode="x unified",
        )

        return fig

    def calculate_time_in_range(
        self, glucose_record_type, lower_limit=70, upper_limit=180
    ):
        """Calculate time in range statistics for glucose data"""
        glucose_df = self.numeric_df[[glucose_record_type]].dropna().copy()
        total_readings = len(glucose_df)
        in_range_readings = glucose_df[
            (glucose_df[glucose_record_type] >= lower_limit)
            & (glucose_df[glucose_record_type] <= upper_limit)
        ]
        below_range_readings = glucose_df[glucose_df[glucose_record_type] < lower_limit]
        above_range_readings = glucose_df[glucose_df[glucose_record_type] > upper_limit]

        tir = (
            (len(in_range_readings) / total_readings) * 100 if total_readings > 0 else 0
        )
        below_range = (
            (len(below_range_readings) / total_readings) * 100
            if total_readings > 0
            else 0
        )
        above_range = (
            (len(above_range_readings) / total_readings) * 100
            if total_readings > 0
            else 0
        )

        return tir, below_range, above_range

    def create_time_in_range_pie(self, tir, below_range, above_range):
        """Create a pie chart for time in range statistics"""
        labels = ["In Range", "Below Range", "Above Range"]
        values = [tir, below_range, above_range]
        colors = ["#2ca02c", "#d62728", "#1f77b4"]

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))]
        )
        fig.update_layout(title="Time in Range", legend_title="Ranges")

        return fig


def main():
    st.title("Apple Health Data Analyzer")

    # Initialize processor
    processor = HealthDataProcessor()

    # File upload section
    st.sidebar.header("Data Upload")
    file_type = st.sidebar.radio("Choose file type", ["XML", "CSV"])

    if file_type == "XML":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Apple Health Export (XML)", type="xml"
        )
        if uploaded_file is not None:
            export = ET.parse(uploaded_file)
            export_root = export.getroot()
            processor.load_xml_data(export_root)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Preprocessed Data (CSV)", type="csv"
        )
        if uploaded_file is not None:
            processor.load_csv_data(uploaded_file)

    if processor.numeric_df is not None or processor.categorical_df is not None:
        # Display date ranges
        if not processor.numeric_df.empty:
            st.write(
                "Numeric Data Range:",
                f"{processor.numeric_df.index.min().date()} to {processor.numeric_df.index.max().date()}",
            )
        if not processor.categorical_df.empty:
            st.write(
                "Categorical Data Range:",
                f"{processor.categorical_df.index.min().date()} to {processor.categorical_df.index.max().date()}",
            )

        # Analysis Options
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.radio(
            "Choose Analysis Type",
            [
                "Numeric Time Series",
                "Categorical Analysis",
                "Correlation Analysis",
                "Glucose Analysis",
            ],
        )

        if analysis_type == "Numeric Time Series":
            if not processor.numeric_df.empty:
                numeric_records = [
                    col for col in processor.numeric_df.columns if col != "DayOfWeek"
                ]
                selected_record = st.sidebar.selectbox(
                    "Select Numeric Metric", numeric_records
                )

                fig = processor.create_numeric_time_series_plot(selected_record)
                st.plotly_chart(fig, use_container_width=True)

                # Display basic statistics
                st.subheader("Basic Statistics")
                stats = processor.numeric_df[selected_record].describe()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                    st.metric("Minimum", f"{stats['min']:.2f}")
                    st.metric("25th Percentile", f"{stats['25%']:.2f}")
                with col2:
                    st.metric("Maximum", f"{stats['max']:.2f}")
                    st.metric("75th Percentile", f"{stats['75%']:.2f}")
                    st.metric("Standard Deviation", f"{stats['std']:.2f}")
            else:
                st.write("No numeric data available")

        elif analysis_type == "Categorical Analysis":
            if not processor.categorical_df.empty:
                categorical_records = [
                    col
                    for col in processor.categorical_df.columns
                    if col != "DayOfWeek"
                ]
                selected_record = st.sidebar.selectbox(
                    "Select Categorical Metric", categorical_records
                )

                fig = processor.create_categorical_frequency_plot(selected_record)
                st.plotly_chart(fig, use_container_width=True)

                # Day of week analysis
                st.subheader("Distribution by Day of Week")
                dow_counts = (
                    processor.categorical_df.groupby("DayOfWeek")[selected_record]
                    .value_counts()
                    .unstack()
                )
                st.bar_chart(dow_counts)
            else:
                st.write("No categorical data available")

        elif analysis_type == "Glucose Analysis":
            glucose_records = [
                col for col in processor.numeric_df.columns if "Glucose" in col
            ]
            if glucose_records:
                selected_record = st.sidebar.selectbox(
                    "Select Glucose Metric", glucose_records
                )

                # Get all unique dates
                all_dates = sorted(np.unique(processor.numeric_df.index.date))

                # Create pagination controls
                plots_per_page = 5
                total_pages = len(all_dates) // plots_per_page + (
                    1 if len(all_dates) % plots_per_page > 0 else 0
                )

                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("← Previous"):
                        if "current_page" in st.session_state:
                            st.session_state.current_page = max(
                                0, st.session_state.current_page - 1
                            )
                        else:
                            st.session_state.current_page = 0

                with col2:
                    if "current_page" not in st.session_state:
                        st.session_state.current_page = 0
                    st.write(
                        f"Page {st.session_state.current_page + 1} of {total_pages}"
                    )

                with col3:
                    if st.button("Next →"):
                        if "current_page" in st.session_state:
                            st.session_state.current_page = min(
                                total_pages - 1, st.session_state.current_page + 1
                            )
                        else:
                            st.session_state.current_page = 0

                # Get dates for current page
                start_idx = st.session_state.current_page * plots_per_page
                end_idx = start_idx + plots_per_page
                current_dates = all_dates[start_idx:end_idx]

                # Generate plots for current page
                daily_plots = processor.create_daily_glucose_plot_for_dates(
                    selected_record, current_dates, target_range=(70, 180)
                )

                # Display plots
                for plot in daily_plots:
                    st.plotly_chart(plot, use_container_width=True)

                # Calculate AGP metrics
                agp_df = processor.calculate_agp(selected_record)
                agp_fig = processor.create_agp_plot(agp_df)
                st.plotly_chart(agp_fig, use_container_width=True)

                # Calculate and display time in range
                lower_limit = st.sidebar.number_input("Lower Limit (mg/dL)", value=70)
                upper_limit = st.sidebar.number_input("Upper Limit (mg/dL)", value=180)

                tir, below_range, above_range = processor.calculate_time_in_range(
                    selected_record, lower_limit, upper_limit
                )
                tir_fig = processor.create_time_in_range_pie(
                    tir, below_range, above_range
                )
                st.plotly_chart(tir_fig, use_container_width=True)

                # Display statistics
                st.subheader("Time in Range Statistics")
                st.write(
                    f"**Time in Range ({lower_limit}-{upper_limit} mg/dL):** {tir:.2f}%"
                )
                st.write(f"**Below Range (<{lower_limit} mg/dL):** {below_range:.2f}%")
                st.write(f"**Above Range (>{upper_limit} mg/dL):** {above_range:.2f}%")
            else:
                st.write("No glucose data available for analysis")
        else:  # Correlation Analysis
            if not processor.numeric_df.empty:
                corr_fig = processor.create_correlation_matrix()
                st.plotly_chart(corr_fig, use_container_width=True)

                # Display strongest correlations
                st.subheader("Strongest Correlations")
                corr_matrix = processor.numeric_df.drop(columns=["DayOfWeek"]).corr()
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        correlations.append(
                            {
                                "Metric 1": corr_matrix.columns[i],
                                "Metric 2": corr_matrix.columns[j],
                                "Correlation": corr_matrix.iloc[i, j],
                            }
                        )

                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(corr_df)
            else:
                st.write("No numeric data available for correlation analysis")

        # Export functionality
        st.sidebar.header("Export Options")
        if st.sidebar.button("Export Processed Data"):
            numeric_csv = (
                processor.numeric_df.to_csv() if not processor.numeric_df.empty else ""
            )
            categorical_csv = (
                processor.categorical_df.to_csv()
                if not processor.categorical_df.empty
                else ""
            )

            if numeric_csv:
                st.sidebar.download_button(
                    "Download Numeric Data CSV",
                    numeric_csv,
                    file_name="numeric_health_data.csv",
                    mime="text/csv",
                )
            if categorical_csv:
                st.sidebar.download_button(
                    "Download Categorical Data CSV",
                    categorical_csv,
                    file_name="categorical_health_data.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
