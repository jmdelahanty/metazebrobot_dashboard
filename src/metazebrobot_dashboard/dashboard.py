import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
import os
import traceback
import re
# Use pandas for styling compatibility
import pandas as pd

# ---- Import for statistical tests ----
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # This warning is now placed inside the app logic where tests are selected


# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Fish Survival Analysis")

st.title("Fish Survivability Analysis Dashboard")

# --- Helper Functions ---

def hex_to_rgba(hex_color, alpha=0.3):
    """Converts a hex color string to an rgba string."""
    try:
        if hex_color.startswith('#'):
            rgb = px.colors.hex_to_rgb(hex_color)
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
        elif hex_color.startswith('rgb'):
            parts = re.findall(r'\d+', hex_color)
            if len(parts) >= 3:
                return f'rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})'
    except Exception:
        pass # Fallback below
    print(f"Warning: Could not parse color '{hex_color}' for RGBA conversion. Using default.")
    return 'rgba(128, 128, 128, 0.3)' # Default grey semi-transparent

# --- Plotting and Analysis Functions ---

# --- plot_survival_curves_pl (Handles Genotype/Housing, Density, and Cross ID grouping) ---
def plot_survival_curves_pl(
    df: pl.DataFrame,
    color_by: str, # 'genotype_housing', 'density_category', or 'cross_id'
    visible_conditions_list: Optional[List[str]] = None
) -> go.Figure:
    """
    Plot survival curves over time using Polars DataFrame and Plotly.
    Includes mean lines with shaded standard deviation bounds.
    Handles grouping by different columns. Assumes color_by column is Utf8 type.
    """
    print(f"Plotting survival curves (color_by={color_by})...")
    plot_df = df.clone()

    color_col = color_by
    title_map = {
        'genotype_housing': 'Survival Curves by Genotype and Housing Type (Mean +/- Std Dev)',
        'density_category': 'Survival Curves by Fish Density (Mean +/- Std Dev)',
        'cross_id': 'Survival Curves by Cross ID (Mean +/- Std Dev)'
    }
    legend_title_map = {
        'genotype_housing': 'Genotype & Housing',
        'density_category': 'Fish Density',
        'cross_id': 'Cross ID'
    }

    title = title_map.get(color_col, f'Survival Curves by {color_col} (Mean +/- Std Dev)')
    legend_title = legend_title_map.get(color_col, color_col)
    COLOR_SEQUENCE = px.colors.qualitative.Vivid

    # Ensure color column exists and is Utf8 (as standardized in load_processed_data)
    if color_col not in plot_df.columns:
        st.error(f"Coloring column '{color_col}' not found in data for survival plot.")
        return go.Figure(layout=go.Layout(title=title + " - Error: Column Missing", height=600, yaxis=dict(range=[0, 100.5])))
    if plot_df[color_col].dtype != pl.Utf8:
         # Attempt to cast safely if somehow it's not Utf8
         try:
            plot_df = plot_df.with_columns(pl.col(color_col).cast(pl.Utf8))
            st.warning(f"Plotting function had to cast '{color_col}' back to Utf8. Check loading logic if this persists.")
         except Exception as e:
             st.error(f"Failed to cast column '{color_col}' to Utf8 for plotting: {e}")
             return go.Figure(layout=go.Layout(title=title + f" - Error: Cannot Cast {color_col}", height=600, yaxis=dict(range=[0, 100.5])))


    # --- Filter data based on visible_conditions_list BEFORE plotting ---
    if visible_conditions_list is not None:
        if not visible_conditions_list:
             st.warning(f"No groups selected for '{color_col}' to display.")
             return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Survival Rate (%)', yaxis=dict(range=[0, 100])))
        else:
            print(f"Filtering plot data for {color_col}: {visible_conditions_list}")
            # Filter list should already be strings, compare directly with Utf8 column
            filter_list_str = [str(item) for item in visible_conditions_list]
            plot_df = plot_df.filter(pl.col(color_col).is_in(filter_list_str))
            if plot_df.height == 0:
                 st.warning(f"No data matches the selected filters for '{color_col}'.")
                 return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Survival Rate (%)', yaxis=dict(range=[0, 100])))

    # --- Get unique groups and assign colors deterministically ---
    # Column is already Utf8
    unique_groups_agg = plot_df.group_by(color_col).agg(pl.len()).sort(color_col)
    unique_cats = unique_groups_agg[color_col].drop_nulls().to_list()
    color_map = {cat: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, cat in enumerate(unique_cats)}

    # --- Create the base figure ---
    fig = go.Figure()

    # Add individual survival lines first (lightly)
    try:
        # Group by dish_id and the Utf8 color_col
        unique_dishes_data = plot_df.group_by(['dish_id', color_col]).agg(pl.first('days_since_fertilization')).drop_nulls()

        for dish_group in unique_dishes_data.iter_rows(named=True):
            dish_id = dish_group['dish_id']
            group_name = dish_group[color_col] # Already Utf8
            dish_data = plot_df.filter(pl.col('dish_id') == dish_id).sort('days_since_fertilization')
            group_color = color_map.get(group_name) # Use Utf8 group_name for lookup

            if group_color:
                fig.add_trace(go.Scatter(
                    x=dish_data['days_since_fertilization'], y=dish_data['survival_rate'], mode='lines',
                    line=dict(width=0.75, color=group_color), opacity=0.3, name=f"Dish_{dish_id}", # Unique name
                    legendgroup=str(group_name), showlegend=False,
                    hovertext=[f"Dish: {dish_id}<br>Day: {d}<br>Survival: {s:.1f}%<br>Group: {group_name}"
                               for d, s in zip(dish_data['days_since_fertilization'], dish_data['survival_rate'])],
                    hoverinfo="text"
                ))
    except Exception as e:
        print(f"Could not add individual lines: {e}")
        traceback.print_exc()

    # --- Aggregate for Mean and Std Dev ---
    if plot_df.height > 0:
        plot_df_agg = plot_df.with_columns([
            pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
            pl.col('survival_rate').cast(pl.Float64, strict=False)
        ]).drop_nulls(subset=['days_since_fertilization', 'survival_rate', color_col])

        if plot_df_agg.height > 0:
            # Group by Utf8 color_col
            agg_df = plot_df_agg.group_by([color_col, 'days_since_fertilization']).agg([
                pl.mean('survival_rate').alias('mean_survival_rate'),
                pl.std('survival_rate').alias('std_survival_rate')
            ]).sort(color_col, 'days_since_fertilization')

            agg_df = agg_df.with_columns([
                (pl.col('mean_survival_rate') + pl.col('std_survival_rate')).clip(lower_bound=0, upper_bound=100).alias('upper_bound'),
                (pl.col('mean_survival_rate') - pl.col('std_survival_rate')).clip(lower_bound=0, upper_bound=100).alias('lower_bound')
            ])

            for cat in unique_cats: # unique_cats are Utf8 strings
                # Filter using Utf8 string
                cat_data = agg_df.filter(pl.col(color_col) == cat)
                if cat_data.height == 0: continue

                color = color_map.get(cat)
                if not color: continue # Skip if color wasn't assigned
                fill_color = hex_to_rgba(color, 0.2) # Use helper for fill

                x_vals = cat_data['days_since_fertilization']
                y_upper = cat_data['upper_bound']
                y_lower = cat_data['lower_bound']
                y_mean = cat_data['mean_survival_rate']
                y_std = cat_data['std_survival_rate']

                fig.add_trace(go.Scatter(x=x_vals, y=y_upper, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False, name=f'{cat}_upper'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_lower, mode='lines', line=dict(width=0), fillcolor=fill_color, fill='tonexty', hoverinfo='skip', showlegend=False, name=f'{cat}_lower'))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_mean, mode='lines', line=dict(width=3, dash='dash', color=color), name=f'{cat} (mean)', legendgroup=str(cat),
                    hovertext=[f"Day: {d}<br>Avg Survival: {m:.1f}%<br>Std Dev: {'N/A' if s is None else f'{s:.1f}'}<br>Group: {cat}" for d, m, s in zip(x_vals, y_mean, y_std)],
                    hoverinfo="text"
                ))
        else: print("No non-null data available for mean/std dev calculation.")

    fig.update_layout(
        title=title, xaxis_title='Days Since Fertilization', yaxis_title='Survival Rate (%)', yaxis=dict(range=[0, 100.5]),
        legend_title_text=legend_title, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified', height=600
    )
    return fig

# --- calculate_approx_median_survival --- MODIFIED ---
@st.cache_data
def calculate_approx_median_survival(df: pl.DataFrame, group_col: str, visible_groups: Optional[List[str]] = None) -> Optional[pl.DataFrame]:
    """
    Calculates the approximate day when the mean survival rate first drops below 50%
    for each group specified by group_col. Ensures output column is Utf8.
    """
    print(f"Calculating approximate median survival for groups in '{group_col}'...")
    if group_col not in df.columns or 'days_since_fertilization' not in df.columns or 'survival_rate' not in df.columns:
        st.warning(f"Missing required columns ({group_col}, days_since_fertilization, survival_rate) for median survival calculation.")
        return None

    calc_df = df.clone()

    # Ensure group_col is Utf8 for consistent processing
    if calc_df[group_col].dtype != pl.Utf8:
        calc_df = calc_df.with_columns(pl.col(group_col).cast(pl.Utf8))

    # Filter for visible groups if provided (compare as strings)
    if visible_groups is not None:
        if not visible_groups:
            st.info("No groups selected to calculate median survival.")
            return None
        visible_groups_str = [str(g) for g in visible_groups]
        calc_df = calc_df.filter(pl.col(group_col).is_in(visible_groups_str))
        if calc_df.height == 0:
            st.info("No data matches the selected filters for median survival calculation.")
            return None

    # Aggregate mean survival per group per day (group by Utf8)
    mean_survival = calc_df.group_by([group_col, 'days_since_fertilization']).agg(
        pl.mean('survival_rate').alias('mean_survival')
    ).sort(group_col, 'days_since_fertilization')

    # Find the first day where mean survival is <= 50%
    median_survival_days = mean_survival.filter(
        pl.col('mean_survival') <= 50
    ).group_by(group_col).agg(
        pl.min('days_since_fertilization').alias('Median Survival Day (Approx)')
    )

    # Check which groups never reached 50%
    groups_never_below_50 = mean_survival.group_by(group_col).agg(
        pl.min('mean_survival').alias('min_mean_survival')
    ).filter(pl.col('min_mean_survival') > 50).select(group_col)

    # Create a DataFrame for groups that never went below 50
    never_below_df = groups_never_below_50.with_columns(
        pl.lit("> Max Day").alias('Median Survival Day (Approx)') # This column is Utf8
    )

    # *** FIX: Cast the numeric median day to Utf8 before concatenating ***
    if median_survival_days.height > 0:
        median_survival_days = median_survival_days.with_columns(
            pl.col('Median Survival Day (Approx)').cast(pl.Utf8)
        )

    # Combine the results (now both have Utf8 median day column)
    if median_survival_days.height > 0 and never_below_df.height > 0:
        final_median_df = pl.concat([
            median_survival_days,
            never_below_df
        ], how='vertical').sort(group_col)
    elif median_survival_days.height > 0:
        final_median_df = median_survival_days.sort(group_col)
    elif never_below_df.height > 0:
        final_median_df = never_below_df.sort(group_col)
    else:
        final_median_df = pl.DataFrame({group_col: [], 'Median Survival Day (Approx)': []}, schema={group_col: pl.Utf8, 'Median Survival Day (Approx)': pl.Utf8})


    if final_median_df.height == 0:
        st.info("Could not determine approximate median survival for any selected groups.")
        # Return empty df with correct schema
        return pl.DataFrame({group_col: [], 'N (Dishes)': [], 'Median Survival Day (Approx)': []}, schema={group_col: pl.Utf8, 'N (Dishes)': pl.Int64, 'Median Survival Day (Approx)': pl.Utf8})


    # Get the total N (dishes) for each group
    n_dishes = calc_df.group_by(group_col).agg(pl.n_unique('dish_id').alias('N (Dishes)'))

    final_median_df = final_median_df.join(n_dishes, on=group_col, how='left')

    # Ensure N (Dishes) is Int64, handle potential nulls from join
    final_median_df = final_median_df.with_columns(pl.col('N (Dishes)').fill_null(0).cast(pl.Int64))

    return final_median_df.select([group_col, 'N (Dishes)', 'Median Survival Day (Approx)'])


# --- plot_cumulative_deaths_pl ---
def plot_cumulative_deaths_pl(
    df: pl.DataFrame,
    group_by: str, # 'genotype_housing', 'density_category', or 'cross_id'
    visible_conditions_list: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots the mean cumulative number of deaths over time, grouped by the specified column.
    Includes shaded standard deviation bounds. Assumes group_by column is Utf8.
    """
    print(f"Plotting cumulative deaths (group_by={group_by})...")
    plot_df = df.clone()

    group_col = group_by
    required_cols = ['dish_id', 'days_since_fertilization', 'cumulative_deaths', group_col]
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.error(f"Missing required columns for cumulative deaths plot: {missing}.")
        return go.Figure(layout=go.Layout(title="Cumulative Deaths Plot - Error: Columns Missing", height=600))

    title_map = {
        'genotype_housing': 'Cumulative Deaths by Genotype and Housing (Mean +/- Std Dev)',
        'density_category': 'Cumulative Deaths by Fish Density (Mean +/- Std Dev)',
        'cross_id': 'Cumulative Deaths by Cross ID (Mean +/- Std Dev)'
    }
    legend_title_map = {
        'genotype_housing': 'Genotype & Housing',
        'density_category': 'Fish Density',
        'cross_id': 'Cross ID'
    }
    title = title_map.get(group_col, f'Cumulative Deaths by {group_col} (Mean +/- Std Dev)')
    legend_title = legend_title_map.get(group_col, group_col)
    COLOR_SEQUENCE = px.colors.qualitative.Plotly # Use a different sequence potentially

    # Ensure group column is Utf8 for consistent processing
    if plot_df[group_col].dtype != pl.Utf8:
        plot_df = plot_df.with_columns(pl.col(group_col).cast(pl.Utf8))

    # Filter based on visible conditions (compare as strings)
    if visible_conditions_list is not None:
        if not visible_conditions_list:
            st.warning(f"No groups selected for '{group_col}' to display.")
            return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Cumulative Deaths'))
        else:
            visible_groups_str = [str(g) for g in visible_conditions_list]
            plot_df = plot_df.filter(pl.col(group_col).is_in(visible_groups_str))
            if plot_df.height == 0:
                st.warning(f"No data matches the selected filters for '{group_col}'.")
                return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Cumulative Deaths'))

    # Ensure numeric types for aggregation and drop nulls
    plot_df_agg = plot_df.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('cumulative_deaths').cast(pl.Float64, strict=False) # Use float for mean/std
    ]).drop_nulls(subset=['days_since_fertilization', 'cumulative_deaths', group_col])

    if plot_df_agg.height == 0:
        st.warning(f"No non-null data available for cumulative deaths plot ({group_col}).")
        return go.Figure(layout=go.Layout(title=title + " - No Data", height=600))

    # Aggregate mean and std dev of cumulative deaths (group by Utf8)
    agg_df = plot_df_agg.group_by([group_col, 'days_since_fertilization']).agg([
        pl.mean('cumulative_deaths').alias('mean_cumulative_deaths'),
        pl.std('cumulative_deaths').alias('std_cumulative_deaths')
    ]).sort(group_col, 'days_since_fertilization')

    # Calculate upper and lower bounds for shading
    agg_df = agg_df.with_columns([
        (pl.col('mean_cumulative_deaths') + pl.col('std_cumulative_deaths')).alias('upper_bound'),
        (pl.col('mean_cumulative_deaths') - pl.col('std_cumulative_deaths')).clip(lower_bound=0).alias('lower_bound') # Ensure non-negative
    ])

    # Get unique groups (already Utf8) and assign colors
    unique_groups = agg_df[group_col].drop_nulls().unique().sort().to_list()
    color_map = {cat: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, cat in enumerate(unique_groups)}

    # --- Create the plot using go.Scatter ---
    fig = go.Figure()

    for cat in unique_groups:
        cat_data = agg_df.filter(pl.col(group_col) == cat)
        if cat_data.height == 0: continue

        color = color_map.get(cat)
        if not color: continue
        fill_color = hex_to_rgba(color, 0.2)

        x_vals = cat_data['days_since_fertilization']
        y_upper = cat_data['upper_bound']
        y_lower = cat_data['lower_bound']
        y_mean = cat_data['mean_cumulative_deaths']
        y_std = cat_data['std_cumulative_deaths']

        # Add shaded area (Std Dev)
        fig.add_trace(go.Scatter(x=x_vals, y=y_upper, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False, name=f'{cat}_upper_death'))
        fig.add_trace(go.Scatter(x=x_vals, y=y_lower, mode='lines', line=dict(width=0), fillcolor=fill_color, fill='tonexty', hoverinfo='skip', showlegend=False, name=f'{cat}_lower_death'))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_mean, mode='lines', line=dict(width=2, color=color), name=f'{cat} (mean)', legendgroup=str(cat),
            hovertext=[f"Day: {d}<br>Avg Cum. Deaths: {m:.2f}<br>Std Dev: {'N/A' if s is None else f'{s:.2f}'}<br>Group: {cat}"
                       for d, m, s in zip(x_vals, y_mean, y_std)],
            hoverinfo="text"
        ))

    # Update layout
    fig.update_layout(
        title=title, xaxis_title='Days Since Fertilization', yaxis_title='Mean Cumulative Deaths',
        legend_title_text=legend_title, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified', height=600
    )
    fig.update_yaxes(rangemode='tozero') # Ensure y-axis starts at 0
    return fig


# --- plot_distributions_pl ---
def plot_distributions_pl(
    df: pl.DataFrame,
    metric_col: str,
    group_col: str,
    plot_type: str = 'box', # 'box' or 'histogram'
    visible_groups: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots the distribution of a metric (e.g., final survival, critical day)
    grouped by a category (e.g., genotype, density, housing). Assumes group_col is Utf8.
    """
    print(f"Plotting distribution (metric={metric_col}, group={group_col}, type={plot_type})...")
    plot_df = df.clone()

    # Check for dish_id and group_col first
    if 'dish_id' not in plot_df.columns or group_col not in plot_df.columns:
         missing_base = [col for col in ['dish_id', group_col] if col not in plot_df.columns]
         st.error(f"Missing base columns for distribution plot: {missing_base}.")
         return go.Figure(layout=go.Layout(title="Distribution Plot - Error: Base Columns Missing", height=600))

    # Check if metric column exists OR if base columns to calculate it exist
    metric_exists = metric_col in plot_df.columns
    can_calc_final_survival = metric_col == 'final_survival_rate' and 'survival_rate' in plot_df.columns and 'days_since_fertilization' in plot_df.columns
    can_calc_critical_day = metric_col == 'critical_day' and 'survival_rate_change' in plot_df.columns and 'days_since_fertilization' in plot_df.columns

    if not (metric_exists or can_calc_final_survival or can_calc_critical_day):
         st.error(f"Metric column '{metric_col}' not found and cannot be calculated from available columns.")
         return go.Figure(layout=go.Layout(title=f"Distribution Plot - Error: Metric '{metric_col}' Unavailable", height=600))


    # Ensure group_col is Utf8 for grouping consistency
    if plot_df[group_col].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col(group_col).cast(pl.Utf8))


    if metric_col == 'final_survival_rate':
        # Calculate final survival rate per dish
        plot_df = plot_df.sort('dish_id', 'days_since_fertilization').group_by('dish_id').agg(
            pl.last('survival_rate').alias('final_survival_rate'),
            pl.first(group_col).alias(group_col) # Keep the group column (already Utf8)
        ).drop_nulls(subset=['final_survival_rate', group_col])
        metric_col_to_plot = 'final_survival_rate' # Use the newly created column name

    elif metric_col == 'critical_day':
        # Calculate critical day per dish
        critical_periods = plot_df.drop_nulls(
            subset=['dish_id', group_col, 'days_since_fertilization', 'survival_rate_change']
        ).sort(
            'dish_id', 'days_since_fertilization'
        ).group_by(
            ['dish_id'], maintain_order=True # Group only by dish
        ).agg([
            pl.col('days_since_fertilization').gather(pl.col('survival_rate_change').arg_min()).first().alias('critical_day'),
            pl.first(group_col).alias(group_col) # Keep the group associated with the dish (already Utf8)
        ]).drop_nulls(subset=['critical_day', group_col])
        plot_df = critical_periods # Use this df for plotting
        metric_col_to_plot = 'critical_day'
    else:
        # For other potential metrics, assume one value per dish_id exists
         plot_df = plot_df.group_by('dish_id').agg(
             pl.first(metric_col).alias(metric_col),
             pl.first(group_col).alias(group_col) # Keep group col (already Utf8)
         ).drop_nulls(subset=[metric_col, group_col])
         metric_col_to_plot = metric_col


    # Ensure metric type is numeric
    if not isinstance(plot_df[metric_col_to_plot].dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64)):
         try:
             plot_df = plot_df.with_columns(pl.col(metric_col_to_plot).cast(pl.Float64, strict=False))
         except Exception as e:
             st.error(f"Could not convert metric column '{metric_col_to_plot}' to numeric type: {e}")
             return go.Figure(layout=go.Layout(title=f"Distribution Plot - Error: Invalid Metric Type '{metric_col_to_plot}'", height=600))

    plot_df = plot_df.drop_nulls(subset=[metric_col_to_plot, group_col])

    # Filter based on visible groups (compare as strings)
    if visible_groups is not None:
        if not visible_groups:
            st.warning(f"No groups selected for '{group_col}' to display.")
            return go.Figure(layout=go.Layout(title="Distribution Plot - No Groups Selected", height=600))
        else:
            visible_groups_str = [str(g) for g in visible_groups]
            plot_df = plot_df.filter(pl.col(group_col).is_in(visible_groups_str))
            if plot_df.height == 0:
                st.warning(f"No data matches the selected filters for '{group_col}'.")
                return go.Figure(layout=go.Layout(title="Distribution Plot - No Matching Data", height=600))

    if plot_df.height == 0:
        st.warning(f"No data available for distribution plot (metric={metric_col_to_plot}, group={group_col}).")
        return go.Figure(layout=go.Layout(title="Distribution Plot - No Data", height=600))

    title = f"Distribution of {metric_col_to_plot.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}"
    fig = None

    # Create plot using pandas df for plotly express compatibility
    plot_pd = plot_df.to_pandas()
    if plot_type == 'box':
        fig = px.box(plot_pd, x=group_col, y=metric_col_to_plot, color=group_col,
                     points="all", # Show individual points
                     title=title,
                     labels={metric_col_to_plot: metric_col_to_plot.replace('_', ' ').title(), group_col: group_col.replace('_', ' ').title()})
    elif plot_type == 'histogram':
        fig = px.histogram(plot_pd, x=metric_col_to_plot, color=group_col,
                           marginal="rug", # Add rug plot
                           barmode='overlay', # Overlay histograms
                           opacity=0.7,
                           title=title,
                           labels={metric_col_to_plot: metric_col_to_plot.replace('_', ' ').title(), group_col: group_col.replace('_', ' ').title()})
        fig.update_layout(bargap=0.1)
    else:
        st.error(f"Invalid distribution plot type: {plot_type}. Choose 'box' or 'histogram'.")
        return go.Figure(layout=go.Layout(title="Distribution Plot - Invalid Type", height=600))

    fig.update_layout(
        xaxis_title=group_col.replace('_', ' ').title(),
        yaxis_title=metric_col_to_plot.replace('_', ' ').title(),
        legend_title_text=group_col.replace('_', ' ').title(),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=600
    )

    return fig


# --- Statistical Test Functions ---

# --- perform_housing_ttest ---
@st.cache_data
def perform_housing_ttest(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Performs an independent two-sample t-test comparing final survival rates
    between housing types ('Beaker' vs 'Dish') for each genotype.
    Assumes unequal variances (Welch's t-test). Assumes genotype and housing cols are Utf8.
    """
    if not SCIPY_AVAILABLE:
        st.error("Statistical tests require the `scipy` library. Please install it (`pip install scipy`).")
        return None

    print("Performing Housing T-tests by Genotype...")
    test_df = df.clone()

    required_cols = ['dish_id', 'days_since_fertilization', 'survival_rate', 'genotype', 'housing'] # Use housing directly
    if not all(col in test_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in test_df.columns]
        st.warning(f"Missing required columns for Housing T-test: {missing}. Skipping tests.")
        return None

    # Ensure housing and genotype are Utf8 for consistent processing
    if test_df['housing'].dtype != pl.Utf8:
         test_df = test_df.with_columns(pl.col('housing').cast(pl.Utf8))
    if test_df['genotype'].dtype != pl.Utf8:
         test_df = test_df.with_columns(pl.col('genotype').cast(pl.Utf8))


    # Get final survival rate per dish
    final_survival = test_df.sort('dish_id', 'days_since_fertilization').group_by('dish_id').agg(
        pl.last('survival_rate').alias('final_survival_rate'),
        pl.first('genotype').alias('genotype'), # Already Utf8
        pl.first('housing').alias('housing') # Already Utf8
    ).drop_nulls()

    if final_survival.height < 2: # Need at least two data points overall
        st.warning("Insufficient data (less than 2 dishes with final survival) for Housing T-tests.")
        return None

    results = []
    genotypes = final_survival['genotype'].drop_nulls().unique().sort().to_list()

    for genotype in genotypes:
        genotype_data = final_survival.filter(pl.col('genotype') == genotype)
        # Filter housing using string literals
        beaker_data = genotype_data.filter(pl.col('housing') == 'Beaker')['final_survival_rate'].to_numpy()
        dish_data = genotype_data.filter(pl.col('housing') == 'Dish')['final_survival_rate'].to_numpy()

        # Check if enough data in both groups for this genotype
        if len(beaker_data) < 2 or len(dish_data) < 2:
            print(f"Skipping t-test for genotype '{genotype}': insufficient data (need >= 2 in each group). Beaker: {len(beaker_data)}, Dish: {len(dish_data)}")
            results.append({
                'Genotype': genotype,
                'T-Statistic': 'N/A',
                'P-Value': 'N/A',
                'Comparison': 'Beaker vs Dish',
                'N_Beaker': len(beaker_data),
                'N_Dish': len(dish_data),
                'Mean_Beaker': np.mean(beaker_data) if len(beaker_data) > 0 else 'N/A',
                'Mean_Dish': np.mean(dish_data) if len(dish_data) > 0 else 'N/A',
                'Note': 'Insufficient data'
            })
            continue

        try:
            # Perform Welch's t-test (assumes unequal variances)
            t_stat, p_value = stats.ttest_ind(beaker_data, dish_data, equal_var=False, nan_policy='omit')

            if np.isnan(t_stat) or np.isnan(p_value):
                 note = "Calculation resulted in NaN"
                 t_stat_disp, p_value_disp = 'NaN', 'NaN'
            else:
                 note = "Significant (p<0.05)" if p_value < 0.05 else "Not Significant (p>=0.05)"
                 t_stat_disp, p_value_disp = f"{t_stat:.3f}", f"{p_value:.4f}"

            results.append({
                'Genotype': genotype,
                'T-Statistic': t_stat_disp,
                'P-Value': p_value_disp,
                'Comparison': 'Beaker vs Dish (Final Survival)',
                'N_Beaker': len(beaker_data),
                'N_Dish': len(dish_data),
                'Mean_Beaker': f"{np.mean(beaker_data):.2f}%" if len(beaker_data) > 0 else 'N/A',
                'Mean_Dish': f"{np.mean(dish_data):.2f}%" if len(dish_data) > 0 else 'N/A',
                'Note': note
            })
        except Exception as e:
            print(f"Error during t-test for genotype '{genotype}': {e}")
            results.append({
                'Genotype': genotype,
                'T-Statistic': 'Error',
                'P-Value': 'Error',
                'Comparison': 'Beaker vs Dish',
                'N_Beaker': len(beaker_data),
                'N_Dish': len(dish_data),
                 'Mean_Beaker': 'Error',
                'Mean_Dish': 'Error',
                'Note': f"Error: {e}"
            })

    if not results:
        st.warning("No genotypes had sufficient data in both housing groups for T-tests.")
        return None

    return pl.DataFrame(results)


# --- perform_density_anova ---
@st.cache_data
def perform_density_anova(df: pl.DataFrame, metric_col: str = 'final_survival_rate') -> Optional[pl.DataFrame]:
    """
    Performs a One-Way ANOVA comparing a metric (default: final survival rate)
    across different density categories. Assumes density_category col is Utf8.
    """
    if not SCIPY_AVAILABLE:
        st.error("Statistical tests require the `scipy` library. Please install it (`pip install scipy`).")
        return None

    print(f"Performing Density ANOVA for metric '{metric_col}'...")
    test_df = df.clone()

    required_cols = ['dish_id', 'days_since_fertilization', 'density_category']
    metric_to_prepare = None
    # Check if metric column exists OR if base columns to calculate it exist
    metric_exists = metric_col in test_df.columns
    can_calc_final_survival = metric_col == 'final_survival_rate' and 'survival_rate' in test_df.columns
    can_calc_critical_day = metric_col == 'critical_day' and 'survival_rate_change' in test_df.columns

    if metric_col == 'final_survival_rate':
        if not can_calc_final_survival:
             st.warning("Column 'survival_rate' needed for Density ANOVA on Final Survival is missing.")
             return None
        required_cols.append('survival_rate')
        metric_to_prepare = 'final_survival_rate'
    elif metric_col == 'critical_day':
         if not can_calc_critical_day:
              st.warning("Column 'survival_rate_change' needed to calculate 'critical_day' for ANOVA is missing.")
              return None
         required_cols.append('survival_rate_change')
         metric_to_prepare = 'critical_day'
    elif not metric_exists:
        st.warning(f"Metric column '{metric_col}' not found for Density ANOVA.")
        return None
    else:
         required_cols.append(metric_col) # Add the metric column itself if it exists
         metric_to_prepare = metric_col


    if not all(col in test_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in test_df.columns]
        st.warning(f"Missing required columns for Density ANOVA: {missing}. Skipping test.")
        return None

    # Ensure density category is Utf8 for consistent processing
    if test_df['density_category'].dtype != pl.Utf8:
        test_df = test_df.with_columns(pl.col('density_category').cast(pl.Utf8))


    # Prepare the metric data per dish
    if metric_to_prepare == 'final_survival_rate':
        metric_data_per_dish = test_df.sort('dish_id', 'days_since_fertilization').group_by('dish_id').agg(
            pl.last('survival_rate').alias(metric_col),
            pl.first('density_category').alias('density_category') # Already Utf8
        ).drop_nulls(subset=[metric_col, 'density_category'])
        metric_label = "Final Survival Rate"
    elif metric_to_prepare == 'critical_day':
         # Calculate critical day per dish
        critical_periods = test_df.drop_nulls(
            subset=['dish_id', 'density_category', 'days_since_fertilization', 'survival_rate_change']
        ).sort(
            'dish_id', 'days_since_fertilization'
        ).group_by(
            ['dish_id'], maintain_order=True
        ).agg([
            pl.col('days_since_fertilization').gather(pl.col('survival_rate_change').arg_min()).first().alias(metric_col),
            pl.first('density_category').alias('density_category') # Keep the group (already Utf8)
        ]).drop_nulls(subset=[metric_col, 'density_category'])
        metric_data_per_dish = critical_periods
        metric_label = "Critical Day (Day of Max Decline)"
    else:
        # Assume metric exists, take first value per dish
        metric_data_per_dish = test_df.group_by('dish_id').agg(
             pl.first(metric_col).alias(metric_col),
             pl.first('density_category').alias('density_category') # Already Utf8
         ).drop_nulls(subset=[metric_col, 'density_category'])
        metric_label = metric_col.replace('_', ' ').title()

    if metric_data_per_dish.height == 0:
        st.warning(f"No non-null data available for metric '{metric_label}' after processing.")
        return None


    # Group data by density category for ANOVA input
    density_groups = metric_data_per_dish['density_category'].drop_nulls().unique().sort().to_list()
    grouped_data = []
    group_stats = []

    for density_cat in density_groups:
        # Filter using string literal
        data = metric_data_per_dish.filter(pl.col('density_category') == density_cat)[metric_col].to_numpy()
        # Ensure data is finite for stats calculation
        data = data[np.isfinite(data)]

        if len(data) >= 2: # Need at least 2 data points per group for ANOVA
            grouped_data.append(data)
            group_stats.append({
                'Density Category': density_cat,
                'N': len(data),
                f'Mean {metric_label}': f"{np.mean(data):.2f}" + ("%" if metric_col=='final_survival_rate' else ""),
                f'Std Dev {metric_label}': f"{np.std(data):.2f}"
            })
        else:
             print(f"Skipping density category '{density_cat}' for ANOVA: insufficient finite data (found {len(data)}, need >= 2).")
             # Still add row to stats table, but indicate N/A for stats
             mean_val = np.mean(data) if len(data) > 0 else 'N/A'
             std_val = np.std(data) if len(data) > 0 else 'N/A'
             mean_disp = f"{mean_val:.2f}%" if isinstance(mean_val, (int, float)) and metric_col=='final_survival_rate' else str(mean_val)
             std_disp = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else str(std_val)

             group_stats.append({
                'Density Category': density_cat,
                'N': len(data),
                f'Mean {metric_label}': mean_disp,
                f'Std Dev {metric_label}': std_disp,
            })


    if len(grouped_data) < 2: # Need at least 2 groups for ANOVA
        st.warning(f"Insufficient groups ({len(grouped_data)}) with enough data (>=2 points) for ANOVA on '{metric_label}'.")
        if group_stats:
            stats_only_df = pl.DataFrame(group_stats)
            # Ensure columns exist before selecting
            cols_to_select = ['Density Category', 'N']
            if f'Mean {metric_label}' in stats_only_df.columns: cols_to_select.append(f'Mean {metric_label}')
            if f'Std Dev {metric_label}' in stats_only_df.columns: cols_to_select.append(f'Std Dev {metric_label}')
            return stats_only_df.select(cols_to_select).with_columns(pl.lit("ANOVA Not Performed").alias("ANOVA Result"))
        else: return None


    results = {}
    try:
        # Perform One-Way ANOVA
        f_stat, p_value = stats.f_oneway(*grouped_data) # Pass unpacked list of arrays

        results['Metric Compared'] = metric_label
        results['F-Statistic'] = f"{f_stat:.3f}" if not np.isnan(f_stat) else 'NaN'
        results['P-Value'] = f"{p_value:.4f}" if not np.isnan(p_value) else 'NaN'
        if not np.isnan(p_value):
             results['Conclusion'] = f"Significant difference between density groups (p<0.05)" if p_value < 0.05 else f"No significant difference between density groups (p>=0.05)"
        else: results['Conclusion'] = "Result is NaN"

    except Exception as e:
        print(f"Error during ANOVA for metric '{metric_col}': {e}")
        results['Metric Compared'] = metric_label
        results['F-Statistic'] = 'Error'
        results['P-Value'] = 'Error'
        results['Conclusion'] = f"Error during calculation: {e}"

    # Combine results and group stats
    summary_df = pl.DataFrame(group_stats)
    anova_result_df = pl.DataFrame([results])

    # Add ANOVA result columns to the summary table
    for col, val in results.items():
        summary_df = summary_df.with_columns(pl.lit(val).alias(col))

    # Ensure columns exist before selecting
    final_cols_order = ['Density Category', 'N']
    if f'Mean {metric_label}' in summary_df.columns: final_cols_order.append(f'Mean {metric_label}')
    if f'Std Dev {metric_label}' in summary_df.columns: final_cols_order.append(f'Std Dev {metric_label}')
    final_cols_order.extend(['F-Statistic', 'P-Value', 'Conclusion'])

    # Select only existing columns in the final order
    existing_final_cols = [col for col in final_cols_order if col in summary_df.columns]

    return summary_df.select(existing_final_cols)


# --- plot_housing_comparison_pl ---
def plot_housing_comparison_pl(df: pl.DataFrame) -> go.Figure:
    """Compare survival rates between beaker and dish housing using Polars and Plotly. Assumes genotype and housing cols are Utf8."""
    print("Plotting housing comparison...")
    plot_df = df.clone() # Work on a copy

    required_cols = ['dish_id', 'days_since_fertilization', 'survival_rate', 'genotype', 'housing'] # Use housing directly
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.warning(f"Missing required columns for housing comparison: {missing}. Skipping plot.")
        return go.Figure(layout=go.Layout(title="Housing comparison not available - Required columns missing"))

    # Ensure housing and genotype are Utf8 for consistent processing
    if plot_df['housing'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('housing').cast(pl.Utf8))
    if plot_df['genotype'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('genotype').cast(pl.Utf8))


    # Ensure days_since_fertilization is numeric for max()
    plot_df = plot_df.with_columns(pl.col('days_since_fertilization').cast(pl.Int64, strict=False))

    latest_records = plot_df.filter(
        pl.col('days_since_fertilization') == pl.max('days_since_fertilization').over('dish_id')
    ).drop_nulls(subset=['survival_rate', 'genotype', 'housing']) # Drop nulls needed for grouping

    if latest_records.height == 0:
         st.warning("No data available for housing comparison plot after filtering for latest records.")
         return go.Figure(layout=go.Layout(title="Housing comparison - No data"))

    # Ensure survival_rate is float for aggregations
    latest_records = latest_records.with_columns(pl.col('survival_rate').cast(pl.Float64, strict=False))

    summary = latest_records.group_by(['genotype', 'housing']).agg(
        pl.mean('survival_rate').alias('mean'),
        pl.std('survival_rate').alias('std'),
        pl.len().alias('count')
    ).with_columns(
        pl.when(pl.col('count') > 0)
        .then(pl.col('std') / pl.col('count').sqrt()) # Calculate Standard Error
        .otherwise(None)
        .alias('se')
    ).sort('genotype', 'housing')

    if summary.height == 0:
         st.warning("No summary data available for housing comparison plot.")
         return go.Figure(layout=go.Layout(title="Housing comparison - No summary data"))

    fig = px.bar(
        summary.to_pandas(), x='genotype', y='mean', color='housing', barmode='group', error_y='se',
        labels={'mean': 'Final Survival Rate (%)', 'genotype': 'Genotype', 'housing': 'Housing Type', 'se': 'Standard Error'},
        title='Comparison of Final Survival Rates by Genotype and Housing Type',
        color_discrete_map={'Beaker': px.colors.qualitative.Plotly[0], 'Dish': px.colors.qualitative.Plotly[1]} # Consistent colors
    )

    # Add individual points
    if latest_records.height > 0:
        color_discrete_map_points = {
            'Beaker': px.colors.qualitative.Plotly[0],
            'Dish': px.colors.qualitative.Plotly[1]
        }
        for housing_type in ['Beaker', 'Dish']:
            housing_data = latest_records.filter(pl.col('housing') == housing_type)
            if housing_data.height > 0:
                 bar_color = color_discrete_map_points.get(housing_type, 'grey')
                 # Use add_trace for better control over hovertemplate
                 fig.add_trace(
                     go.Scatter(
                         x=housing_data['genotype'], y=housing_data['survival_rate'], mode='markers',
                         marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black'), color=bar_color),
                         name=f'{housing_type} (individual)', legendgroup=housing_type, showlegend=False, # Hide duplicate legend entry
                         customdata=housing_data[['dish_id', 'genotype', 'housing', 'survival_rate']].to_numpy(),
                         hovertemplate = (
                            "<b>Dish:</b> %{customdata[0]}<br>" +
                            "<b>Genotype:</b> %{customdata[1]}<br>" +
                            "<b>Housing:</b> %{customdata[2]}<br>" +
                            "<b>Final Survival:</b> %{customdata[3]:.1f}%" +
                            "<extra></extra>" # Hide trace name in hover
                         )
                     )
                 )

    fig.update_layout(
        xaxis_title='Genotype', yaxis_title='Survival Rate (%)', yaxis=dict(range=[0, 100.5]), # Extend slightly
        legend_title_text='Housing Type',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest', height=600
    )
    return fig


# --- plot_density_vs_survival_pl ---
def plot_density_vs_survival_pl(
    df: pl.DataFrame,
    days: Optional[int] = None,
    use_volume_per_fish: bool = True,
    visible_conditions_list: Optional[List[str]] = None # Genotypes to show
) -> go.Figure:
    """
    Create a scatter plot (density vs survival) using Polars and Plotly. Assumes genotype col is Utf8.
    """
    print(f"Plotting density vs survival (use_volume_per_fish={use_volume_per_fish})...") # Keep print for debug
    plot_df = df.clone() # Work on a copy

    required_base = ['dish_id', 'days_since_fertilization', 'survival_rate', 'genotype', 'initial_count']
    required_density = ['volume_per_fish'] if use_volume_per_fish else ['fish_density']
    density_col_name = required_density[0]

    if not all(col in plot_df.columns for col in required_base + required_density):
         missing = [col for col in required_base + required_density if col not in plot_df.columns]
         st.warning(f"Missing required columns for density vs survival plot: {missing}. Skipping plot.")
         return go.Figure(layout=go.Layout(title="Density vs Survival plot not available - Required columns missing"))

    # Ensure days_since_fertilization is numeric for filtering/max()
    plot_df = plot_df.with_columns(pl.col('days_since_fertilization').cast(pl.Int64, strict=False))

    if days is not None:
        plot_df = plot_df.filter(pl.col('days_since_fertilization') == days)
        plot_title_suffix = f' (Day {days})'
    else:
        # Get latest measurement per dish
        plot_df = plot_df.filter(
            pl.col('days_since_fertilization') == pl.max('days_since_fertilization').over('dish_id')
        )
        plot_title_suffix = ' (Latest Measurement)'

    x_column = density_col_name
    x_label = 'Volume per Fish (mL/fish)' if use_volume_per_fish else 'Fish Density (fish/mL)'
    # Ensure numeric type for plotting and calculations
    plot_df = plot_df.with_columns(pl.col(x_column).cast(pl.Float64, strict=False))


    plot_title = f'{"Volume per Fish" if use_volume_per_fish else "Fish Density"} vs Survival Rate{plot_title_suffix}'

    # Ensure survival_rate and initial_count are numeric
    plot_df = plot_df.with_columns([
        pl.col('survival_rate').cast(pl.Float64, strict=False),
        pl.col('initial_count').cast(pl.Int64, strict=False) # For size mapping
        ])

    # Ensure genotype is Utf8 for filtering/coloring
    if 'genotype' in plot_df.columns and plot_df['genotype'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('genotype').cast(pl.Utf8))


    # Filter based on visible genotypes (compare as strings)
    plot_df_filtered = plot_df
    if visible_conditions_list is not None:
        if not visible_conditions_list:
             st.warning("No genotypes selected to display.")
             return go.Figure(layout=go.Layout(title=f"{plot_title} - No data to display"))
        else:
            print(f"Filtering scatter plot data for: {visible_conditions_list}")
            visible_groups_str = [str(g) for g in visible_conditions_list]
            plot_df_filtered = plot_df_filtered.filter(pl.col('genotype').is_in(visible_groups_str))
            if plot_df_filtered.height == 0:
                 st.warning("No data matches the selected filters.")
                 return go.Figure(layout=go.Layout(title=f"{plot_title} - No data to display"))

    # Drop nulls required for plotting
    plot_df_filtered = plot_df_filtered.drop_nulls(subset=[x_column, 'survival_rate', 'genotype', 'initial_count'])

    if plot_df_filtered.height == 0:
         st.warning("No data available for density vs survival plot after filtering and cleaning.")
         return go.Figure(layout=go.Layout(title=f"{plot_title} - No data"))

    fig = px.scatter(
        plot_df_filtered.to_pandas(), x=x_column, y='survival_rate', color='genotype', size='initial_count',
        hover_data=['dish_id', 'days_since_fertilization', 'vol_water_total', 'housing', 'density_category'], # Use housing
        labels={x_column: x_label, 'survival_rate': 'Survival Rate (%)', 'genotype': 'Genotype', 'initial_count': 'Initial Fish Count'},
        title=plot_title
    )

    # Add regression line and correlation (overall trend)
    if plot_df_filtered.height > 1:
        x_np = plot_df_filtered.get_column(x_column).to_numpy()
        y_np = plot_df_filtered.get_column('survival_rate').to_numpy()
        # Ensure finite values before calculations
        finite_mask = np.isfinite(x_np) & np.isfinite(y_np)
        x_np_finite = x_np[finite_mask]
        y_np_finite = y_np[finite_mask]

        if len(x_np_finite) > 1: # Need at least 2 points for trend
            try:
                # Check for sufficient variation
                if np.std(x_np_finite) > 1e-6 and np.std(y_np_finite) > 1e-6:
                    coeffs = np.polyfit(x_np_finite, y_np_finite, 1)
                    if not np.isnan(coeffs).any():
                        line_x = np.array([x_np_finite.min(), x_np_finite.max()])
                        line_y = np.polyval(coeffs, line_x).clip(0, 100) # Clip trend line to 0-100
                        fig.add_trace(
                            go.Scatter(x=line_x, y=line_y, mode='lines', name=f'Overall Trend', line=dict(color='rgba(0,0,0,0.7)', width=2, dash='dash'))
                        )
                        correlation = np.corrcoef(x_np_finite, y_np_finite)[0, 1]
                        if not np.isnan(correlation):
                             # Position annotation carefully
                             fig.add_annotation(
                                 x=0.98, y=0.02, xref="paper", yref="paper", align='right',
                                 text=f"Overall Trend:<br>y={coeffs[0]:.2f}x+{coeffs[1]:.2f}<br>R = {correlation:.3f}",
                                 showarrow=False, font=dict(size=12, color="black"),
                                 bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, borderpad=4
                             )
                        else: print("Warning: Correlation calculation resulted in NaN.")
                    else: print("Warning: Trend line calculation resulted in NaN coefficients.")
                else: print("Warning: Insufficient variation in finite data; skipping trend line/correlation.")
            except Exception as e: print(f"Could not calculate or add trend line/correlation: {e}")
        else: print("Warning: Less than 2 finite data points; skipping trend line/correlation.")


    fig.update_layout(
        xaxis_title=x_label, yaxis_title='Survival Rate (%)', yaxis=dict(range=[0, 100.5]), # Extend slightly
        legend_title_text='Genotype',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest', height=600
    )
    return fig


# --- plot_remaining_by_group_pl ---
def plot_remaining_by_group_pl(df: pl.DataFrame) -> go.Figure:
    """
    Plots the average number of remaining fish over time, grouped by housing and density,
    including shaded standard deviation bounds. Assumes housing and density_category cols are Utf8.
    """
    print("Plotting remaining fish by housing and density with std bounds...")
    plot_df = df.clone() # Work on a copy

    required_cols = ['days_since_fertilization', 'remaining', 'housing', 'density_category'] # Use housing directly
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.warning(f"Missing required columns for Remaining Fish plot: {missing}. Skipping plot.")
        return go.Figure(layout=go.Layout(title="Remaining Fish by Housing/Density not available - Required columns missing"))

    # Ensure grouping columns are Utf8
    if plot_df['housing'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('housing').cast(pl.Utf8))
    if plot_df['density_category'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('density_category').cast(pl.Utf8))


    # Create combined grouping column, handling potential nulls
    plot_df = plot_df.with_columns(
        pl.concat_str(
            [pl.col("housing").fill_null("Unknown"), pl.lit(" - "), pl.col("density_category").fill_null("Unknown")],
            separator=""
        ).alias("housing_density_group") # Already Utf8
    )

    # Ensure numeric types for aggregation
    plot_df = plot_df.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('remaining').cast(pl.Float64, strict=False) # Use float for mean/std
    ]).drop_nulls(subset=['days_since_fertilization', 'remaining', 'housing_density_group']) # Drop nulls before grouping

    if plot_df.height == 0:
        st.warning("No data available for Remaining Fish plot after cleaning.")
        return go.Figure(layout=go.Layout(title="Remaining Fish by Housing & Density - No data"))

    # Group by combined category and day, calculate mean and std remaining
    agg_df = plot_df.group_by(['housing_density_group', 'days_since_fertilization']).agg([
        pl.mean('remaining').alias('mean_remaining'),
        pl.std('remaining').alias('std_remaining') # Calculate standard deviation
    ]).sort('housing_density_group', 'days_since_fertilization')

    # Calculate bounds
    agg_df = agg_df.with_columns([
        (pl.col('mean_remaining') + pl.col('std_remaining')).alias('upper_bound'),
        (pl.col('mean_remaining') - pl.col('std_remaining')).clip(lower_bound=0).alias('lower_bound') # Ensure lower bound doesn't go below 0
    ])


    if agg_df.height == 0:
        st.warning("No aggregated data available for Remaining Fish plot.")
        return go.Figure(layout=go.Layout(title="Remaining Fish by Housing & Density - No aggregated data"))

    # --- Create the plot using go.Scatter for more control ---
    fig = go.Figure()
    unique_groups = agg_df['housing_density_group'].drop_nulls().unique().sort().to_list() # Already Utf8
    colors = px.colors.qualitative.Plotly # Get a color sequence

    for i, group_name in enumerate(unique_groups):
        group_data = agg_df.filter(pl.col('housing_density_group') == group_name)
        if group_data.height == 0:
            continue

        color = colors[i % len(colors)] # Cycle through colors
        fill_color = hex_to_rgba(color, 0.2) # Use helper

        x_vals = group_data['days_since_fertilization']
        y_upper = group_data['upper_bound']
        y_lower = group_data['lower_bound']
        y_mean = group_data['mean_remaining']
        y_std = group_data['std_remaining'] # Get std dev column

        # Add upper bound trace (invisible line)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_upper,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip', # Don't show hover for bounds
            showlegend=False,
            name=f'{group_name}_upper' # Unique name for linking fill
        ))

        # Add lower bound trace (invisible line, fill to upper bound)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_lower,
            mode='lines',
            line=dict(width=0),
            fillcolor=fill_color,
            fill='tonexty', # Fill area between this trace and the previous one (upper bound)
            hoverinfo='skip', # Don't show hover for bounds
            showlegend=False,
            name=f'{group_name}_lower' # Unique name
        ))

        # Add mean line trace (visible)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_mean,
            mode='lines+markers',
            line=dict(color=color),
            marker=dict(size=6),
            name=str(group_name), # Ensure name is string for legend
            hovertext=[f"Day: {d}<br>Avg Remaining: {m:.2f}<br>Std Dev: {'N/A' if s is None else f'{s:.2f}'}<br>Group: {group_name}"
                       for d, m, s in zip(x_vals, y_mean, y_std)], # Use y_std here
            hoverinfo="text"
        ))

    # Update layout
    fig.update_layout(
        title='Average Fish Remaining by Housing Type and Density Category (Mean +/- Std Dev)',
        xaxis_title='Days Since Fertilization',
        yaxis_title='Average Fish Remaining',
        legend_title_text='Housing - Density',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest',
        height=600
    )
    fig.update_yaxes(rangemode='tozero')
    return fig


# --- plot_survival_rate_change_pl ---
def plot_survival_rate_change_pl(
    df: pl.DataFrame,
    visible_conditions_list: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots the average daily change in survival rate over time,
    grouped by genotype and housing, including shaded standard error bounds. Assumes genotype_housing col is Utf8.
    """
    print("Plotting daily survival rate change by Genotype/Housing...")
    plot_df = df.clone()

    required_cols = ['days_since_fertilization', 'survival_rate_change', 'genotype_housing'] # Use combined directly
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.warning(f"Missing required columns for Rate Change plot: {missing}. Skipping plot.")
        return go.Figure(layout=go.Layout(title="Daily Survival Rate Change not available - Required columns missing"))

    # Ensure genotype_housing is Utf8
    if plot_df['genotype_housing'].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col('genotype_housing').cast(pl.Utf8))

    color_col = 'genotype_housing'
    title = 'Daily Change in Survival Rate by Genotype/Housing (Mean +/- SE)'
    legend_title = 'Genotype & Housing'

    plot_df_filtered = plot_df
    if visible_conditions_list is not None:
        if not visible_conditions_list:
             st.warning(f"No groups selected for '{color_col}' to display.")
             return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)'))
        else:
            print(f"Filtering rate change plot data for {color_col}: {visible_conditions_list}")
            visible_groups_str = [str(g) for g in visible_conditions_list]
            plot_df_filtered = plot_df_filtered.filter(pl.col(color_col).is_in(visible_groups_str))
            if plot_df_filtered.height == 0:
                 st.warning(f"No data matches the selected filters for '{color_col}'.")
                 return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)'))

    plot_df_agg = plot_df_filtered.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('survival_rate_change').cast(pl.Float64, strict=False)
    ]).drop_nulls(subset=['days_since_fertilization', 'survival_rate_change', color_col])

    if plot_df_agg.height == 0:
        st.warning(f"No data available for Rate Change plot ({color_col}) after cleaning.")
        return go.Figure(layout=go.Layout(title=title + " - No data"))

    agg_df = plot_df_agg.group_by([color_col, 'days_since_fertilization']).agg([
        pl.mean('survival_rate_change').alias('mean_change'),
        pl.std('survival_rate_change').alias('std_change'),
        pl.len().alias('count')
    ]).sort(color_col, 'days_since_fertilization')

    if agg_df.height == 0:
        st.warning("No aggregated data groups found (perhaps due to filtering or lack of data).")
        return go.Figure(layout=go.Layout(title=title + " - Insufficient data"))

    agg_df = agg_df.with_columns(
        pl.when(pl.col('count') > 0).then(pl.col('std_change') / pl.col('count').sqrt()).otherwise(None).alias('se_change')
    ).with_columns([
        (pl.col('mean_change') + pl.col('se_change')).alias('upper_bound'),
        (pl.col('mean_change') - pl.col('se_change')).alias('lower_bound')
    ])

    fig = go.Figure()
    unique_groups = agg_df[color_col].drop_nulls().unique().sort().to_list() # Already Utf8
    colors = px.colors.qualitative.Plotly

    for i, group_name in enumerate(unique_groups):
        group_data = agg_df.filter(pl.col(color_col) == group_name)
        if group_data.height == 0: continue

        color = colors[i % len(colors)]
        fill_color = hex_to_rgba(color, 0.2) # Use helper

        x_vals = group_data['days_since_fertilization']
        y_upper = group_data['upper_bound']
        y_lower = group_data['lower_bound']
        y_mean = group_data['mean_change']
        y_se = group_data['se_change']

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_upper, mode='lines', line=dict(width=0),
            hoverinfo='skip', showlegend=False, name=f'{group_name}_upper_se'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_lower, mode='lines', line=dict(width=0),
            fillcolor=fill_color, fill='tonexty',
            hoverinfo='skip', showlegend=False, name=f'{group_name}_lower_se'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_mean, mode='lines+markers',
            line=dict(color=color), marker=dict(size=6), name=str(group_name),
            hovertext=[f"Day: {d}<br>Avg Change: {m:.2f}%/day<br>SE: {'N/A' if s is None else f'{s:.2f}'}<br>Group: {group_name}"
                       for d, m, s in zip(x_vals, y_mean, y_se)],
            hoverinfo="text"
        ))

    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="red", opacity=0.7, annotation_text="No Change", annotation_position="bottom right")
    fig.update_layout(
        title=title, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)',
        legend_title_text=legend_title, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified', height=600
    )
    return fig


# --- plot_rate_change_by_density_pl ---
def plot_rate_change_by_density_pl(
    df: pl.DataFrame,
    visible_conditions_list: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots the average daily change in survival rate over time,
    grouped by density category, including shaded standard error bounds and annotations. Assumes density_category col is Utf8.
    """
    print("Plotting daily survival rate change by density...")
    plot_df = df.clone()

    required_cols = ['days_since_fertilization', 'survival_rate_change', 'density_category']
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.warning(f"Missing required columns for Rate Change by Density plot: {missing}. Skipping plot.")
        return go.Figure(layout=go.Layout(title="Daily Survival Rate Change by Density not available - Required columns missing"))

    color_col = 'density_category'
    title = 'Daily Change in Survival Rate by Density Category (Mean +/- SE)'
    legend_title = 'Density Category'

    # Ensure density_category is Utf8
    if plot_df[color_col].dtype != pl.Utf8:
         plot_df = plot_df.with_columns(pl.col(color_col).cast(pl.Utf8))

    plot_df_filtered = plot_df
    if visible_conditions_list is not None:
        if not visible_conditions_list:
             st.warning(f"No groups selected for '{color_col}' to display.")
             return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)'))
        else:
            print(f"Filtering rate change plot data for {color_col}: {visible_conditions_list}")
            visible_groups_str = [str(g) for g in visible_conditions_list]
            plot_df_filtered = plot_df_filtered.filter(pl.col(color_col).is_in(visible_groups_str))
            if plot_df_filtered.height == 0:
                 st.warning(f"No data matches the selected filters for '{color_col}'.")
                 return go.Figure(layout=go.Layout(title=title, height=600, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)'))

    plot_df_agg = plot_df_filtered.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('survival_rate_change').cast(pl.Float64, strict=False)
    ]).drop_nulls(subset=['days_since_fertilization', 'survival_rate_change', color_col])

    if plot_df_agg.height == 0:
        st.warning("No data available for Rate Change by Density plot after cleaning.")
        return go.Figure(layout=go.Layout(title=title + " - No data"))

    agg_df = plot_df_agg.group_by([color_col, 'days_since_fertilization']).agg([
        pl.mean('survival_rate_change').alias('mean_change'),
        pl.std('survival_rate_change').alias('std_change'),
        pl.len().alias('count')
    ]).sort(color_col, 'days_since_fertilization')

    # Filter groups with at least 3 data points for SE calculation reliability
    agg_df = agg_df.filter(pl.col('count') >= 3) # Keep this filter

    if agg_df.height == 0:
        st.warning("No aggregated data groups with sufficient data points (>=3) found.")
        return go.Figure(layout=go.Layout(title=title + " - Insufficient data"))

    agg_df = agg_df.with_columns(
        pl.when(pl.col('count') > 0).then(pl.col('std_change') / pl.col('count').sqrt()).otherwise(None).alias('se_change')
    ).with_columns([
        (pl.col('mean_change') + pl.col('se_change')).alias('upper_bound'),
        (pl.col('mean_change') - pl.col('se_change')).alias('lower_bound')
    ])

    fig = go.Figure()
    unique_groups = agg_df[color_col].drop_nulls().unique().sort().to_list() # Already Utf8
    colors = px.colors.qualitative.Plotly
    critical_points = {} # To store annotation points

    for i, group_name in enumerate(unique_groups):
        group_data = agg_df.filter(pl.col(color_col) == group_name)
        if group_data.height == 0: continue

        color = colors[i % len(colors)]
        fill_color = hex_to_rgba(color, 0.2) # Use helper

        x_vals = group_data['days_since_fertilization']
        y_upper = group_data['upper_bound']
        y_lower = group_data['lower_bound']
        y_mean = group_data['mean_change']
        y_se = group_data['se_change']

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_upper, mode='lines', line=dict(width=0),
            hoverinfo='skip', showlegend=False, name=f'{group_name}_upper_se'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_lower, mode='lines', line=dict(width=0),
            fillcolor=fill_color, fill='tonexty',
            hoverinfo='skip', showlegend=False, name=f'{group_name}_lower_se'
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_mean, mode='lines+markers',
            line=dict(color=color), marker=dict(size=6), name=str(group_name),
            hovertext=[f"Day: {d}<br>Avg Change: {m:.2f}%/day<br>SE: {'N/A' if s is None else f'{s:.2f}'}<br>Density: {group_name}"
                       for d, m, s in zip(x_vals, y_mean, y_se)],
            hoverinfo="text"
        ))

        # Find point of maximum decline (most negative mean_change) for annotation
        if group_data.height > 0:
            min_mean_change_row = group_data.filter(pl.col('mean_change') == pl.min('mean_change'))
            if min_mean_change_row.height > 0:
                 # Take the first one if multiple days have the same min change
                 critical_day = min_mean_change_row['days_since_fertilization'][0]
                 critical_value = min_mean_change_row['mean_change'][0]
                 critical_points[group_name] = {'day': critical_day, 'value': critical_value, 'color': color}

    # Add annotations for critical points
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="red", opacity=0.7, annotation_text="No Change", annotation_position="bottom right")

    # Sort annotations by day to potentially improve layout later if needed
    sorted_critical_days = sorted(critical_points.items(), key=lambda item: item[1]['day'])

    for group_name, point_info in sorted_critical_days:
        critical_day = point_info['day']
        critical_value = point_info['value']
        color = point_info['color']
        # Add a vertical line marker (optional)
        # fig.add_vline(x=critical_day, line_width=1, line_dash="dot", line_color='grey', opacity=0.5)
        fig.add_annotation(
            x=critical_day, y=critical_value, xref="x", yref="y",
            text=f"{group_name}: Day {critical_day}<br>({critical_value:.1f}%/day)",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=20, ay=-40, # Adjust arrow position
            font=dict(size=10, color="#ffffff"),
            bgcolor="rgba(0,0,0,0.6)", # Dark semi-transparent background
            bordercolor=color, borderwidth=1, borderpad=2,
            opacity=0.8
        )

    fig.update_layout(
        title=title, xaxis_title='Days Since Fertilization', yaxis_title='Daily Change in Survival Rate (%/day)',
        legend_title_text=legend_title, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified', height=600
    )
    return fig


# --- calculate_critical_period_stats ---
@st.cache_data
def calculate_critical_period_stats(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Calculates statistics about the critical period (day of max decline)
    for each density category, based on individual dish data. Assumes density_category col is Utf8.
    """
    print("Calculating critical period statistics...")
    stats_df = df.clone()

    required_cols = ['dish_id', 'density_category', 'days_since_fertilization', 'survival_rate_change']
    if not all(col in stats_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in stats_df.columns]
        st.warning(f"Missing required columns for Critical Period Stats: {missing}. Skipping calculation.")
        return None

    # Ensure density_category is Utf8
    if 'density_category' not in stats_df.columns:
         st.warning("Column 'density_category' missing. Cannot calculate critical period stats.")
         return None
    elif stats_df['density_category'].dtype != pl.Utf8:
         stats_df = stats_df.with_columns(pl.col('density_category').cast(pl.Utf8))

    # Calculate critical day and max decline per dish
    critical_periods = stats_df.drop_nulls(
        subset=['dish_id', 'density_category', 'days_since_fertilization', 'survival_rate_change']
    ).sort(
        'dish_id', 'days_since_fertilization' # Sort by day within dish
    ).group_by(
        ['dish_id'], maintain_order=True # Group only by dish to find its specific min
    ).agg([
        pl.col('days_since_fertilization').gather(pl.col('survival_rate_change').arg_min()).first().alias('critical_day'),
        pl.min('survival_rate_change').alias('max_decline'),
        pl.first('density_category').alias('density_category') # Keep the density category associated with the dish (already Utf8)
    ])

    if critical_periods.height == 0:
        st.warning("Could not determine critical periods for any dishes (check for nulls or missing data in survival_rate_change).")
        return None

    critical_periods = critical_periods.with_columns([
        pl.col('critical_day').cast(pl.Float64, strict=False),
        pl.col('max_decline').cast(pl.Float64, strict=False)
    ]).drop_nulls(subset=['density_category', 'critical_day', 'max_decline']) # Ensure no nulls before final agg


    if critical_periods.height == 0:
        st.warning("No valid critical period data found after initial calculation.")
        return None

    # Aggregate stats by density category (group by Utf8)
    density_stats = critical_periods.group_by('density_category').agg([
        pl.mean('critical_day').alias('Mean Critical Day'),
        pl.std('critical_day').alias('Std Dev Critical Day'),
        pl.median('critical_day').alias('Median Critical Day'), # Added Median
        pl.mean('max_decline').alias('Mean Max Decline (%/day)'),
        pl.std('max_decline').alias('Std Dev Max Decline'),
        pl.median('max_decline').alias('Median Max Decline (%/day)'), # Added Median
        pl.len().alias('N (Dishes)')
    ]).sort('density_category')

    # Reorder columns for better readability
    final_cols = [
        'density_category', 'N (Dishes)',
        'Mean Critical Day', 'Median Critical Day', 'Std Dev Critical Day',
        'Mean Max Decline (%/day)', 'Median Max Decline (%/day)', 'Std Dev Max Decline'
        ]
    # Select only columns that exist in the dataframe
    final_cols = [col for col in final_cols if col in density_stats.columns]

    return density_stats.select(final_cols)


# --- display_stats_table ---
def display_stats_table(stats_df: Optional[pl.DataFrame], title: str):
    """Displays a generic statistics DataFrame in Streamlit."""
    st.subheader(title)
    if stats_df is not None and stats_df.height > 0:
        # Format numeric columns for better display
        stats_pd = stats_df.to_pandas() # Convert to pandas for styling
        float_cols = stats_pd.select_dtypes(include=np.number).columns
        # Exclude known integer count columns from float formatting
        count_cols = ['N (Dishes)', 'N_Beaker', 'N_Dish', 'N']
        format_dict = {col: "{:.2f}" for col in float_cols if col not in count_cols}

        # Handle specific formatting if needed (e.g., p-values)
        if 'P-Value' in stats_pd.columns:
             # Try converting P-Value to numeric for comparison, handling errors ('N/A', 'Error', 'NaN')
             stats_pd['P-Value_numeric'] = pd.to_numeric(stats_pd['P-Value'], errors='coerce')
             format_dict['P-Value'] = "{:.4f}" # Format original P-Value column
             try:
                 # Create styler from the original data (without temp numeric col)
                 styler = stats_pd.drop(columns=['P-Value_numeric']).style.format(format_dict, na_rep='N/A')
                 # Define a function to apply the style based on the numeric column
                 def highlight_significant(row):
                      p_val = row['P-Value_numeric'] # Access the temporary numeric column
                      # Default style is empty string
                      style = [''] * len(row)
                      if pd.notna(p_val) and p_val < 0.05:
                           # Apply green background to all columns in the row
                           style = ['background-color: #aaffaa'] * len(row)
                      return style

                 # Apply the styling function row-wise
                 styler = styler.apply(highlight_significant, axis=1,
                                       # Pass the numeric column data explicitly if needed, though accessing via name might work
                                       # subset=pd.IndexSlice[:, stats_pd.columns.drop('P-Value_numeric')] # Apply style to original columns
                                       )
                 st.dataframe(styler)
             except Exception as e:
                 print(f"Error applying style, showing raw dataframe: {e}")
                 st.dataframe(stats_pd.drop(columns=['P-Value_numeric'])) # Show without temp col

        else:
            # Display without p-value styling
            try:
                st.dataframe(stats_pd.style.format(format_dict, na_rep='N/A'))
            except Exception as e:
                print(f"Error formatting table, showing raw data: {e}")
                st.dataframe(stats_pd)

    elif stats_df is not None and stats_df.height == 0:
        st.info("No statistics calculated (perhaps insufficient data or groups).")
    else:
        st.warning("Statistics could not be calculated or retrieved (check warnings/errors).")


# --- Streamlit App Logic ---

CSV_FILE_PATH = "src/metazebrobot_dashboard/survivability_report.csv"

# Add a persistent state for caching control
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- load_processed_data --- MODIFIED ---
@st.cache_data # Still cache the data itself
def load_processed_data(csv_path: str) -> Optional[pl.DataFrame]:
    """
    Loads and standardizes the processed CSV file. Ensures key grouping columns
    (genotype, density_category, cross_id, housing, genotype_housing) are present and Utf8 type.
    """
    if not os.path.exists(csv_path):
        st.error(f"Error: Input file not found at '{csv_path}'. Please ensure the file exists.")
        return None
    try:
        with st.spinner(f"Loading processed data from {csv_path}..."):
            df = pl.read_csv(csv_path, try_parse_dates=True, ignore_errors=True)
            st.write("Standardizing data types...")

            # --- Stage 1: Ensure Base Columns & Cast to Utf8 ---
            base_ops = []
            # Ensure base categorical columns exist and cast to Utf8
            base_categorical_cols = ['genotype', 'in_beaker', 'density_category', 'cross_id', 'status', 'water_changed']
            for col in base_categorical_cols:
                if col in df.columns:
                    if df[col].dtype != pl.Utf8:
                        base_ops.append(pl.col(col).cast(pl.Utf8))
                        print(f"- Ensuring base column '{col}' is Utf8")
                else:
                    print(f"Warning: Base column '{col}' not found in CSV.")
            if base_ops:
                df = df.with_columns(base_ops)

            # --- Stage 2: Create Derived Columns (as Utf8) ---
            derived_ops = []
            # Create 'housing' from 'in_beaker'
            if 'housing' not in df.columns and 'in_beaker' in df.columns:
                 derived_ops.append(
                     pl.when(pl.col('in_beaker').str.to_lowercase() == 'yes')
                     .then(pl.lit('Beaker')).otherwise(pl.lit('Dish'))
                     .alias('housing') # Keep as Utf8
                 )
                 print("- Creating 'housing' column (as Utf8)")

            # Create 'genotype_housing' from 'genotype' and 'housing'
            # Apply housing creation first if needed
            if derived_ops:
                 df = df.with_columns(derived_ops)
                 derived_ops = [] # Reset for genotype_housing

            if 'genotype_housing' not in df.columns and 'genotype' in df.columns and 'housing' in df.columns:
                 derived_ops.append(
                    pl.concat_str(
                        [pl.col('genotype').fill_null("NA"), pl.lit(" ("), pl.col('housing').fill_null("Unknown"), pl.lit(")")],
                        separator=""
                    ).alias('genotype_housing') # Keep as Utf8
                 )
                 print("- Creating 'genotype_housing' column (as Utf8)")

            if derived_ops:
                 df = df.with_columns(derived_ops)


            # --- Stage 3: Cast Numeric Columns ---
            numeric_cast_ops = []
            # Float columns
            for col in ['survival_rate', 'volume_per_fish', 'fish_density', 'survival_rate_change', 'remaining', 'cumulative_deaths', 'max_decline']:
                 if col in df.columns:
                     # Check if not already float
                     if not isinstance(df[col].dtype, (pl.Float32, pl.Float64)):
                         numeric_cast_ops.append(pl.col(col).cast(pl.Float64, strict=False))
                         print(f"- Casting '{col}' to Float64")
                 else: print(f"Info: Numeric column '{col}' not found.")


            # Integer columns (handle potential errors more carefully)
            for col in ['days_since_fertilization', 'initial_count', 'fish_deaths', 'vol_water_total', 'vol_water_changed', 'critical_day']:
                 if col in df.columns:
                      # Check if not already integer
                     if not isinstance(df[col].dtype, (pl.Int32, pl.Int64)):
                         # Try casting to Float first to handle non-integer strings, then to Int
                         numeric_cast_ops.append(pl.col(col).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False))
                         print(f"- Casting '{col}' to Float64 -> Int64 (handling potential non-integers)")
                 else: print(f"Info: Integer column '{col}' not found.")


            if numeric_cast_ops:
                 df = df.with_columns(numeric_cast_ops)

            # --- Final Schema Check (Optional Debugging) ---
            # st.write("Final Schema after Loading/Standardization:")
            # st.write(df.schema)

            st.success(f"Successfully loaded and standardized {df.height} processed records.")
            return df

    except Exception as e:
        st.error(f"A critical error occurred during data loading or standardization: {e}")
        traceback.print_exc()
        return None


# --- Main App Flow ---
if st.sidebar.button("Reload Data", key="reload_button"):
    st.cache_data.clear() # Clear the cache
    st.session_state.data_loaded = False # Reset flag
    st.rerun()

# Load data only if not already loaded or if reload was pressed
if not st.session_state.data_loaded:
    df_processed = load_processed_data(CSV_FILE_PATH)
    if df_processed is not None:
        st.session_state.df_processed = df_processed # Store in session state
        st.session_state.data_loaded = True
    else:
        st.session_state.df_processed = None # Ensure it's None if loading failed
        st.session_state.data_loaded = False
else:
    # Retrieve data from session state if already loaded
    df_processed = st.session_state.get('df_processed', None)


st.sidebar.title("Plot & Analysis Controls")

if df_processed is not None and df_processed.height > 0:
    # Define plot options including the new ones
    plot_options = [
        "Survival by Genotype/Housing",
        "Survival by Density",
        "Survival by Cross ID", # New
        "Cumulative Deaths", # New
        "Density vs Survival",
        "Housing Comparison",
        "Remaining Fish by Housing/Density",
        "Daily Survival Rate Change (Geno/Housing)",
        "Rate Change by Density",
        "Distribution Plots", # New
        "Statistical Tests" # New
    ]
    plot_type = st.sidebar.radio("Select Analysis Type:", options=plot_options, key="plot_type_selector")

    # --- Prepare Filter Options ---
    # Derive options AFTER data loading and standardization (using Utf8 columns)
    def get_unique_options(df, col_name):
        """Safely gets unique, sorted, non-null string options from a column."""
        if col_name in df.columns:
            try:
                # Ensure Utf8, get unique, drop nulls, sort, convert to list
                return df.select(
                        pl.col(col_name).cast(pl.Utf8).drop_nulls()
                    ).unique().sort(col_name)[col_name].to_list()
            except Exception as e:
                print(f"Could not get unique options for {col_name}: {e}")
                return []
        return []

    # Get options using the helper function
    all_geno_housing_options = get_unique_options(df_processed, 'genotype_housing')
    all_density_options = get_unique_options(df_processed, 'density_category')
    all_cross_id_options = get_unique_options(df_processed, 'cross_id')
    all_genotype_options = get_unique_options(df_processed, 'genotype')
    all_housing_options = get_unique_options(df_processed, 'housing')


    # --- Sidebar Controls based on Plot Type ---
    selected_geno_housing_groups = None
    selected_density_groups = None
    selected_cross_id_groups = None
    selected_genotype_groups = None
    median_survival_group_col = None
    cumulative_deaths_group_col = None
    dist_metric_col = None
    dist_group_col = None
    dist_plot_type = None
    selected_dist_groups = None
    stat_test_type = None
    anova_metric = None

    st.sidebar.markdown("---") # Separator

    # --- Configure controls based on selected plot_type ---
    if plot_type == "Survival by Genotype/Housing":
        if all_geno_housing_options: selected_geno_housing_groups = st.sidebar.multiselect("Select Genotype/Housing Groups:", options=all_geno_housing_options, default=all_geno_housing_options, key="survival_geno_house_multi")
        else: st.sidebar.warning("Column 'genotype_housing' not found or has no options.")
        median_survival_group_col = 'genotype_housing' # Calculate median for this view

    elif plot_type == "Survival by Density":
        if all_density_options: selected_density_groups = st.sidebar.multiselect("Select Density Categories:", options=all_density_options, default=all_density_options, key="survival_density_multi")
        else: st.sidebar.warning("Column 'density_category' not found or has no options.")
        median_survival_group_col = 'density_category' # Calculate median for this view

    elif plot_type == "Survival by Cross ID":
        if all_cross_id_options: selected_cross_id_groups = st.sidebar.multiselect("Select Cross IDs:", options=all_cross_id_options, default=all_cross_id_options, key="survival_crossid_multi")
        else: st.sidebar.warning("Column 'cross_id' not found or has no options.")
        median_survival_group_col = 'cross_id' # Calculate median for this view

    elif plot_type == "Cumulative Deaths":
         group_options = []
         # Check if options *exist* for the columns
         if all_geno_housing_options: group_options.append('Genotype/Housing')
         if all_density_options: group_options.append('Density Category')
         if all_cross_id_options: group_options.append('Cross ID')

         if group_options:
             selected_grouping = st.sidebar.selectbox("Group By:", options=group_options, key="cum_death_group")
             if selected_grouping == 'Genotype/Housing':
                  cumulative_deaths_group_col = 'genotype_housing'
                  selected_geno_housing_groups = st.sidebar.multiselect("Select Groups:", options=all_geno_housing_options, default=all_geno_housing_options, key="cum_death_geno_house_multi")
             elif selected_grouping == 'Density Category':
                  cumulative_deaths_group_col = 'density_category'
                  selected_density_groups = st.sidebar.multiselect("Select Groups:", options=all_density_options, default=all_density_options, key="cum_death_density_multi")
             elif selected_grouping == 'Cross ID':
                  cumulative_deaths_group_col = 'cross_id'
                  selected_cross_id_groups = st.sidebar.multiselect("Select Groups:", options=all_cross_id_options, default=all_cross_id_options, key="cum_death_crossid_multi")
         else: st.sidebar.warning("No suitable grouping columns with options found (genotype_housing, density_category, cross_id).")

    elif plot_type == "Density vs Survival":
        use_vol_per_fish = st.sidebar.radio("X-Axis Metric:", (True, False), format_func=lambda x: "Volume per Fish" if x else "Fish Density (Fish/mL)", index=0, key="density_xaxis_radio")
        if all_genotype_options: selected_genotype_groups = st.sidebar.multiselect("Filter by Genotypes:", options=all_genotype_options, default=all_genotype_options, key="density_geno_multi")
        else: st.sidebar.warning("Column 'genotype' not found or has no options.")

    elif plot_type == "Daily Survival Rate Change (Geno/Housing)":
         if all_geno_housing_options: selected_geno_housing_groups = st.sidebar.multiselect("Select Genotype/Housing Groups:", options=all_geno_housing_options, default=all_geno_housing_options, key="rate_change_geno_house_multi")
         else: st.sidebar.warning("Column 'genotype_housing' not found or has no options.")

    elif plot_type == "Rate Change by Density":
        if all_density_options: selected_density_groups = st.sidebar.multiselect("Select Density Categories:", options=all_density_options, default=all_density_options, key="rate_change_density_multi")
        else: st.sidebar.warning("Column 'density_category' not found or has no options.")

    elif plot_type == "Distribution Plots":
         metric_options = {}
         # Check if base columns exist for calculation OR if metric col exists
         if 'survival_rate' in df_processed.columns: metric_options['Final Survival Rate'] = 'final_survival_rate'
         if 'survival_rate_change' in df_processed.columns: metric_options['Critical Day (Max Decline)'] = 'critical_day'
         # Add other potential metrics here by checking if the column exists

         if metric_options:
             selected_metric_label = st.sidebar.selectbox("Select Metric:", options=list(metric_options.keys()), key="dist_metric_select")
             dist_metric_col = metric_options[selected_metric_label]

             grouping_options = {}
             if all_genotype_options: grouping_options['Genotype'] = 'genotype'
             if all_density_options: grouping_options['Density Category'] = 'density_category'
             if all_housing_options: grouping_options['Housing'] = 'housing'
             if all_cross_id_options: grouping_options['Cross ID'] = 'cross_id'

             if grouping_options:
                 selected_group_label = st.sidebar.selectbox("Group By:", options=list(grouping_options.keys()), key="dist_group_select")
                 dist_group_col = grouping_options[selected_group_label]

                 # Get the appropriate list of groups based on selection
                 if dist_group_col == 'genotype': all_dist_groups = all_genotype_options
                 elif dist_group_col == 'density_category': all_dist_groups = all_density_options
                 elif dist_group_col == 'housing': all_dist_groups = all_housing_options
                 elif dist_group_col == 'cross_id': all_dist_groups = all_cross_id_options
                 else: all_dist_groups = []

                 if all_dist_groups:
                      selected_dist_groups = st.sidebar.multiselect(f"Select {selected_group_label} Groups:", options=all_dist_groups, default=all_dist_groups, key="dist_groups_multi")
                 else: st.sidebar.warning(f"No groups found or available for '{selected_group_label}'.")

                 dist_plot_type = st.sidebar.radio("Plot Type:", ('box', 'histogram'), key="dist_plot_type_radio")

             else: st.sidebar.warning("No suitable grouping columns with options found.")
         else: st.sidebar.warning("No suitable metrics found or calculable for distribution plots.")


    elif plot_type == "Statistical Tests":
         test_options = []
         # Check if necessary columns for each test exist
         if SCIPY_AVAILABLE:
             # Housing T-Test needs genotype, housing, survival_rate, days_since_fertilization
             if all(col in df_processed.columns for col in ['genotype', 'housing', 'survival_rate', 'days_since_fertilization']):
                 test_options.append("Housing T-Test (Final Survival by Genotype)")
             # Density ANOVA (Survival) needs density_category, survival_rate, days_since_fertilization
             if all(col in df_processed.columns for col in ['density_category', 'survival_rate', 'days_since_fertilization']):
                 test_options.append("Density ANOVA (Final Survival)")
             # Density ANOVA (Critical Day) needs density_category, survival_rate_change, days_since_fertilization
             if all(col in df_processed.columns for col in ['density_category', 'survival_rate_change', 'days_since_fertilization']):
                 test_options.append("Density ANOVA (Critical Day)")
         else:
             st.sidebar.warning("Statistical tests require `scipy`. Install with `pip install scipy`.", icon="⚠️")


         if test_options:
             stat_test_type = st.sidebar.selectbox("Select Test:", options=test_options, key="stat_test_select")
             if stat_test_type == "Density ANOVA (Critical Day)":
                 anova_metric = 'critical_day'
             elif stat_test_type == "Density ANOVA (Final Survival)":
                 anova_metric = 'final_survival_rate'
         else:
             if SCIPY_AVAILABLE:
                  st.sidebar.warning("Insufficient columns found in the data to perform available statistical tests.")


    # --- Main Panel: Display Plot / Stats ---
    st.subheader(f"Displaying: {plot_type}")
    fig = None
    stats_data = None
    test_results_df = None

    try:
        # Determine visible groups based on plot type for median survival calc
        visible_groups_for_median = None
        if plot_type == "Survival by Genotype/Housing":
            fig = plot_survival_curves_pl(df_processed, color_by='genotype_housing', visible_conditions_list=selected_geno_housing_groups)
            if median_survival_group_col: visible_groups_for_median = selected_geno_housing_groups
        elif plot_type == "Survival by Density":
            fig = plot_survival_curves_pl(df_processed, color_by='density_category', visible_conditions_list=selected_density_groups)
            if median_survival_group_col: visible_groups_for_median = selected_density_groups
        elif plot_type == "Survival by Cross ID":
            fig = plot_survival_curves_pl(df_processed, color_by='cross_id', visible_conditions_list=selected_cross_id_groups)
            if median_survival_group_col: visible_groups_for_median = selected_cross_id_groups
        elif plot_type == "Cumulative Deaths":
             if cumulative_deaths_group_col:
                 visible_groups = None
                 if cumulative_deaths_group_col == 'genotype_housing': visible_groups = selected_geno_housing_groups
                 elif cumulative_deaths_group_col == 'density_category': visible_groups = selected_density_groups
                 elif cumulative_deaths_group_col == 'cross_id': visible_groups = selected_cross_id_groups
                 fig = plot_cumulative_deaths_pl(df_processed, group_by=cumulative_deaths_group_col, visible_conditions_list=visible_groups)
             else: st.warning("Please select a grouping variable in the sidebar.")
        elif plot_type == "Housing Comparison":
            fig = plot_housing_comparison_pl(df_processed)
        elif plot_type == "Density vs Survival":
            fig = plot_density_vs_survival_pl(df_processed, use_volume_per_fish=use_vol_per_fish, visible_conditions_list=selected_genotype_groups, days=None) # Using latest day
        elif plot_type == "Remaining Fish by Housing/Density":
            fig = plot_remaining_by_group_pl(df_processed)
        elif plot_type == "Daily Survival Rate Change (Geno/Housing)":
            fig = plot_survival_rate_change_pl(df_processed, visible_conditions_list=selected_geno_housing_groups)
        elif plot_type == "Rate Change by Density":
            fig = plot_rate_change_by_density_pl(df_processed, visible_conditions_list=selected_density_groups)
            stats_data = calculate_critical_period_stats(df_processed) # Calculate critical period stats only for this plot
        elif plot_type == "Distribution Plots":
            if dist_metric_col and dist_group_col:
                 fig = plot_distributions_pl(df_processed, metric_col=dist_metric_col, group_col=dist_group_col, plot_type=dist_plot_type, visible_groups=selected_dist_groups)
            else: st.warning("Please select a metric and grouping variable in the sidebar.")
        elif plot_type == "Statistical Tests":
             if stat_test_type == "Housing T-Test (Final Survival by Genotype)":
                 test_results_df = perform_housing_ttest(df_processed)
                 display_stats_table(test_results_df, "Housing T-Test Results (Final Survival by Genotype)")
             elif stat_test_type == "Density ANOVA (Final Survival)" or stat_test_type == "Density ANOVA (Critical Day)":
                 test_results_df = perform_density_anova(df_processed, metric_col=anova_metric)
                 display_stats_table(test_results_df, f"Density ANOVA Results ({anova_metric.replace('_', ' ').title()})")
             elif stat_test_type:
                 st.info(f"Statistical test '{stat_test_type}' selected, but no display logic implemented yet.")
             else: st.info("Select a statistical test from the sidebar.")


        # Display plot if generated
        if fig: st.plotly_chart(fig, use_container_width=True)

        # Calculate and Display Median Survival for relevant plots
        if median_survival_group_col and plot_type in ["Survival by Genotype/Housing", "Survival by Density", "Survival by Cross ID"]:
            median_stats_data = calculate_approx_median_survival(df_processed, median_survival_group_col, visible_groups_for_median)
            display_stats_table(median_stats_data, "Approximate Median Survival Time")

        # Display other stats table if generated (e.g., critical period)
        if stats_data is not None and plot_type == "Rate Change by Density":
            display_stats_table(stats_data, "Critical Period Statistics by Density")

    except Exception as plot_error:
        st.error(f"An error occurred while generating the plot or stats for '{plot_type}': {plot_error}")
        st.exception(plot_error) # Show full traceback in the app for debugging

    st.markdown("---") # Separator before data table
    if st.checkbox("Show Processed Data Table"):
        # Display schema for debugging purposes
        st.write("DataFrame Schema:", df_processed.schema)
        st.dataframe(df_processed)

elif df_processed is not None and df_processed.height == 0:
    st.warning("Loaded data file is empty or processing failed. Please check the file format and content.")
else:
    # Error message shown during loading if df_processed is None
    st.error("Dashboard cannot be displayed due to data loading errors. Please check the console or sidebar for details and ensure the CSV file is valid.")

