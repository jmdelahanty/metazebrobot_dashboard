import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
import os
import traceback
import re
import pandas as pd # Keep for styling compatibility

try:
    import lifelines
    import matplotlib.pyplot as plt # Lifelines plotting often uses Matplotlib
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    # Warning placed inside app logic

# ---- NEW IMPORTS for GCS ----
from google.cloud import storage
from google.oauth2 import service_account
import io # To handle byte stream from GCS

# ---- Import for statistical tests ----
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Warning placed inside app logic

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
    Plots the cumulative number of deaths for each individual dish over time,
    grouped and colored by the specified column. Assumes group_by column is Utf8.
    """
    print(f"Plotting cumulative deaths per dish (group_by={group_by})...")
    plot_df = df.clone()

    group_col = group_by
    required_cols = ['dish_id', 'days_since_fertilization', 'cumulative_deaths', group_col]
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.error(f"Missing required columns for cumulative deaths plot: {missing}.")
        return go.Figure(layout=go.Layout(title="Cumulative Deaths Plot - Error: Columns Missing", height=600))

    # --- Setup Titles and Colors ---
    title_map = {
        'genotype_housing': 'Cumulative Deaths per Dish by Genotype and Housing',
        'density_category': 'Cumulative Deaths per Dish by Fish Density',
        'cross_id': 'Cumulative Deaths per Dish by Cross ID'
    }
    legend_title_map = {
        'genotype_housing': 'Genotype & Housing',
        'density_category': 'Fish Density',
        'cross_id': 'Cross ID'
    }
    title = title_map.get(group_col, f'Cumulative Deaths per Dish by {group_col}')
    legend_title = legend_title_map.get(group_col, group_col)
    COLOR_SEQUENCE = px.colors.qualitative.Plotly # Or any other sequence

    # --- Data Preparation ---
    # Ensure group column is Utf8 for consistent processing
    if plot_df[group_col].dtype != pl.Utf8:
        try:
            plot_df = plot_df.with_columns(pl.col(group_col).cast(pl.Utf8))
            st.warning(f"Plotting function had to cast '{group_col}' back to Utf8. Check loading logic if this persists.")
        except Exception as e:
             st.error(f"Failed to cast column '{group_col}' to Utf8 for plotting: {e}")
             return go.Figure(layout=go.Layout(title=title + f" - Error: Cannot Cast {group_col}", height=600))

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

    # Ensure numeric types for plotting and drop nulls required for lines
    plot_df_clean = plot_df.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('cumulative_deaths').cast(pl.Float64, strict=False) # Keep as float in case of non-integer deaths recorded
    ]).drop_nulls(subset=['dish_id', 'days_since_fertilization', 'cumulative_deaths', group_col])

    if plot_df_clean.height == 0:
        st.warning(f"No non-null data available for cumulative deaths plot ({group_col}) after filtering.")
        return go.Figure(layout=go.Layout(title=title + " - No Data", height=600))

    # --- Get unique groups (from filtered data) and assign colors ---
    unique_groups_agg = plot_df_clean.group_by(group_col).agg(pl.len()).sort(group_col)
    unique_cats = unique_groups_agg[group_col].drop_nulls().to_list()
    if not unique_cats:
        st.warning(f"No valid groups found in column '{group_col}' after filtering and cleaning.")
        return go.Figure(layout=go.Layout(title=title + " - No Groups Found", height=600))
    color_map = {cat: COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i, cat in enumerate(unique_cats)}

    # --- Create the plot ---
    fig = go.Figure()

    # --- Add individual dish lines ---
    # Get unique pairings of dish_id and its group from the cleaned data
    unique_dishes_data = plot_df_clean.group_by(['dish_id', group_col]).agg(
        pl.first('days_since_fertilization') # Just need one row per dish/group pair
    ).drop_nulls()

    added_legend_groups = set() # Keep track of groups added to legend

    for dish_info in unique_dishes_data.iter_rows(named=True):
        dish_id = dish_info['dish_id']
        group_name = dish_info[group_col] # This is Utf8
        group_color = color_map.get(group_name)

        if group_color: # Only plot if the group is valid and has a color
            # Filter the cleaned data for the current dish
            dish_data = plot_df_clean.filter(pl.col('dish_id') == dish_id).sort('days_since_fertilization')

            if dish_data.height > 0:
                # Determine if this is the first trace for this group to show legend
                show_legend_for_trace = False
                if group_name not in added_legend_groups:
                     show_legend_for_trace = True
                     added_legend_groups.add(group_name)

                fig.add_trace(go.Scatter(
                    x=dish_data['days_since_fertilization'],
                    y=dish_data['cumulative_deaths'],
                    mode='lines',
                    line=dict(width=1, color=group_color),
                    opacity=0.7, # Make individual lines slightly transparent
                    name=str(group_name) if show_legend_for_trace else f"_hidden_{dish_id}", # Show group name only once in legend
                    legendgroup=str(group_name), # Group lines by condition
                    showlegend=show_legend_for_trace, # Show legend only for the first trace of the group
                    hovertext=[f"Dish: {dish_id}<br>Day: {d}<br>Cum. Deaths: {c:.1f}<br>Group: {group_name}"
                               for d, c in zip(dish_data['days_since_fertilization'], dish_data['cumulative_deaths'])],
                    hoverinfo="text"
                ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Days Since Fertilization',
        yaxis_title='Cumulative Deaths per Dish',
        legend_title_text=legend_title,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest', # Use 'closest' for individual lines
        height=600
    )
    fig.update_yaxes(rangemode='tozero') # Ensure y-axis starts at 0
    return fig

def plot_survival_faceted_by_density_housing(
    df: pl.DataFrame,
    facet_col: str = 'density_category', # Column to create subplots for ('density_category' or 'housing')
    color_col: str = 'housing',        # Column to color lines by ('housing' or 'density_category')
    visible_densities: Optional[List[str]] = None,
    visible_housings: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots individual fish survival curves over time, faceted by one condition
    (density or housing) and colored by the other.
    """
    print(f"Plotting faceted survival curves (facet: {facet_col}, color: {color_col})...")
    plot_df = df.clone()

    # --- Input Validation ---
    valid_cols = ['density_category', 'housing']
    if facet_col not in valid_cols or color_col not in valid_cols or facet_col == color_col:
        st.error("Invalid facet_col or color_col specified. Choose different columns from 'density_category', 'housing'.")
        return go.Figure(layout=go.Layout(title="Faceted Plot Error: Invalid Column Choice"))

    required_cols = ['dish_id', 'days_since_fertilization', 'survival_rate', facet_col, color_col]
    if not all(col in plot_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in plot_df.columns]
        st.error(f"Missing required columns for faceted survival plot: {missing}.")
        return go.Figure(layout=go.Layout(title="Faceted Plot Error: Columns Missing", height=600))

    # --- Data Preparation ---
    # Ensure grouping/coloring columns are Utf8
    cast_ops = []
    for col in [facet_col, color_col]:
         if plot_df[col].dtype != pl.Utf8:
              cast_ops.append(pl.col(col).cast(pl.Utf8))
    if cast_ops:
        try:
            plot_df = plot_df.with_columns(cast_ops)
            st.warning(f"Faceted plot function had to cast columns to Utf8. Check loading logic if this persists.")
        except Exception as e:
             st.error(f"Failed to cast columns to Utf8 for faceted plotting: {e}")
             return go.Figure(layout=go.Layout(title="Faceted Plot Error: Cannot Cast Columns", height=600))

    # Filter based on visible selections
    if visible_densities is not None and 'density_category' in plot_df.columns:
        visible_densities_str = [str(d) for d in visible_densities]
        plot_df = plot_df.filter(pl.col('density_category').is_in(visible_densities_str))

    if visible_housings is not None and 'housing' in plot_df.columns:
        visible_housings_str = [str(h) for h in visible_housings]
        plot_df = plot_df.filter(pl.col('housing').is_in(visible_housings_str))

    if plot_df.height == 0:
        st.warning("No data matches the selected filters for the faceted plot.")
        return go.Figure(layout=go.Layout(title="Faceted Survival Plot - No Matching Data", height=600))

    # Ensure numeric types for plotting and drop nulls required for lines
    plot_df_clean = plot_df.with_columns([
        pl.col('days_since_fertilization').cast(pl.Int64, strict=False),
        pl.col('survival_rate').cast(pl.Float64, strict=False)
    ]).drop_nulls(subset=['dish_id', 'days_since_fertilization', 'survival_rate', facet_col, color_col])

    if plot_df_clean.height == 0:
        st.warning(f"No non-null data available for faceted plot after filtering.")
        return go.Figure(layout=go.Layout(title="Faceted Survival Plot - No Data", height=600))

    # --- Create Plot using Plotly Express ---
    title = f"Individual Survival Curves by {color_col.replace('_', ' ').title()} (Faceted by {facet_col.replace('_', ' ').title()})"
    labels = { # For better axis/legend titles
        'days_since_fertilization': 'Days Since Fertilization',
        'survival_rate': 'Survival Rate (%)',
        'density_category': 'Density Category',
        'housing': 'Housing Type'
    }

    try:
        # Use line_group='dish_id' to draw a separate line for each dish
        fig = px.line(
            plot_df_clean.to_pandas(), # px works best with pandas for now
            x='days_since_fertilization',
            y='survival_rate',
            color=color_col,      # Color lines by this column
            facet_col=facet_col,  # Create subplot columns for this column
            line_group='dish_id', # Crucial: draw line per dish
            labels=labels,
            title=title,
            category_orders={ # Optional: control facet order if needed
                 facet_col: sorted(plot_df_clean[facet_col].unique().to_list())
            },
            hover_data={ # Add extra info to hover tooltips
                 'dish_id': True,
                 facet_col: True,
                 color_col: True,
                 'days_since_fertilization': True,
                 'survival_rate': ':.1f' # Format survival rate
            }
        )

        # Customize layout
        fig.update_traces(opacity=0.75, line=dict(width=1)) # Make individual lines clearer
        fig.update_yaxes(range=[0, 100.5], title_text='Survival Rate (%)') # Ensure consistent Y axis
        fig.update_xaxes(title_text='Days Since Fertilization')
        # Adjust facet titles if needed (e.g., remove "density_category=")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(
            height=600, # Adjust height as needed
            legend_title_text=labels.get(color_col, color_col.replace('_', ' ').title()),
            hovermode='closest'
        )
        # fig.update_yaxes(matches=None) # Uncomment if you want independent Y-axes per facet

    except Exception as e:
        st.error(f"An error occurred while creating the faceted plot: {e}")
        traceback.print_exc()
        return go.Figure(layout=go.Layout(title="Faceted Plot Error: Plot Generation Failed", height=600))

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

@st.cache_data # Cache the prepared data
def prepare_lifelines_data(df: pl.DataFrame) -> Optional[pd.DataFrame]:
    """
    Transforms the time-series data into a format suitable for Lifelines analysis.
    Each row represents one dish.
    'duration' is the last day observed for the dish.
    'event_observed' is 1 if survival reached 0% by the last day, 0 otherwise (censored).
    """
    print("Preparing data for Lifelines...")
    required_cols = ['dish_id', 'days_since_fertilization', 'survival_rate']
    grouping_cols = ['genotype', 'housing', 'density_category', 'cross_id', 'genotype_housing'] # Add all potential grouping cols

    if not all(c in df.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        st.warning(f"Missing required columns for Lifelines data prep: {missing}. Cannot proceed.")
        return None

    # Find the last observation for each dish
    last_observation = df.sort('dish_id', 'days_since_fertilization').group_by('dish_id').agg(
        pl.last('days_since_fertilization').alias('duration'),
        pl.last('survival_rate').alias('final_survival_rate'),
        # Keep the first non-null value found for each grouping column for that dish
        *[pl.first(col).alias(col) for col in grouping_cols if col in df.columns]
    )

    # Define the event: Did the dish reach 0% survival?
    # Using a small threshold (<1%) to handle potential floating point inaccuracies
    lifelines_ready_df = last_observation.with_columns(
        pl.when(pl.col('final_survival_rate') < 1.0)
        .then(1) # Event occurred (all died)
        .otherwise(0) # Censored (some fish still alive at last check)
        .alias('event_observed')
    ).select(
        ['dish_id', 'duration', 'event_observed'] + # Core columns
        [col for col in grouping_cols if col in last_observation.columns] # Add existing grouping columns
    ).drop_nulls(subset=['dish_id', 'duration', 'event_observed']) # Ensure core columns are not null

    if lifelines_ready_df.height == 0:
        st.warning("No valid dish data found after preparing for Lifelines analysis.")
        return None

    print(f"Lifelines data prepared with {lifelines_ready_df.height} dishes.")
    # Lifelines often works more smoothly with Pandas DataFrames
    return lifelines_ready_df.to_pandas()

# --- Lifelines Plotting Function ---
def plot_cumulative_hazard_individuals(
    survival_df: pd.DataFrame, # Expects Pandas DataFrame from reconstruction
    visible_housings: Optional[List[str]] = None,
    visible_densities: Optional[List[str]] = None,
    show_ci: bool = False, # Option to show/hide confidence intervals
    ci_style: str = "shaded" # Options: "shaded", "lines", "both"
) -> Optional[go.Figure]:
    """
    Plots the cumulative hazard based on reconstructed individual fish data
    using Plotly for visualization with improved confidence interval display.

    Args:
        survival_df: Pandas DataFrame from reconstruct_individual_fish_data.
        visible_housings: List of housing types to include.
        visible_densities: List of density categories to include.
        show_ci: Whether to calculate and display 95% confidence intervals.
        ci_style: Style for confidence intervals ("shaded", "lines", or "both")

    Returns:
        A Plotly Figure object or None if plotting fails.
    """
    print("Plotting Lifelines Cumulative Hazard with Plotly (Individual Fish)...")

    if not LIFELINES_AVAILABLE:
        st.error("`lifelines` package not installed. Please install it (`pip install lifelines`) to use this feature.")
        return None

    required_cols = ['housing_type', 'density_category', 'time', 'event']
    if not all(c in survival_df.columns for c in required_cols):
        missing = [c for c in required_cols if c not in survival_df.columns]
        st.error(f"Reconstructed data is missing columns: {missing}")
        fig = go.Figure().add_annotation(text=f"Reconstructed data missing columns: {missing}", showarrow=False)
        return fig

    # Filter based on visible conditions BEFORE grouping
    plot_df = survival_df.copy()
    if visible_housings is not None:
        plot_df = plot_df[plot_df['housing_type'].isin(visible_housings)]
    if visible_densities is not None:
        plot_df = plot_df[plot_df['density_category'].isin(visible_densities)]

    if plot_df.empty:
        st.warning("No reconstructed fish data matches the selected filters.")
        fig = go.Figure().add_annotation(text="No data matches filters.", showarrow=False)
        return fig

    # --- Plotting with Plotly ---
    fig = go.Figure()
    naf = lifelines.NelsonAalenFitter()
    colors = px.colors.qualitative.Plotly
    groups = sorted(plot_df.groupby(['housing_type', 'density_category']))
    color_idx = 0

    for (housing, density), group_data in groups:
        if group_data.empty: continue
        label = f"{housing} - {density}"
        color = colors[color_idx % len(colors)]
        color_idx += 1

        try:
            naf.fit(
                durations=group_data['time'],
                event_observed=group_data['event'],
                label=label,
                alpha=0.05 # Corresponds to 95% CI
            )

            hazard_df_raw = naf.cumulative_hazard_
            if hazard_df_raw.empty: 
                st.warning(f"Cumulative hazard calculation returned empty for group '{label}'. Skipping.")
                continue

            estimate_col_name = hazard_df_raw.columns[0]
            hazard_curve = hazard_df_raw.reset_index()
            time_col_name = hazard_curve.columns[0]
            if time_col_name != 'time': 
                hazard_curve = hazard_curve.rename(columns={time_col_name: 'time'})
            
            # Add zero point if not present
            if 0 not in hazard_curve['time'].values: 
                hazard_curve = pd.concat([pd.DataFrame({'time': [0], estimate_col_name: [0.0]}), hazard_curve], 
                                        ignore_index=True).sort_values('time')

            # --- Extract and prepare confidence interval data ---
            conf_int_plotted = False
            if show_ci:
                try:
                    conf_int_raw = naf.confidence_interval_
                    if not conf_int_raw.empty:
                        # --- Dynamically find CI column names ---
                        ci_cols = conf_int_raw.columns.tolist()
                        lower_ci_col_name = next((col for col in ci_cols if 'lower' in col.lower()), None)
                        upper_ci_col_name = next((col for col in ci_cols if 'upper' in col.lower()), None)

                        if lower_ci_col_name and upper_ci_col_name:
                            conf_int = conf_int_raw.reset_index()
                            ci_time_col_name = conf_int.columns[0]
                            if ci_time_col_name != 'time': 
                                conf_int = conf_int.rename(columns={ci_time_col_name: 'time'})

                            # Add zero point to confidence intervals if not present
                            if 0 not in conf_int['time'].values:
                                zero_row = pd.DataFrame({'time': [0], 
                                                        lower_ci_col_name: [0.0], 
                                                        upper_ci_col_name: [0.0]})
                                conf_int = pd.concat([zero_row, conf_int], 
                                                    ignore_index=True).sort_values('time')

                            # Determine line width for CI bounds based on style
                            upper_line_width = 0
                            upper_line_dash = None
                            if ci_style in ["lines", "both"]:
                                upper_line_width = 1
                                upper_line_dash = "dot"

                            # Add upper CI bound
                            fig.add_trace(go.Scatter(
                                x=conf_int['time'], 
                                y=conf_int[upper_ci_col_name], 
                                mode='lines',
                                line=dict(
                                    width=upper_line_width, 
                                    color=color, 
                                    dash=upper_line_dash,
                                    shape='hv'  # Use step shape to match the main curve
                                ),
                                hovertemplate=(f"<b>{label}</b><br>" +
                                            "Time: %{x}<br>" +
                                            "Upper 95% CI: %{y:.3f}<extra></extra>") 
                                            if ci_style in ["lines", "both"] else None,
                                hoverinfo='text' if ci_style in ["lines", "both"] else 'skip',
                                showlegend=False, 
                                name=f'{label}_upper_ci'
                            ))
                            
                            # Set fill color with adjusted opacity based on style
                            fill_opacity = 0.35  # Increased from 0.2 for better visibility on dark background
                            if ci_style == "lines":
                                fill_opacity = 0  # No fill for lines-only style
                            
                            # Add lower CI bound
                            fig.add_trace(go.Scatter(
                                x=conf_int['time'], 
                                y=conf_int[lower_ci_col_name], 
                                mode='lines',
                                line=dict(
                                    width=upper_line_width, 
                                    color=color, 
                                    dash=upper_line_dash,
                                    shape='hv'  # Use step shape to match the main curve
                                ),
                                fillcolor=hex_to_rgba(color, fill_opacity),
                                fill='tonexty' if ci_style in ["shaded", "both"] else None,
                                hovertemplate=(f"<b>{label}</b><br>" +
                                            "Time: %{x}<br>" +
                                            "Lower 95% CI: %{y:.3f}<extra></extra>") 
                                            if ci_style in ["lines", "both"] else None,
                                hoverinfo='text' if ci_style in ["lines", "both"] else 'skip',
                                showlegend=False, 
                                name=f'{label}_lower_ci'
                            ))
                            
                            conf_int_plotted = True
                        else:
                            st.warning(f"Could not dynamically identify lower/upper CI columns in {ci_cols} for group '{label}'.")
                    else:
                        st.warning(f"Confidence interval calculation returned empty for group '{label}'.")

                except Exception as ci_e:
                    st.warning(f"Could not calculate or plot confidence interval for group '{label}': {ci_e}")

                # If CI plotting failed for any reason, ensure the flag reflects it
                if not conf_int_plotted:
                    st.info(f"Confidence interval will not be shown for group '{label}'.")

            # --- Add the main cumulative hazard curve (step plot) ---
            fig.add_trace(go.Scatter(
                x=hazard_curve['time'],
                y=hazard_curve[estimate_col_name],
                mode='lines',
                line=dict(shape='hv', color=color, width=2),
                name=label,
                legendgroup=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Time: %{x}<br>"
                    "Cumulative Hazard: %{y:.3f}<extra></extra>"
                )
            ))

        except Exception as e:
            st.warning(f"Could not fit or plot cumulative hazard for group '{label}' using Plotly: {e}", icon="⚠️")
            # st.exception(e) # Uncomment for full traceback if needed

    # --- Customize Layout ---
    fig.update_layout(
        title='Cumulative Hazard (Risk of Death) by Housing Type and Density',
        xaxis_title='Days Since Fertilization',
        yaxis_title='Cumulative Hazard H(t)',
        legend_title_text='Housing - Density',
        hovermode='x unified',
        height=700,
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', range=[0, None]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    if not fig.data:
        fig.add_annotation(text="No hazard curves could be generated.", showarrow=False)

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

@st.cache_data # Cache the result of this potentially slow reconstruction
def reconstruct_individual_fish_data(df_processed: pl.DataFrame) -> Optional[pd.DataFrame]:
    """
    Reconstructs individual fish survival data from dish-level time series data.

    Args:
        df_processed: The main Polars DataFrame after loading and initial processing.

    Returns:
        A Pandas DataFrame where each row is an individual fish's outcome
        (time of death or time of censoring), suitable for lifelines.
        Returns None if required columns are missing.
    """
    print("Reconstructing individual fish data for Lifelines...")
    required_cols = [
        'dish_id', 'days_since_fertilization', 'cumulative_deaths',
        'initial_count', 'remaining', 'genotype', 'housing', # Use 'housing' directly
        'density_category' # Add other potential grouping columns if needed
    ]
    if not all(c in df_processed.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df_processed.columns]
        st.warning(f"Missing required columns for individual fish reconstruction: {missing}. Cannot proceed.")
        return None

    # Ensure necessary columns have compatible types before converting to Pandas
    try:
        df_pd = df_processed.select(required_cols).with_columns([
            pl.col('days_since_fertilization').cast(pl.Int64),
            pl.col('cumulative_deaths').cast(pl.Float64).cast(pl.Int64), # Ensure integer deaths
            pl.col('initial_count').cast(pl.Int64),
            pl.col('remaining').cast(pl.Float64).cast(pl.Int64), # Ensure integer remaining
            pl.col('dish_id'), # Keep original type, usually int or str
            pl.col('genotype').cast(pl.Utf8),
            pl.col('housing').cast(pl.Utf8), # Ensure housing is Utf8
            pl.col('density_category').cast(pl.Utf8)
        ]).to_pandas()
    except Exception as e:
        st.error(f"Error converting Polars columns for reconstruction: {e}")
        return None


    survival_data = []
    # Group by dish_id to track individual dishes over time
    for dish_id, dish_data in df_pd.groupby('dish_id'):
        # Sort by days since fertilization
        dish_data = dish_data.sort_values('days_since_fertilization')

        if dish_data.empty: continue # Skip empty groups

        # Get metadata from the first row (should be consistent within dish)
        # Use .iloc[0] safely after checking it's not empty
        genotype = dish_data['genotype'].iloc[0]
        housing_type = dish_data['housing'].iloc[0] # Use 'housing' column
        density_category = dish_data['density_category'].iloc[0]

        # Get the initial count (use first non-null ideally, but first row is usually ok if sorted)
        initial_count = dish_data['initial_count'].iloc[0]
        if pd.isna(initial_count): continue # Skip if initial count is missing

        # Track the previously observed cumulative deaths
        prev_cum_deaths = 0
        last_day_observed = dish_data['days_since_fertilization'].max()

        # For each time point
        for _, row in dish_data.iterrows():
            days = row['days_since_fertilization']
            cum_deaths = row['cumulative_deaths']
            if pd.isna(cum_deaths): continue # Skip rows with missing cumulative deaths

            # Calculate new deaths in this period (handle potential float issues)
            new_deaths = max(0, int(round(cum_deaths - prev_cum_deaths))) # Ensure non-negative integer

            # Record each death
            if new_deaths > 0:
                for _ in range(new_deaths):
                    survival_data.append({
                        'dish_id': dish_id,
                        'genotype': genotype,
                        'housing_type': housing_type, # Use 'housing' column name
                        'density_category': density_category,
                        'time': days,
                        'event': 1  # 1 indicates death occurred
                    })

            # Record censored (survived) observations ONLY at the last time point for this dish
            if days == last_day_observed:
                remaining = row['remaining']
                if not pd.isna(remaining) and remaining > 0:
                     # Ensure remaining is integer
                     remaining_int = int(round(remaining))
                     for _ in range(remaining_int):
                        survival_data.append({
                            'dish_id': dish_id,
                            'genotype': genotype,
                            'housing_type': housing_type, # Use 'housing' column name
                            'density_category': density_category,
                            'time': days,
                            'event': 0  # 0 indicates censoring (still alive)
                        })

            # Update previously observed cumulative deaths
            prev_cum_deaths = cum_deaths

    if not survival_data:
        st.warning("No individual fish survival data could be reconstructed.")
        return None

    # Create a dataframe from the survival data
    survival_df = pd.DataFrame(survival_data)
    print(f"Individual fish data reconstructed with {len(survival_df)} entries.")
    return survival_df

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

# Add a persistent state for caching control
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- load_processed_data --- MODIFIED for GCS ---
@st.cache_data # Still cache the data itself
def load_processed_data() -> Optional[pl.DataFrame]:
    """
    Loads and standardizes the processed CSV file from Google Cloud Storage.
    Ensures key grouping columns (genotype, density_category, cross_id, housing,
    genotype_housing) are present and Utf8 type.
    Credentials, bucket name, and object name are loaded from Streamlit Secrets.
    """
    # --- Load Credentials and Config from Secrets ---
    try:
        # Check if secrets are loaded (essential for GCS access)
        if "gcs" not in st.secrets:
             st.error("GCS credentials not found in Streamlit Secrets. Please configure `secrets.toml` and add it to the deployment.")
             return None

        gcs_secrets = st.secrets["gcs"]
        bucket_name = gcs_secrets.get("bucket_name")
        object_name = gcs_secrets.get("object_name")

        if not bucket_name or not object_name:
            st.error("Missing 'bucket_name' or 'object_name' in GCS secrets.")
            return None

        # Create credentials object from secrets dictionary
        # Assumes the entire service account JSON content is under the [gcs] key
        creds = service_account.Credentials.from_service_account_info(gcs_secrets)
        print("Successfully loaded GCS credentials from secrets.")

    except Exception as e:
        st.error(f"Error loading GCS credentials from Streamlit Secrets: {e}")
        traceback.print_exc()
        return None

    # --- Download from GCS ---
    try:
        with st.spinner(f"Loading data from GCS: gs://{bucket_name}/{object_name}..."):
            storage_client = storage.Client(credentials=creds)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)

            if not blob.exists():
                st.error(f"Error: File not found in GCS at gs://{bucket_name}/{object_name}")
                return None

            # Download content as bytes
            data_bytes = blob.download_as_bytes()
            print(f"Downloaded {len(data_bytes)} bytes from GCS.")

            # --- Load Bytes into Polars ---
            # Use io.BytesIO to treat the bytes like a file
            df = pl.read_csv(io.BytesIO(data_bytes), try_parse_dates=True, ignore_errors=True)
            print("CSV data loaded into Polars DataFrame.")

            # --- Standardize Data Types (same logic as before) ---
            st.write("Standardizing data types...")

            # --- Stage 1: Ensure Base Columns & Cast to Utf8 ---
            base_ops = []
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

            # Apply housing creation first if needed
            if derived_ops:
                 df = df.with_columns(derived_ops)
                 derived_ops = [] # Reset for genotype_housing

            # Create 'genotype_housing' from 'genotype' and 'housing'
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

            st.success(f"Successfully loaded and standardized {df.height} records from GCS.")
            return df

    except storage.exceptions.NotFound:
         st.error(f"Error: File not found in GCS bucket '{bucket_name}' at path '{object_name}'. Check bucket/object names in secrets and ensure the file exists.")
         return None
    except service_account.exceptions.RefreshError as cred_error:
         st.error(f"Error authenticating with Google Cloud Storage using Service Account: {cred_error}. Check credentials in secrets.")
         traceback.print_exc()
         return None
    except Exception as e:
        st.error(f"A critical error occurred during GCS data loading or standardization: {e}")
        traceback.print_exc()
        return None


# --- Main App Flow ---
if st.sidebar.button("Reload Data", key="reload_button"):
    st.cache_data.clear() # Clear the cache
    st.session_state.data_loaded = False # Reset flag
    st.rerun()

# Load data only if not already loaded or if reload was pressed
if not st.session_state.data_loaded:
    # Call the modified function (no arguments needed now)
    df_processed = load_processed_data()
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
    # Define plot options including the new one
    plot_options = [
        "Survival by Genotype/Housing",
        "Survival by Density",
        "Survival by Cross ID",
        "Cumulative Deaths",
        "Density vs Survival",
        "Housing Comparison",
        "Remaining Fish by Housing/Density",
        "Daily Survival Rate Change (Geno/Housing)",
        "Rate Change by Density",
        "Distribution Plots",
        "Statistical Tests",
        "Faceted Survival (Density/Housing)",
        "Cumulative Hazard (Individual Fish)"
    ]
    plot_type = st.sidebar.radio("Select Analysis Type:", options=plot_options, key="plot_type_selector")

    # --- Prepare Filter Options ---
    def get_unique_options(df, col_name):
        """Safely gets unique, sorted, non-null string options from a column."""
        if col_name in df.columns:
            try:
                return df.select(pl.col(col_name).cast(pl.Utf8).drop_nulls()).unique().sort(col_name)[col_name].to_list()
            except Exception as e: print(f"Could not get unique options for {col_name}: {e}"); return []
        return []

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
    selected_visible_housings_indiv = None
    selected_visible_densities_indiv = None
    show_ci_indiv = False # Default to False

    st.sidebar.markdown("---") # Separator

    # --- Configure controls based on selected plot_type ---
    if plot_type == "Faceted Survival (Density/Housing)":
        st.sidebar.markdown("### Faceted Plot Options")
        facet_options = {'Density Category': 'density_category', 'Housing Type': 'housing'}
        facet_label = st.sidebar.selectbox("Facet By (Subplots):", options=list(facet_options.keys()), key="facet_select")
        facet_column = facet_options[facet_label]

        # Determine the coloring column based on the facet choice
        if facet_column == 'density_category':
            color_column = 'housing'
            color_label = 'Housing Type'
        else:
            color_column = 'density_category'
            color_label = 'Density Category'
        st.sidebar.write(f"Coloring By: {color_label}") # Display the automatic choice

        # Add multiselect filters for both density and housing
        if all_density_options:
             selected_visible_densities = st.sidebar.multiselect("Select Density Categories:", options=all_density_options, default=all_density_options, key="facet_density_multi")
        else: st.sidebar.warning("Column 'density_category' not found or has no options.")

        if all_housing_options:
             selected_visible_housings = st.sidebar.multiselect("Select Housing Types:", options=all_housing_options, default=all_housing_options, key="facet_housing_multi")
        else: st.sidebar.warning("Column 'housing' not found or has no options.")

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

    elif plot_type == "Cumulative Hazard (Individual Fish)":
        if not LIFELINES_AVAILABLE:
            st.sidebar.warning("`lifelines` package not installed. This feature is unavailable.", icon="⚠️")
        else:
            st.sidebar.markdown("### Individual Hazard Plot Options")
            # Filters for density and housing
            if all_density_options:
                selected_visible_densities_indiv = st.sidebar.multiselect(
                    "Select Density Categories:", 
                    options=all_density_options, 
                    default=all_density_options, 
                    key="indiv_haz_density_multi"
                )
            else:
                st.sidebar.warning("Column 'density_category' not found or has no options.")

            if all_housing_options:
                selected_visible_housings_indiv = st.sidebar.multiselect(
                    "Select Housing Types:", 
                    options=all_housing_options, 
                    default=all_housing_options, 
                    key="indiv_haz_housing_multi"
                )
            else:
                st.sidebar.warning("Column 'housing' not found or has no options.")

            # Checkbox for confidence intervals
            show_ci_indiv = st.sidebar.checkbox(
                "Show 95% Confidence Intervals", 
                value=True,  # Set to True by default
                key="indiv_haz_ci"
            )
            
            # Add CI style selector (only show when CIs are enabled)
            if show_ci_indiv:
                ci_style = st.sidebar.selectbox(
                    "Confidence Interval Style:", 
                    ["shaded", "lines", "both"], 
                    index=0,  # Default to "shaded"
                    key="indiv_haz_ci_style"
                )
            else:
                ci_style = "shaded"  # Default value, won't be used when CIs are hidden


    # --- Main Panel: Display Plot / Stats ---
    st.subheader(f"Displaying: {plot_type}")
    fig = None
    stats_data = None
    test_results_df = None
    matplotlib_fig = None

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
        elif plot_type == "Faceted Survival (Density/Housing)":
              if facet_column and color_column:
                   fig = plot_survival_faceted_by_density_housing(
                       df_processed,
                       facet_col=facet_column,
                       color_col=color_column,
                       visible_densities=selected_visible_densities,
                       visible_housings=selected_visible_housings
                   )
              else:
                  st.warning("Please select faceting options in the sidebar.")
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
        elif plot_type == "Cumulative Hazard (Individual Fish)":
            if LIFELINES_AVAILABLE:
                # Reconstruct the individual fish data
                individual_survival_df = reconstruct_individual_fish_data(df_processed)
                
                if individual_survival_df is not None and not individual_survival_df.empty:
                    # Generate the Plotly figure with corrected parameter names
                    fig = plot_cumulative_hazard_individuals(
                        survival_df=individual_survival_df,
                        visible_housings=selected_visible_housings_indiv,
                        visible_densities=selected_visible_densities_indiv,
                        show_ci=show_ci_indiv,  # Pass the checkbox value
                        ci_style=ci_style  # Pass the selected style
                    )
                    
                    if fig is None:
                        st.warning("Plot generation failed.")
                else:
                    st.warning("Could not reconstruct individual fish data suitable for Lifelines analysis.")
            else:
                st.error("`lifelines` package is required for this plot but not installed.")
                fig = None  # Ensure fig is None if lifelines not available


        # Display plot if generated
        if fig: # Plotly figure
            st.plotly_chart(fig, use_container_width=True)

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
    st.error("Dashboard cannot be displayed due to data loading errors. Please check the console or sidebar for details and ensure the CSV file is valid and accessible via GCS.")

