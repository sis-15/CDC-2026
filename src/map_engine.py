import folium

def build_carto_map(df, selected_metric):
    """Constructs a Folium map dynamically centered on the filtered county dataset."""
    if df.empty:
        # Fallback center (US Center)
        center_lat, center_lon = 37.0902, -95.7129
        zoom = 4
    else:
        # Calculate dynamic centroid of filtered counties
        center_lat = df["Lat"].mean()
        center_lon = df["Lon"].mean()
        zoom = 6 if len(df) < 200 else 4

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")
    
    # Check if selected metric exists
    if selected_metric not in df.columns:
        df[selected_metric] = 1.0

    metric_min = df[selected_metric].min()
    metric_max = df[selected_metric].max()
    metric_range = metric_max - metric_min if metric_max != metric_min else 1

    for _, row in df.iterrows():
        val = row.get(selected_metric, 0)
        
        # Dynamic sizing between 8px and 26px
        normalized_size = (val - metric_min) / metric_range if metric_range > 0 else 0.5
        radius = 8 + (normalized_size * 18)
        
        color = "red" if val > df[selected_metric].median() else "blue"
        
        popup_text = f"""
        <b>County:</b> {row.get('County', 'N/A')}, {row.get('State', 'N/A')}<br>
        <b>FIPS:</b> {row.get('fips', 'N/A')}<br>
        <b>{selected_metric.replace('_', ' ')}:</b> {val:.3f}<br>
        <b>Denial Rate:</b> {row.get('HMDA_Denial_Rate', 0)*100:.1f}%<br>
        <b>Total Complaints:</b> {int(row.get('Total_Complaints', 0))}
        """
        
        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=popup_text
        ).add_to(m)
        
    return m