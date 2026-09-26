import folium

def build_carto_map(df, selected_metric, center_lat=35.7796, center_lon=-78.6382, zoom=8):
    """Constructs a Folium map with dynamically scaled marker circles based on the selected metric."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")
    
    # Check if the requested metric column exists in the DataFrame
    if selected_metric not in df.columns:
        # Create a fallback column if missing
        df[selected_metric] = 1.0

    metric_min = df[selected_metric].min()
    metric_max = df[selected_metric].max()
    metric_range = metric_max - metric_min if metric_max != metric_min else 1

    for _, row in df.iterrows():
        val = row.get(selected_metric, 0)
        
        # Scale radius dynamically between 8px and 28px
        normalized_size = (val - metric_min) / metric_range if metric_range > 0 else 0.5
        radius = 8 + (normalized_size * 20)
        
        color = "red" if val > df[selected_metric].median() else "blue"
        
        popup_text = f"""
        <b>County:</b> {row.get('County', 'N/A')}, {row.get('State', 'NC')}<br>
        <b>{selected_metric.replace('_', ' ')}:</b> {val:.3f}<br>
        <b>Denial Rate:</b> {row.get('HMDA_Denial_Rate', 0)*100:.1f}%<br>
        <b>Total Complaints:</b> {row.get('Total_Complaints', 0)}
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