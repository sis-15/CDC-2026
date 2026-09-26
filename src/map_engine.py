import folium

def build_carto_map(df, selected_metric, carto_key=None, center_lat=35.7796, center_lon=-78.6382, zoom=8):
    """
    Constructs a Folium map using standard OpenStreetMap tiles and county marker dots.
    (carto_key is retained as an optional argument for backwards compatibility with app.py).
    """
    # Create base map with native OpenStreetMap tiles
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=zoom, 
        tiles="OpenStreetMap"
    )

    # Add county marker circles
    for _, row in df.iterrows():
        radius = max(8, row.get("Total_Complaints", 100) / 50)
        color = "red" if row[selected_metric] > df[selected_metric].median() else "blue"
        
        popup_text = f"""
        <b>County:</b> {row.get('County', 'N/A')}, {row.get('State', 'NC')}<br>
        <b>{selected_metric}:</b> {row[selected_metric]}<br>
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