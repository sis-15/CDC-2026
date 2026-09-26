import folium

def build_carto_map(df, selected_metric, carto_key, center_lat=35.7796, center_lon=-78.6382, zoom=8):
    """Constructs a Folium map with CARTO tiles and county marker dots."""
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    
    # CARTO Voyager tile URL
    carto_tile_url = f"https://basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?api_key={carto_key}"

    # Add tile layer
    folium.TileLayer(
        tiles=carto_tile_url,
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CARTO Voyager",
        max_zoom=19
    ).add_to(m)

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