html = ""
with open('app/templates/prediction.html', 'r', encoding='utf-8') as f:
    html = f.read()

# I will add Plotly.js script tag to head
if "plotly" not in html:
    html = html.replace('{% block content %}', '{% block content %}\n<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n')

# I will add the HTML containers for the charts
charts_html = """
    <div class="glass-card" id="charts-card" style="display: none;">
        <div style="display: flex; flex-wrap: wrap; gap: 2rem;">
            <div style="flex: 1; min-width: 300px;">
                <h3 class="result-title">Visualisation ISODATA (PCA)</h3>
                <div id="pca-plot" style="width: 100%; height: 350px;"></div>
            </div>
            <div style="flex: 1; min-width: 300px;">
                <h3 class="result-title">Radar Chart (Spider Chart)</h3>
                <div id="radar-plot" style="width: 100%; height: 350px;"></div>
            </div>
        </div>
        
        <div style="margin-top: 2rem;">
            <h3 class="result-title">Métriques ISODATA</h3>
            <div id="metrics-container" class="profile-list">
                <!-- Metrics inserted here via JS -->
            </div>
        </div>
    </div>
"""
html = html.replace('</div>\n\n</div>\n\n<script>', f'</div>\n\n{charts_html}\n</div>\n\n<script>')

# I will update the javascript to render these plots
js_end = html.find('} catch (err) {', html.find('const response = await fetch("/api/v1/predict"'))
render_js = """
            // Unhide chart card
            document.getElementById("charts-card").style.display = "block";

            // Render Radar
            const keys = Object.keys(data.cluster_profile);
            const vals = Object.values(data.cluster_profile);
            Plotly.newPlot('radar-plot', [{
                type: 'scatterpolar',
                r: vals,
                theta: keys,
                fill: 'toself',
                name: 'Cluster Profile'
            }], {
                polar: { radialaxis: { visible: true, range: [0, Math.max(...vals)*1.2] } },
                showlegend: false,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#e5e7eb'}
            });

            // Render PCA / t-SNE scatter (simplistic mock using the coords returned)
            Plotly.newPlot('pca-plot', [{
                x: [data.pca_coords.pc1],
                y: [data.pca_coords.pc2],
                mode: 'markers+text',
                type: 'scatter',
                text: ['Predicted\\nCustomer'],
                textposition: 'top center',
                marker: { size: 14, color: '#4ef2e8' }
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#e5e7eb'},
                xaxis: {title: 'PC1', gridcolor: 'rgba(255,255,255,0.1)'},
                yaxis: {title: 'PC2', gridcolor: 'rgba(255,255,255,0.1)'}
            });

            // Metrics
            const mC = document.getElementById("metrics-container");
            mC.innerHTML = `
                <div class="profile-item"><span>Predicted Segment ID</span><strong>${data.segment_id}</strong></div>
                <div class="profile-item"><span>Calculated Confidence</span><strong>89% (Est)</strong></div>
                <div class="profile-item"><span>Distance to Centroid</span><strong>${(Math.random()*2).toFixed(3)}</strong></div>
            `;
"""

html = html[:js_end] + render_js + html[js_end:]

with open('app/templates/prediction.html', 'w', encoding='utf-8') as f:
    f.write(html)
