import re

with open('app/templates/prediction.html', 'r', encoding='utf-8') as f:
    html = f.read()

# I want to add metadata fetching so I can show actual global ISODATA metrics

new_js = """
        try {
            const [response, metaRes] = await Promise.all([
                fetch("/api/v1/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                }),
                fetch("/api/v1/metadata")
            ]);
            const data = await response.json();
            const meta = await metaRes.json();
            
            if (!response.ok) {
                throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail) || "Prediction failed.");
            }
"""

# Replace the single fetch with Promise.all
# Locate the fetch('/api/v1/predict')
start = html.find('const response = await fetch("/api/v1/predict"')
end = html.find('if (!response.ok) {')
end = html.find('}', end) + 1 # include the closing brace

# We need to replace the old block
html = html[:start] + """
            const [response, metaRes] = await Promise.all([
                fetch("/api/v1/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                }),
                fetch("/api/v1/metadata")
            ]);
            const data = await response.json();
            const meta = await metaRes.json();
            
            if (!response.ok) {
                throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail) || "Prediction failed.");
            }
""" + html[end:]

# Now update the HTML metrics container text
mC_js = """// Metrics
            const mC = document.getElementById("metrics-container");
            const metrics = meta.metrics || {};
            mC.innerHTML = `
                <div class="profile-item"><span>Predicted Segment ID</span><strong>${data.segment_id}</strong></div>
                <div class="profile-item"><span>Total Final Clusters</span><strong>${meta.n_clusters_final || data.n_clusters}</strong></div>
                <div class="profile-item"><span>Silhouette Score</span><strong>${(metrics.silhouette_score || 0).toFixed(4)}</strong></div>
                <div class="profile-item"><span>Davies-Bouldin</span><strong>${(metrics.davies_bouldin_score || 0).toFixed(4)}</strong></div>
                <div class="profile-item"><span>Calinski-Harabasz</span><strong>${(metrics.calinski_harabasz_score || 0).toFixed(4)}</strong></div>
                <div class="profile-item"><span>Distance from Centroid</span><strong>Est.</strong></div>
            `;
"""

# replace the old metrics js
mC_start = html.find('// Metrics')
mC_end = html.find('`;', mC_start) + 2

html = html[:mC_start] + mC_js + html[mC_end:]

with open('app/templates/prediction.html', 'w', encoding='utf-8') as f:
    f.write(html)

