with open('app/templates/prediction.html', 'r') as f:
    html = f.read()

# Make it single column
html = html.replace('<div class="prediction-layout">', '<div class="prediction-layout-single" style="display: flex; flex-direction: column; gap: 2rem; max-width: 800px;">')

# Fix chip-container style
html = html.replace('<div class="chip-container" style="margin-bottom: 2rem;">', '<div class="chip-container" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 2rem;">')

# Write back
with open('app/templates/prediction.html', 'w') as f:
    f.write(html)
