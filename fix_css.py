html = ""
with open('app/templates/prediction.html', 'r') as f:
    html = f.read()

# I want to make sure it looks visually pleasing and centered if needed.
html = html.replace('max-width: 800px;', 'max-width: 900px; margin: 0 auto;')

with open('app/templates/prediction.html', 'w') as f:
    f.write(html)
