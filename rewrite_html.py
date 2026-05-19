with open('app/templates/prediction.html', 'r') as f:
    text = f.read()

# Replace the field grid with a chip container
old_html = """            <div class="field-grid">
                {% if feature_columns %}
                    {% for col in feature_columns %}
                        {% if col.upper() != "CUST_ID" and col.upper() != "ID" %}
                        <label>
                            {{ col.replace('_', ' ').title() }}
                            <input type="number" step="any" name="{{ col }}" placeholder="0.0" required>
                        </label>
                        {% endif %}
                    {% endfor %}
                {% else %}
                    <p class="text-muted" style="grid-column: 1 / -1;">No dataset available. Please upload a dataset or train a model to load features.</p>
                {% endif %}
            </div>"""

new_html = """            <div class="chip-container" style="margin-bottom: 2rem;">
                {% if feature_columns %}
                    {% for col in feature_columns %}
                        {% if col.upper() != "CUST_ID" and col.upper() != "ID" %}
                        <span class="chip">{{ col }}</span>
                        {% endif %}
                    {% endfor %}
                {% else %}
                    <p class="text-muted" style="grid-column: 1 / -1;">No dataset available. Please upload a dataset to load features.</p>
                {% endif %}
            </div>"""

text = text.replace(old_html, new_html)

# Update Javascript
old_js = """        const payload = {};
        const missing = [];
        const inputs = form.querySelectorAll("input[name]");

        inputs.forEach((input) => {
            const name = input.name;
            const value = toNumber(input.value.trim());
            if (value === null) {
                if (input.required) {
                    missing.push(name);
                }
                return;
            }
            payload[name] = value;
        });

        if (missing.length) {
            error.textContent = `Missing required fields: ${missing.join(", ")}`;
            result.innerHTML = "<p class=\"text-muted\">Fill in the missing fields to run a prediction.</p>";
            return;
        }"""

new_js = """        let payload = {};
        try {
            // Fetch a random sample from the dataset
            const sampleRes = await fetch("/api/dataset/sample");
            const sampleData = await sampleRes.json();
            if (sampleData.error) {
                throw new Error(sampleData.error);
            }
            payload = sampleData.sample;
            // Handle NaNs/nulls
            for (let k in payload) {
                if (payload[k] === null || Number.isNaN(Number(payload[k]))) {
                    payload[k] = 0;
                }
            }
        } catch (err) {
            error.textContent = err.message || "Failed to fetch sample record.";
            return;
        }"""

text = text.replace(old_js, new_js)

# Update button text
text = text.replace("Predict Cluster", "Predict on Sample Dataset Record")

with open('app/templates/prediction.html', 'w') as f:
    f.write(text)
