import re

with open('app/templates/prediction.html', 'r') as f:
    text = f.read()

# Replace the JS logic inside form logic
js_start = text.find('form.addEventListener("submit", async (event) => {') + 51
js_end = text.find('try {', js_start)

new_js = """
        event.preventDefault();
        error.textContent = "";
        result.innerHTML = "";

        let payload = {};
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
        }

        """

text = text[:js_start] + new_js + text[js_end:]

# Also fix error message [object Object] printing handling
text = text.replace('throw new Error(data.detail || "Prediction failed.");', 'throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail) || "Prediction failed.");')

with open('app/templates/prediction.html', 'w') as f:
    f.write(text)
