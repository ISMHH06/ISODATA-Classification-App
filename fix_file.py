with open("app/routes/prediction.py", "r") as f:
    text = f.read()

text = text.replace("context=context)import", "context=context)\n\nimport")

with open("app/routes/prediction.py", "w") as f:
    f.write(text)
