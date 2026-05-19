with open("app/routes/prediction.py", "r") as f:
    text = f.read()

# I want to fillna(0) before taking to_dict to guarantee clean JSON serialization
old = """    df = pd.read_csv(path)
    sample = df.sample(1).iloc[0].to_dict()
    return {"sample": sample}"""
new = """    df = pd.read_csv(path)
    # Ensure NaN values are replaced with 0 or None so JSON serialization does not break
    sample = df.fillna(0).sample(1).iloc[0].to_dict()
    return {"sample": sample}"""
text = text.replace(old, new)

with open("app/routes/prediction.py", "w") as f:
    f.write(text)
