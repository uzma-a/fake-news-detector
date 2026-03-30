import pandas as pd
from datasets import load_dataset

# GonzaloA load karo
print("Loading GonzaloA dataset...")
ds = load_dataset("GonzaloA/fake_news")
df_gonzalo = pd.DataFrame(ds['train'])
df_gonzalo['content'] = (
    df_gonzalo['title'].fillna('') + " " + 
    df_gonzalo['text'].fillna('')
)
df_gonzalo = df_gonzalo[['content', 'label']]
print(f"✅ GonzaloA: {len(df_gonzalo)} articles")

# Indian data load karo
print("Loading Indian data...")
df_indian = pd.read_csv("indian_news.csv")[['content', 'label']]
print(f"✅ Indian RSS: {len(df_indian)} articles")

# Indian data ko 10x weight do
# (kyunki sirf 232 hai vs 24000 GonzaloA)
df_indian_weighted = pd.concat([df_indian] * 10)
print(f"✅ Indian weighted: {len(df_indian_weighted)} articles")

# Combine
df_final = pd.concat([df_gonzalo, df_indian_weighted])
df_final = df_final.dropna()
df_final = df_final[df_final['content'].str.strip() != '']
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n=== Final Combined Dataset ===")
print(f"Total:  {len(df_final)}")
print(f"Fake:   {(df_final['label']==0).sum()}")
print(f"Real:   {(df_final['label']==1).sum()}")

df_final.to_csv("combined_dataset.csv", index=False)
print("✅ Saved to combined_dataset.csv!")