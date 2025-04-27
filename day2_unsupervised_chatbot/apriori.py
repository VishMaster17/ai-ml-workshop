from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Your revised Indian market basket transactions
transactions = [
    ["bread", "butter", "jam"],
    ["bread", "butter", "chips"],
    ["bread", "jam", "soda"],
    ["chips", "soda"],
    ["bread", "butter", "jam", "soda"],
]

# Step 1: Convert list of transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("One-hot Encoded DataFrame:")
print(df.head())

# Step 2: Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Step 3: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("Association Rules:")
print(rules)

# Show the top rules
# 1️⃣  sort once
rules_sorted = rules.sort_values(by="lift", ascending=False)

# 2️⃣  keep only one copy of any exact (antecedent, consequent) pair
rules_sorted = (
    rules_sorted
        .drop_duplicates(subset=["antecedents", "consequents"])
        .reset_index(drop=True)          # nice tidy index
)

top20 = rules_sorted.head(20)

print("Top 20 rules without duplicates:")
print(top20[["antecedents", "consequents",
                    "support", "confidence", "lift"]])

