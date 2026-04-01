import pandas as pd

# Load data
data = pd.read_csv('../data.csv')

# Remove duplicate transactions
cleaned_data = data.drop_duplicates(subset=['Member_number', 'Date', 'itemDescription'])

# Save the cleaned data
cleaned_data.to_csv('cleaned_data.csv', index=False)

from mlxtend.frequent_patterns import apriori, association_rules

# Load cleaned data
cleaned_data = pd.read_csv('cleaned_data.csv')

# Transform data into the required format
basket = (cleaned_data
          .groupby(['Member_number', 'Date'])['itemDescription']
          .apply(lambda x: list(x))
          .reset_index(drop=True)
          .tolist())

# Convert basket list to dataframe for hot encoding (one-hot encoding)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

# Sort by lift and get top 5 rules
top_rules = rules.sort_values(by="lift", ascending=False).head(5)

# Save to CSV
top_rules.to_csv('association_rules.csv', index=False)

import pandas as pd

# Load association rules
rules = pd.read_csv('association_rules.csv')

# Analyze the rules and write the analysis
analysis = """
Association Rules Analysis
==========================

This analysis highlights the top 5 association rules discovered based on the metric of lift.

1. Rule: {} => {}
   - Support: {:.2f}
   - Confidence: {:.2f}
   - Lift: {:.2f}

2. Rule: {} => {}
   - Support: {:.2f}
   - Confidence: {:.2f}
   - Lift: {:.2f}

3. Rule: {} => {}
   - Support: {:.2f}
   - Confidence: {:.2f}
   - Lift: {:.2f}

4. Rule: {} => {}
   - Support: {:.2f}
   - Confidence: {:.2f}
   - Lift: {:.2f}

5. Rule: {} => {}
   - Support: {:.2f}
   - Confidence: {:.2f}
   - Lift: {:.2f}
""".format(
    rules.iloc[0]['antecedents'], rules.iloc[0]['consequents'], rules.iloc[0]['support'], rules.iloc[0]['confidence'], rules.iloc[0]['lift'],
    rules.iloc[1]['antecedents'], rules.iloc[1]['consequents'], rules.iloc[1]['support'], rules.iloc[1]['confidence'], rules.iloc[1]['lift'],
    rules.iloc[2]['antecedents'], rules.iloc[2]['consequents'], rules.iloc[2]['support'], rules.iloc[2]['confidence'], rules.iloc[2]['lift'],
    rules.iloc[3]['antecedents'], rules.iloc[3]['consequents'], rules.iloc[3]['support'], rules.iloc[3]['confidence'], rules.iloc[3]['lift'],
    rules.iloc[4]['antecedents'], rules.iloc[4]['consequents'], rules.iloc[4]['support'], rules.iloc[4]['confidence'], rules.iloc[4]['lift'],
)

# Save analysis to PDF
from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Association Rules Analysis', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Create PDF
pdf = PDF()
pdf.add_page()
pdf.chapter_title('Analysis of Association Rules')
pdf.chapter_body(analysis)
pdf.output('rules_analysis.pdf')