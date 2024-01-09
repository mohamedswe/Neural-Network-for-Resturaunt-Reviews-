# def clean_text(text):
#     # Keep only letters (remove numbers and punctuation)
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def clean_data():
#     df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

#     df['Review'] = df['Review'].apply(clean_text)

#     tokenizer = Tokenizer(num_words=10000) 

#     tokenizer.fit_on_texts(df['Review'])


#     # Saving the DataFrame to a CSV file
#     df.to_csv('Cleaned Review.csv', index=False, columns=['Review', 'Liked'])
