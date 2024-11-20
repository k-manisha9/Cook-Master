from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = 'C:/Users/ADMIN/Downloads/merged_cleaned_dataset.csv'  # Path to your dataset
dataset = pd.read_csv(file_path)
print(dataset.head())  m
# Preprocess the ingredients by tokenizing and cleaning
def preprocess_ingredients(ingredients):
    return ingredients.lower().replace(',', ' ')

# Apply preprocessing to the ingredients column in the dataset
dataset['Cleaned_Ingredients'] = dataset['Ingredients'].apply(preprocess_ingredients)

# Create the Flask app
app = Flask(_name_)

# Vectorize the ingredients using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset['Cleaned_Ingredients'])

# Function to suggest recipes based on TF-IDF cosine similarity
def suggest_recipes_tfidf(user_ingredients, top_n=5):
    # Preprocess user input
    user_ingredients_cleaned = preprocess_ingredients(user_ingredients)

    # Transform user input into TF-IDF vector
    user_tfidf = vectorizer.transform([user_ingredients_cleaned])

    # Calculate cosine similarity between user input and all recipe ingredients
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Filter out recipes with very low similarity (threshold)
    similarity_threshold = 0.1
    valid_indices = cosine_similarities > similarity_threshold

    # Sort the valid recipes by similarity in descending order
    top_indices = cosine_similarities.argsort()[valid_indices][-top_n:][::-1]

    # Avoid returning duplicate recipes by dropping duplicates
    top_recipes = dataset.iloc[top_indices].drop_duplicates(subset='Recipe Name')

    # Include similarity scores in the result
    top_recipes['Similarity'] = cosine_similarities[top_indices]

    return top_recipes[['Recipe Name', 'Instructions', 'Similarity']]

# Route for the home page
@app.route('/')
def index():
    return '''
<html>
<head>
    <title>Recipe Suggestion Chatbot</title>
    <style>
        body {
            background-image: url('https://img.freepik.com/free-photo/copy-space-italian-food-ingredients_23-2148551732.jpg?w=900&t=st=1729338329~exp=1729338929~hmac=80c8318a901b9d79994663d5186f536ff25cda46d87f7b0361f1263d0090c7a6');  /* Replaced with the new image URL */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Arial', sans-serif;
            color: #fff;
            font-weight: 300;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            background-color: rgba(255, 255, 255, 0.2); /* Slight transparency */
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            width: 50%;
            max-width: 800px;
        }

        h1 {
            font-size: 54px;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px #000;
            font-weight: bold;
            letter-spacing: 2px;
            color:#66bb6a;;
        }

        form {
            margin-top: 20px;
            font-size: 18px;
            color: white;
            font-weight: 300;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            margin-bottom: 20px;
            background-color: rgba(255, 255, 255, 0); /* Fully transparent */
            color: black;  /* Updated to black */
            border-bottom: 2px solid white;
            transition: box-shadow 0.3s ease-in-out, border-color 0.3s ease-in-out;
        }

        input[type="text"]:hover {
            border-bottom: 2px solid #ffdd57;
        }

        input[type="submit"] {
            background-color: #66bb6a;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 300;
            transition: background-color 0.4s, transform 0.3s ease-in-out;
            width: 50%;
        }

        input[type="submit"]:hover {
            transform: scale(1.1);
        }
        label{
        color:#66bb6a;
        }

        .container form {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Recipe Suggestion Chatbot</h1>
        <form action="/suggest_recipes" method="post">
            <label for="ingredients">Enter ingredients (comma-separated):</label><br><br>
            <input type="text" id="ingredients" name="ingredients" placeholder="E.g. chicken, garlic, tomatoes"><br>
            <input type="submit" value="Get Recipes">
        </form>
    </div>
</body>
</html>
    '''

# API route to get recipe suggestions based on ingredients
@app.route('/suggest_recipes', methods=['POST'])
def suggest_recipes():
    user_ingredients = request.form.get('ingredients')

    # Get the top matching recipes by TF-IDF cosine similarity
    top_n = 5  # Show top 5 matches
    suggested_recipes = suggest_recipes_tfidf(user_ingredients, top_n=top_n)

    if suggested_recipes is None or suggested_recipes.empty:
        return '''
        <html>
        <head>
            <title>No Recipes Found</title>
            <style>
                body {
                    background-image: url('https://img.freepik.com/free-photo/copy-space-italian-food-ingredients_23-2148551732.jpg?w=900&t=st=1729338329~exp=1729338929~hmac=80c8318a901b9d79994663d5186f536ff25cda46d87f7b0361f1263d0090c7a6');  /* Background image */
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    font-family: 'Arial', sans-serif;
                    color: #fff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background-color: rgba(255, 255, 255, 0.2); /* Transparency */
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                }
                h2 {
                    font-size: 32px;
                    color: #ffdd57;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>No recipes found. Please try different ingredients.</h2>
            </div>
        </body>
        </html>
        '''

    # Return recipes as HTML with the background and transparent container
    result = '''
    <html>
    <head>
        <title>Recipe Suggestions</title>
        <style>
            body {
                background-image: url('https://img.freepik.com/free-photo/copy-space-italian-food-ingredients_23-2148551732.jpg?w=900&t=st=1729338329~exp=1729338929~hmac=80c8318a901b9d79994663d5186f536ff25cda46d87f7b0361f1263d0090c7a6');  /* Background image */
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                font-family: 'Arial', sans-serif;
                color: #fff;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: rgba(255, 255, 255, 0.2); /* Transparency */
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                width: 60%;
                max-width: 900px;
                overflow-y: auto;  /* Ensure scrolling if content overflows */
                max-height: 80vh;  /* Prevent cutting off by limiting height */
            }
            h2 {
                color: green;  /* Updated heading color to green */
            }
            h3 {
                color: green;  /* Subheading color */
            }
            p{
            color: black;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Recipe Suggestions</h2>
    '''

    for _, recipe in suggested_recipes.iterrows():
        result += f"<h3>{recipe['Recipe Name']}</h3>"
        result += f"<p>{recipe['Instructions']}</p>"
        result += f"<p>Similarity: {recipe['Similarity']:.2f}</p>"

    result += '''
        </div>
    </body>
    </html>
    '''

    return result

# Run the app
if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5004)
