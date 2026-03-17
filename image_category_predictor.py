import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Upgraded from CountVectorizer for better term weighting
from sklearn.ensemble import RandomForestClassifier  # Upgraded from MultinomialNB for better accuracy
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # Added for tuning and validation

# Expanded dataset: more examples per category for better model training (50 entries, balanced)
descriptions = [
    "sunset over mountains landscape", "city skyscrapers at dusk skyline", "eagle soaring in sky wildlife",
    "busy downtown market urban scene", "ocean waves crashing on shore water", "forest with deer and trees nature",
    "desert landscape with cactus nature", "urban nightlife lights and crowds urban", "tiger in jungle stalking wildlife",
    "city traffic gridlock urban jam", "ancient stone temple ruins architecture", "polar bear on ice floe wildlife",
    "lake with swans swimming water", "modern glass skyscraper building architecture", "river with salmon jumping water",
    "gothic cathedral spire architecture", "calm ocean bay water view", "high-rise under construction urban",
    "bird nest high up in tree wildlife", "waterfall mist in rainforest water", "bridge over river architecture",
    "lion in savanna grassland wildlife", "harbor with yachts docked water", "castle with moat architecture",
    "subway rush hour urban crowd", "volcanic eruption lava flow nature", "wolf pack running in snow wildlife",
    "art gallery exhibit interior architecture", "tropical beach with palms nature", "mountain peak climb nature",
    "rainforest canopy view nature", "neon-lit alley urban night", "owl on branch at night wildlife",
    "pedestrian crosswalk urban", "coral reef underwater scene water", "meadow with wildflowers nature",
    "sandy dunes blowing wind nature", "downtown billboard ads urban", "fox in snow forest wildlife",
    "historic coliseum ruins architecture", "pond with lily pads water", "skyline with drones urban",
    "wild horse herd running wildlife", "seaside cliff with waves water", "medieval church tower architecture",
    "bustling airport terminal urban", "geyser erupting hot spring nature", "monkey swinging in trees wildlife",
    "museum hall with statues architecture", "island lagoon blue water", "canyon with river below nature"
]
categories = [
    "nature", "urban", "wildlife", "urban", "water", "nature",
    "nature", "urban", "wildlife", "urban", "architecture", "wildlife",
    "water", "architecture", "water", "architecture", "water", "urban",
    "wildlife", "water", "architecture", "wildlife", "water", "architecture",
    "urban", "nature", "wildlife", "architecture", "nature", "nature",
    "nature", "urban", "wildlife", "urban", "water", "nature",
    "nature", "urban", "wildlife", "architecture", "water", "urban",
    "wildlife", "water", "architecture", "urban", "nature", "wildlife",
    "architecture", "water", "nature"
]

# Convert text to numerical features (upgraded to TF-IDF for better weighting)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)
y = categories

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with tuning (GridSearchCV for max accuracy)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict and evaluate with cross-validation
y_pred = best_model.predict(X_test)
accuracy = best_model.score(X_test, y_test)
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Model Test Accuracy: {accuracy:.2f}")
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")
print(f"Predicted category for test data: {y_pred[0]}")
print("Enter descriptions to test (type 'exit' to quit):")

while True:
    try:
        new_desc = input("> ").lower()
        if new_desc == "exit":
            print("Thanks for testing—Orion’s predictor signing off!")
            break
        new_X = vectorizer.transform([new_desc])
        new_pred = best_model.predict(new_X)
        print(f"Predicted category: {new_pred[0]}")
    except ValueError as e:
        print(f"Error processing input: {e}. Please try a different description.")
print("Orion’s multi-category text-to-image predictor—impress your clients!")