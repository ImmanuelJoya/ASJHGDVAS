#Core Learning Algorithams: Building the model 
import tensorflow as tf
import pandas as pd   

# -----------------------------
# Step 1: Load the Iris dataset
# -----------------------------
train_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

# Column names
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", train_url)
test_path = tf.keras.utils.get_file("iris_test.csv", test_url)

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# -----------------------------
# Step 2: Build preprocessing
# -----------------------------
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train.to_numpy())

# -----------------------------
# Step 3: Build the model
# -----------------------------
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")  # 3 species
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Step 4: Train the model
# -----------------------------
model.fit(train, train_y, epochs=50, batch_size=16, verbose=1)

# -----------------------------
# Step 5: Evaluate the model
# -----------------------------
loss, acc = model.evaluate(test, test_y, verbose=0)
print(f"\nTest accuracy: {acc:.3f}\n")

# -----------------------------
# Step 6: Prediction function
# -----------------------------
def ask_user_and_predict():
    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    user_input = []

    print("Please type numeric values as prompted.")
    for feature in features:
        while True:
            val = input(f"{feature}: ")
            try:
                user_input.append(float(val))
                break
            except ValueError:
                print("❌ Please enter a numeric value.")

    # Convert to tensor
    user_tensor = tf.convert_to_tensor([user_input], dtype=tf.float32)

    # Predict
    probs = model.predict(user_tensor, verbose=0)[0]
    class_id = tf.argmax(probs).numpy()
    print(f'\n🌸 Prediction: "{SPECIES[class_id]}" ({probs[class_id]*100:.1f}%)')

# -----------------------------
# Step 7: Run prediction
# -----------------------------
ask_user_and_predict()

