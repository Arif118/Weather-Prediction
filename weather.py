import requests as r
import pandas as pd
import random
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

API_KEY = '0dcff6d1048e25ef4de7711889a9917e'
weather_data = []

def get_weather_data(data):

    try:
        country = data['sys']['country']

    except KeyError:
        country = 'Unknown'

    features = {

        'Temp': data['main']['temp'] - 273.15,

        'Pressure': data['main']['pressure'],

        'Humidity': data['main']['humidity'],

        'Wind_speed': data['wind']['speed'],

        'Cloudiness': data['clouds']['all'],

        'Country': country,

        'Target': data['weather'][0]['main']
    }

    weather_data.append(features)

for i in range(20):

    lat = random.uniform(-60, 60)

    lon = random.uniform(-100, 100)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={API_KEY}"
    )

    response = r.get(url)

    data = response.json()

    get_weather_data(data)

df = pd.DataFrame(weather_data)
df.head()

X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_features = [
    'Temp',
    'Pressure',
    'Humidity',
    'Wind_speed',
    'Cloudiness'
]

categorical_features = [
    'Country'
]

preprocessor = ColumnTransformer(
    transformers=[

        (
            'num',
            StandardScaler(),
            numeric_features
        ),

        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        )
    ]
)

pipe = Pipeline([

    ('preprocessor', preprocessor),

    (
        'model',

        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

joblib.dump(pipe, "weather_model.pkl")