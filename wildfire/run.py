from wildfire.dataset.model_definitions import EnsembleClassifier, EnsembleRegressor
import sys

sys.modules['__main__'].EnsembleRegressor = EnsembleRegressor
sys.modules['__main__'].EnsembleClassifier = EnsembleClassifier

from wildfire import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)