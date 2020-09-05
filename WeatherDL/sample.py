import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from WeatherDL.data_maker import dataset_maker
from WeatherDL.model_maker import model_3

# Extract data from data_maker
X, y = dataset_maker(window=5, forecast_day=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, shuffle=False)

# Open model from model_maker
model = model_3((5, 8, 20, 6))
print(model.summary())

# Fit model, and extract training & validation metrics
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=5,
                    epochs=30,
                    verbose=2,
                    shuffle=False)

# Prediction
y_pred = model.predict(X_test)

# Data Visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('MAE')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('MAPE')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
