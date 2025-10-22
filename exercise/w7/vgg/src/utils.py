def load_image(image_path, target_size=(224, 224)):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array

def preprocess_input(image_array):
    # Assuming the model expects input in the format (batch_size, height, width, channels)
    return np.expand_dims(image_array, axis=0)

def visualize_predictions(image_array, predictions, class_names):
    import matplotlib.pyplot as plt

    plt.imshow(image_array)
    plt.title(f'Predicted: {class_names[np.argmax(predictions)]}')
    plt.axis('off')
    plt.show()

def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')