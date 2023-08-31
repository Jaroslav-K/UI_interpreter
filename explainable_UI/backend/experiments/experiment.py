from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from keras import backend
import numpy as np
import pandas as pd
from PIL import Image
import time
import sys
import visualkeras
import random
from matplotlib.patches import Rectangle

model = keras.models.load_model('../LeNet5Model_28input.h5')
model.summary()

print('-------------------------DECODER--------------')

layer = model.get_layer('fully_connected_3')
last_layer = model.get_layer('output')

model_fully_connected = keras.Model(inputs=model.input, outputs=layer.output)
model_last_layer = keras.Model(inputs=last_layer.input, outputs=last_layer.output)


decoder = keras.models.load_model('../decoder_v2.h5')
decoder.summary()

model_min = -4.965
model_max = 6.332
step = (model_max - model_min) / 10

plt.gray()


np.set_printoptions(precision=3)


def experiment2():
    datas = get_images_by_value(value=5)
    features = layer_predict(datas[0])

    values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    cols = ['{0:.1f}'.format(col * 0.1) for col in range(0, 11)]

    nrow = 85
    ncol = 12

    fig = plt.figure(figsize=(ncol + 1, nrow + 1), dpi=200)

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.1, hspace=0.45,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    for c in range(ncol):
        ax = plt.subplot(gs[0, c])
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if c != 0:
            ax.text(0.2, 0.3, f'v [{cols[c - 1]}]')

    # get value on index
    for row in range(0, nrow - 1):

        print(row)
        value = features[0][row]
        normalized = normalize(value)
        array = np.asarray(values)
        idx = (np.abs(array - normalized)).argmin()
        nearest = values[idx]

        print(f'Original={value} Normalized={normalized} Nearest={nearest} Index={idx}')

        for c in range(ncol):
            ax = plt.subplot(gs[row + 1, c])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if c == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f'f [{row}]')
            else:
                if idx == c + 1:
                    ax.patch.set_edgecolor('green')
                    ax.patch.set_linewidth('8')

                features[0][row] = model_min + (c - 1) * step
                # predict
                prediction = model_predict(features)[0]
                classification = np.argmax(prediction)
                ax.set_title("[{}]".format(classification))

                data = decoder_predict(features)[0]
                img = keras.utils.array_to_img(data, scale=True)
                img = img.resize((102, 102), Image.ANTIALIAS)
                ax.imshow(img)

        # set value back to original
        features[0][row] = value

    plt.savefig('output.png')


def experiment3(image_index=0, index_f1=0, index_f2=1, file_name='experiment3.png'):
    data = get_all_test_images()

    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.scatter(x, y, zorder=1)

    features = layer_predict(data[image_index])
    values = features[0]

    normalized1 = normalize(values[index_f1])
    normalized2 = normalize(values[index_f2])

    array = np.asarray(x)

    idx1 = (np.abs(array - normalized1)).argmin()
    idx2 = (np.abs(array - normalized2)).argmin()

    nearest1 = array[idx1]
    nearest2 = array[idx2]

    print(f'Original1={values[index_f1]} Original2={values[index_f2]} Normalized={normalized1} Normalized2={normalized2} Index1={idx1} Index2={idx2}')

    for ind_x, x0 in enumerate(x):
        features[0][index_f1] = model_min + step * (x0 * 10)
        for ind_y, y0 in enumerate(y):
            features[0][index_f2] = model_min + step * (y0 * 10)

            prediction = model_predict(features)[0]
            classification = np.argmax(prediction)
            #ax.set_title("[{}]".format(classification))

            data = decoder_predict(features)[0]
            img = keras.utils.array_to_img(data, scale=True)
            img = img.resize((56, 56), Image.ANTIALIAS)

            ab = AnnotationBbox(OffsetImage(img, zoom=1), (x0, y0), frameon=False)

            if idx1 == ind_x and idx2 == ind_y:
                print('original')
                ab = AnnotationBbox(OffsetImage(img, zoom=1), (x0, y0), frameon=True,
                                    bboxprops=dict(edgecolor='green', lw=3))

            ax.add_artist(ab)
            ax.annotate(f"[{classification}]", (x0, y0),
                        xytext=(-9, 33),
                        textcoords='offset points',
                        fontsize=16,
                        zorder=3)

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.xlabel(f"Feature [{index_f1}]")
    plt.ylabel(f"Feature [{index_f2}]")

    plt.grid()
    plt.tight_layout()
    plt.savefig(file_name)


def experiment4(image_index, feature_index):
    data = get_images_by_value(value=5)
    features = layer_predict(data[image_index])

    np.set_printoptions(precision=3)

    values_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    confidence = []
    print(features)
    for i, x in enumerate(values_x):
        features[0][feature_index] = model_min + step * (x * 10)

        prediction = model_predict(features)[0]
        print(prediction)
        classification = np.argmax(prediction)
        conf = prediction[classification]
        confidence.append(conf)

    plt.xlabel('Feature value')
    plt.ylabel('Confidence')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.plot(values_x, confidence)

    plt.show()


def evaluate_classification_model():
    data = get_all_test_images()
    print('start evaluating')
    result = model.evaluate(data, data)
    print(f'Result={result}')


def confidence_chart():
    data = get_images_by_value(value=5)

    values_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for image_i, image in enumerate(data):
        features = layer_predict(image)
        print(f'Image index={image_i}')
        for feature_index in range(84):
            confidence = []
            original = features[0][feature_index]
            for i, x in enumerate(values_x):
                features[0][feature_index] = model_min + step * (x * 10)

                prediction = model_predict(features)[0]
                if i > 19:
                    print(prediction)
                classification = np.argmax(prediction)
                conf = prediction[classification]
                confidence.append(conf)
            features[0][feature_index] = original

            sum = 0
            for array_index in range(1, len(confidence)):
                sum += abs(confidence[array_index] - confidence[array_index - 1])
            if sum > 0.5:
                print(f'Feature index={feature_index} Sum={round(sum, 4)}')


def test_autoencoder():
    cols = ['Original', 'Reconstructed']

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 2))

    pad = 5  # in points
    images = get_all_test_images()
    for c in range(10):

        image = random.choice(images)

        for r in range(2):
            ax = axes[r, c]
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if r == 0:
                print('original')
                ax.imshow(image[0])
            else:
                print('reconsrtructed')
                features = layer_predict(image)

                data = decoder_predict(features)[0]
                img = keras.utils.array_to_img(data, scale=True)
                img = img.resize((102, 102), Image.ANTIALIAS)
                ax.imshow(img)

    fig.tight_layout()

    plt.show()


def find_prediction_changes():
    images = get_all_test_images()
    ind = 0
    with open('filename.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        original_stdout = sys.stdout
        for image in images:
            ind += 1
            print(f'Analyzing image={ind}')
            prediction = model.predict(image)
            classification = np.argmax(prediction)
            features = layer_predict(image)
            for i in range(84):
                original = features[0][i]
                for j in range(10):
                    features[0][i] = model_min + (j + 1) * step
                    predict = model_predict(features)
                    classif = np.argmax(predict)
                    if classification != classif:
                        print(f'Feature= {i} Old classification={classification} new classification={classif}')

                features[0][i] = original
        sys.stdout = original_stdout  # Reset the standard output to its original value


def visualize_model(m, file_path='model.png', legend=True):
    visualkeras.layered_view(m, spacing=25,
                             type_ignore=[keras.layers.Dropout, keras.layers.BatchNormalization, keras.layers.Activation, keras.layers.Reshape],
                             legend=legend,
                             to_file=file_path).show()


def normalize(value):
    return (value - model_min)/(model_max - model_min)


def save_img(originals, features):
    data = originals[0]

    fig.add_subplot(2, 2, 2)
    plt.imshow(data)
    plt.show()

    img.save('my.png')
    img.show()


def experiment1(value, file):
    train = get_images_by_value(value=value)

    total = 0

    for data_0 in train[:100]:
        m = layer_predict(data_0)
        total += m

    avg = total / 100

    abs_min = total / 100
    for i, v in enumerate(abs_min[0]):
        abs_min[0][i] = abs(abs_min[0][i] - model_min)

    abs_max = total / 100
    for i, v in enumerate(abs_max[0]):
        abs_max[0][i] = abs(abs_max[0][i] - model_max)

    k = 3
    idx_max = np.argpartition(abs_max, k)
    idx_min = np.argpartition(abs_min, k)

    print(f'Value= {value} .. 3 max values={idx_max[0][:3]} 3 min values={idx_min[0][:3]}')
    save_data_to_csv(data=avg, filename=file)
    return


def get_min_max():
    features_min = 1
    features_max = -1

    images = get_all_train_images()
    i = 0
    for image in images:
        features = layer_predict(image)
        image_min = np.min(features)
        image_max = np.max(features)
        if image_min < features_min:
            features_min = image_min
        if image_max > features_max:
            features_max = image_max
        i += 1
        print(f'Doing image={i}')
    print(f'Max={features_max}')
    print(f'Min={features_min}')


def get_error_predictions():
    test = pd.read_csv('../train.csv')
    data = test.values
    images = []

    for image in data:
        images.append(image[1:].reshape(-1, 28, 28))

    images = np.array(images)
    images_mean = images.mean()
    images_std = images.std()
    # (images - images_mean) / images_std
    id = 0
    for image in data:
        image_class = image[0]
        img = image[1:].reshape(-1, 28, 28)
        img = (img - images_mean) / images_std
        features = layer_predict(img)
        prediction = model_predict(features)

        max_arg = np.argmax(prediction)
        if max_arg != image_class:
            print(f'Invalid prediction shouldbe={image_class} found={max_arg}')
            image_to_save = image[1:].reshape(1, 28, 28, 1)
            image_predicted = decoder_predict(features)
            img = keras.utils.array_to_img(image_to_save[0], scale=True)
            img.save(f'error_img/error_img_{id}.png')

        id += 1


def upload_image():
    img_path = 'error_img/error_img_644.png'
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)[:, :, :1]
    img = np.expand_dims(img, axis=0)
    img = prepare_data(img)
    # features
    features = extract_features(img)
    layer_min = np.min(features)
    layer_max = np.max(features)
    # layer prediction
    layer_prediction(features)
    # encoder
    decoder.predict(features)
    return True


def prepare_data(img_array):
    mean = img_array.mean().astype(np.float32)
    std = img_array.std().astype(np.float32)
    return (img_array - mean)/std


def save_data_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=';')


def get_images_by_value(value):
    test = pd.read_csv('../train.csv')
    data = test.values
    images = []

    for image in data:
        if image[0] == value:
            images.append(image[1:].reshape(-1, 28, 28))

    images_mean = data.mean()
    images_std = data.std()
    return (images - images_mean)/images_std


def get_all_train_images():
    train = pd.read_csv('../train.csv')
    labels = train[['label']]
    data = train.drop(train.columns[[0]], axis=1)
    data = data.values
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    print(f'Train data mean={data_mean} std={data_std}')
    images = []
    for img in data:
        images.append(img.reshape(-1, 28, 28))

    return images


def get_all_test_images():
    test = pd.read_csv('../test.csv')

    data = test.values
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    print(f'Test data mean={data_mean} std={data_std}')
    images = []
    for img in data:
        images.append(img.reshape(-1, 28, 28))

    return images


def speed_test():
    images = get_all_test_images()
    total_time_encoder, total_time_classificator = 0, 0
    counter = 0
    start_time_encoder, start_time_classificator = 0, 0
    end_time_encoder, end_time_classificator = 0, 0
    for image in images:
        start_time_classificator = time.process_time()
        features = layer_predict(image)
        end_time_classificator = time.process_time()
        total_time_classificator += (end_time_classificator - start_time_classificator)
        start_time_encoder = time.process_time()
        encoded = decoder_predict(features)
        end_time_encoder = time.process_time()
        total_time_encoder += (end_time_encoder - start_time_encoder)
        counter += 1
        if counter % 10 == 0:
            avg_class = total_time_classificator / counter
            avg_encoder = total_time_encoder / counter
            print(f'Counter={counter} average classification={avg_class} average encoder={avg_encoder}')


def layer_predict(img):
    return model_fully_connected.predict(img)


def model_predict(features):
    return model_last_layer.predict(features)


def decoder_predict(features):
    return decoder.predict(features)


def build_autoencoder():
    autoencoder = keras.Sequential([
        model_fully_connected,
        decoder
    ])
    return autoencoder

if __name__ == "__main__":
    #experiment1(0, 'average_0.csv')
    #experiment2()
    #get_error_predictions()
    #find_prediction_changes()
    #experiment3(image_index=165, index_f1=0, index_f2=83, file_name='experiment3_v5.png')
    #confidence_chart()
    experiment4(0, 20)
    #evaluate_classification_model()
    #experiment4()
    #speed_test()
    #get_min_max()
    #get_all_train_images()
    #get_all_test_images()
    #visualize_model(decoder, file_path='autoencoder2.png', legend=True)
    #test_autoencoder()