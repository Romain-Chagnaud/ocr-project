from PIL import Image
import numpy
import os, shutil


BD_TEST = 'baseProjetOCR/test/'


def read_image(path):
    return numpy.asarray(Image.open(path).convert('L'))


def read_images(n_max_images):
    images = []
    nb = 0
    nb_range = 1
    for i in range(0,n_max_images):
        im = Image.open(f'baseProjetOCR/{nb}_{nb_range}.png')

        # redimensionnement de l'image
        im = im.resize((200, 200))

        # converstion de l'image en noir et blanc
        #im = im.convert('1')
        
        im.save(f'{BD_TEST}{nb}_{nb_range}.png')
        images.extend([read_image(f'{BD_TEST}{nb}_{nb_range}.png')])
        im.close
        nb_range += 1
        if nb_range > 10:
            nb_range = 1
            nb += 1
    return images


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    # distance euclidienne entre les vecteurs x et y
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_test, y_test, k):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        print(test_sample_idx, end=' ', flush=True)
        training_distances = get_training_distances_for_test_sample(
            X_test, test_sample
        )
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_test[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    print()
    return y_pred


def delete_test_folder():
    folder = f'{BD_TEST}'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Echec de la suppression %s : %s' % (file_path, e))
    os.rmdir(f'{BD_TEST}')
    pass


def main():
    # options du test
    n_test = 100
    k = 7
    print(f'n_test: {n_test}')
    print(f'k: {k}')
    print(f"Veuillez patienter {n_test} tours :")

    # cr??ation du dossier de test
    if os.path.exists(f'{BD_TEST}'):
        delete_test_folder()
    os.mkdir(f'{BD_TEST}')

    # on pr??pare les images
    X_test = read_images(n_test)
    y_test = [0,0,0,0,0,0,0,0,0,0,
              1,1,1,1,1,1,1,1,1,1,
              2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,
              4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,
              6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,
              8,8,8,8,8,8,8,8,8,8,
              9,9,9,9,9,9,9,9,9,9]
    X_test = extract_features(X_test)

    # on effectue le r??seau de neuronnes
    y_pred = knn(X_test, y_test, k)

    # on calcule la pr??cision du r??sultat
    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_pred, y_test)
    ]) / len(y_test)

    # on affiche le r??sultat
    print(f'\nR??sultats du test OCR :')
    print(f'0| {y_pred[0:9:1]}')
    print(f'1| {y_pred[10:19:1]}')
    print(f'2| {y_pred[20:29:1]}')
    print(f'3| {y_pred[30:39:1]}')
    print(f'4| {y_pred[40:49:1]}')
    print(f'5| {y_pred[50:59:1]}')
    print(f'6| {y_pred[60:69:1]}')
    print(f'7| {y_pred[70:79:1]}')
    print(f'8| {y_pred[80:89:1]}')
    print(f'9| {y_pred[90:99:1]}')

    print(f'Le taux de reconnaissance est de {accuracy * 100}%')

    # suppression du dossier de test
    delete_test_folder()


if __name__ == '__main__':
    main()