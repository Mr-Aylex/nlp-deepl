import numpy as np


# TODO: create a load_data function that loads the text data from the data/raw folder
def normaliser_taille(arr, taille_cible, padding_value=0):
    longueur_actuelle = len(arr)
    if longueur_actuelle > taille_cible:
        # Couper le tableau si plus long que la taille cible
        return arr[:taille_cible]
    elif longueur_actuelle < taille_cible:
        # Étendre le tableau avec des zéros (ou toute autre valeur) si plus court
        return np.pad(arr, (0, taille_cible - longueur_actuelle), 'constant', constant_values=padding_value)

    else:
        # Ne rien faire si la taille est correcte
        return arr


def make_features(df, task, seq_len=None):
    y = get_output(df, task, seq_len=seq_len)

    X = df["video_name"]

    return X, y


def get_output(df, task, seq_len=None):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
        if seq_len is None:
            raise ValueError("seq_len must be specified for task is_name")
        y = y.apply(lambda x: normaliser_taille(x, seq_len))
        y = np.stack(y.values)
        y = y.reshape(y.shape[0], y.shape[1], 1)
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
