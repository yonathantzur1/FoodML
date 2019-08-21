import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

main_dir = "food/"
input_dir = main_dir + "input/"

image_size = (224, 224)

foods = ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden", "Apple Golden", "Apple Golden",
         "Apple Granny Smith", "Apple Pink Lady", "Apple Red", "Apple Red", "Apple Red", "Apple Red Delicious",
         "Apple Red Yellow", "Apple Red Yellow", "Apricot", "Avocado", "Avocado ripe", "Banana",
         "Banana Lady Finger", "Banana Red", "Cactus fruit", "Cantaloupe", "Cantaloupe", "Carambula", "Cherry",
         "Cherry", "Cherry Rainier", "Cherry Wax Black", "Cherry Wax Red", "Cherry Wax Yellow", "Chestnut",
         "Clementine", "Cocos", "Dates", "Granadilla", "Grape Blue", "Grape Pink", "Grape White", "Grape White",
         "Grape White", "Grape White", "Grapefruit Pink", "Grapefruit White", "Guava", "Hazelnut", "Huckleberry",
         "Kaki", "Kiwi", "Kohlrabi", "Kumquats", "Lemon", "Lemon Meyer", "Limes", "Lychee", "Mandarine", "Mango",
         "Mangostan", "Maracuja", "Melon Piel de Sapo", "Mulberry", "Nectarine", "Orange", "Papaya", "Passion Fruit",
         "Peach", "Peach", "Peach Flat", "Pear", "Pear Abate", "Pear Kaiser", "Pear Monster", "Pear Red",
         "Pear Williams", "Pepino", "Pepper Green", "Pepper Red", "Pepper Yellow", "Physalis", "Physalis with Husk",
         "Pineapple", "Pineapple Mini", "Pitahaya Red", "Plum", "Plum", "Plum", "Pomegranate", "Pomelo Sweetie",
         "Quince", "Rambutan", "Raspberry", "Redcurrant", "Salak", "Strawberry", "Strawberry Wedge", "Tamarillo",
         "Tangelo", "Tomato", "Tomato", "Tomato", "Tomato", "Tomato Cherry Red", "Tomato Maroon",
         "Tomato Yellow", "Walnut", "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
         "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito", "bruschetta", "caesar salad",
         "cannoli", "caprese salad", "carrot cake", "ceviche", "cheese plate", "cheesecake", "chicken curry",
         "chicken quesadilla", "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
         "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes", "deviled eggs", "donuts",
         "dumplings", "edamame", "eggs benedict", "escargots", "falafel", "filet mignon", "fish and chips",
         "foie gras", "french fries", "french onion soup", "french toast", "fried calamari", "fried rice",
         "frozen yogurt", "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich", "grilled salmon",
         "guacamole", "gyoza", "hamburger", "hot and sour soup", "hot dog", "huevos rancheros", "hummus", "ice cream",
         "lasagna", "lobster bisque", "lobster roll sandwich", "macaroni and cheese", "macarons", "miso soup",
         "mussels", "nachos", "omelette", "onion rings", "oysters", "pad thai", "paella", "pancakes", "panna cotta",
         "peking duck", "pho", "pizza", "pork chop", "poutine", "prime rib", "pulled pork sandwich", "ramen",
         "ravioli", "red velvet cake", "risotto", "samosa", "sashimi", "scallops", "seaweed salad",
         "shrimp and grits", "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak",
         "strawberry shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna tartare", "waffles"]


def load_image():
    return ImageDataGenerator().flow_from_directory(input_dir,
                                                    target_size=image_size,
                                                    batch_size=1,
                                                    class_mode=None,
                                                    shuffle=False).next()


def get_features_model():
    base_model = load_model(main_dir + "/model.h5")
    features_layer = Flatten()(base_model.get_layer('block5_pool').output)
    return Model(inputs=base_model.input, outputs=features_layer)


def find_food():
    model = load_model("./model.h5")
    input_img = load_image()
    input_features = list(model.predict(input_img))[0]
    max_index = 0
    max_value = 0

    for index in range(len(input_features)):
        if input_features[index] > max_value:
            max_value = input_features[index]
            max_index = index

    print()
    print("Food: " + foods[max_index])
    print("Accuracy: " + str(max_value) + "%")


if __name__ == '__main__':
    find_food()
