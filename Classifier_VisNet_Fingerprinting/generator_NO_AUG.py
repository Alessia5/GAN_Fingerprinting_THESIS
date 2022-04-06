import keras
import config
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

TR_Gen = train_datagen.flow_from_directory(  # TR_Gen
    directory=config.DS_Path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training')

Val_Gen = val_datagen.flow_from_directory(  # Val_Gen
    directory=config.DS_Path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    subset='validation'
)
# aggiungere validation sopra Val_Gen, Val e Tr stesso path (DS_Path) mentre TE path diverso (TE_Path)
test_datagen = ImageDataGenerator(  # nel test non serve l'augmentation
    rescale=1. / 255)

TE_Gen = test_datagen.flow_from_directory(  # TE_Gen
    directory=config.TE_Path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical"

)

num_samples = TR_Gen.n
num_classes = TR_Gen.num_classes
input_shape = TR_Gen.image_shape

classnames = [k for k, v in TR_Gen.class_indices.items()]

print("Image input %s" % str(input_shape))
print("Classes: %r" % classnames)

print('Loaded %d training samples from %d classes.' % (num_samples, num_classes))  # training samples
print('Loaded %d validation samples from %d classes.' % (Val_Gen.n, Val_Gen.num_classes))  # validation samples
print('Loaded %d test samples from %d classes.' % (TE_Gen.n, TE_Gen.num_classes))  # testing samples