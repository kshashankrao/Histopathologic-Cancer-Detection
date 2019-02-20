from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import numpy as np
import keras
image_size = 96
BS = 32
validation_dir = "D:/DeepLearning/histopathologic/dataset/test"
model = load_model('weights-improvement-10-0.86.hdf5')
model.summary()
validation_datagen = ImageDataGenerator(rescale=1 / 255.0)
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=BS,
        class_mode='categorical',
        shuffle=False)
fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices
idx2label = dict((v,k) for k,v in label2index.items())
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
print(classification_report(validation_generator.classes, predicted_classes,
	target_names=validation_generator.class_indices.keys()))
cm = confusion_matrix(validation_generator.classes, predicted_classes)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# for i in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class])
#
#     original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
#     plt.figure(figsize=[7,7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
#     plt.show()
