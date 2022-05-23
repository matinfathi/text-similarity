from siamese_model import siamese_model
from load_data import load_dataset
import json


if __name__ == '__main__':
    # Read constant parameters from config file
    with open('./config.json', 'r') as f:
        config = json.load(f)

    # Load dataset
    train, test, glove_weights = load_dataset(config['dataset_name'],
                                              config['data_length'],
                                              config['glove_path'],
                                              config['maximum_length_sentence'],
                                              config['test_data_size'])
    # Extract x and y train and test
    (x_train1, x_train2, y_train) = train
    (x_test1, x_test2, y_test) = test

    # Create model
    model = siamese_model(glove_weights,
                          config['maximum_length_sentence'],
                          config['lstm_units'],
                          config['dropout'])

    # train model
    history = model.fit([x_train1, x_train2], y_train,
                        validation_data=([x_test1, x_test2], y_test),
                        batch_size=config['batch_size'],
                        epochs=config['epochs'])

    # Save the model
    model.save('./model.h5')
