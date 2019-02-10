import data
import ffnnmodel
import routine
import config

# To run, make sure you have a data/ folder containing MNIST and a model/ folder along with your code/ folder

def run():
    train_loader = data.get_loader("train")
    val_loader = data.get_loader("val")
    test_loader = data.get_loader("test")
    test_labels = data.get_test_labels()

    if config.load_model:
        model = ffnnmodel.FFNN.load()
    else:
        model = ffnnmodel.FFNN()

    routine.train(model, train_loader, val_loader)
    test_outputs = routine.predict(model, test_loader)
    accuracy = (test_outputs == test_labels).mean()
    print("Test accuracy :", accuracy)


if __name__ == "__main__":
    run()
