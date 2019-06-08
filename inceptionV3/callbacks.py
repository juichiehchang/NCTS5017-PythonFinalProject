from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

# Descreasing learning rate as epoch gets larger
def scheduler(epochs):
    if(epochs < 8):
        return 0.000064
    elif(epochs < 16):
        return 0.01
    elif(epochs < 32):
        return 0.002
    elif(epochs < 48):
        return 0.0004
    else:
        return 0.000009

def get_callbacks(weights_file):
    # get checkpoint
    checkpoint = ModelCheckpoint(weights_file, monitor = 'val_acc', save_best_only = True, verbose = 1)

    # get learning rate
    lr = LearningRateScheduler(scheduler)

    # get early stopping
    es = EarlyStopping(monitor = 'val_acc', patience = 3)
    
    return [lr, checkpoint, es]
