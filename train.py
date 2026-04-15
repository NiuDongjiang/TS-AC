import os
import torch.nn as nn
from utils.util import *
import torch


def train(args, model, train_loader, test_loader, class_weight):
    model_save_path = './result/' + args['MODEL'] + '_' + args['TARGET_NAME'] + '.pkl'
    
    if not os.path.exists('./result'):
        os.makedirs('./result')

    criterion = WeightedBCELoss(weights=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["WEIGHT_DECAY"])
 
    num_epochs = 1000
    best_test_ba = 0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        model, loss, total_loss, train_acc, _ = predict(args, model, train_loader, criterion, optimizer, True, epoch, t='train')
        model.eval()
        with torch.no_grad():
            '''
            pre_model = model.load_state_dict(torch.load(pkl_name))
            state_model = model.state_dict()
            for name, param in pre_model.items():
                if name in state_model:
                    state_model[name] = param
            model.load_state_dict(state_model)
            '''
            _, _, test_loss, test_acc, y_proba = predict(args, model, test_loader, criterion, optimizer, False, epoch, t='test')
            print('Epoch: {}/{} '.format(epoch + 1, num_epochs),
                  ' Training Loss: {:.3f}'.format(total_loss / len(train_loader)),
                  ' Training Balanced Accuracy: {:.3f}'.format(train_acc),
                  ' Test Loss: {:.3f}'.format(test_loss / len(test_loader)),
                  ' Test Balanced Accuracy: {:.3f}'.format(test_acc))
        model.train()
        
        if test_acc > best_test_ba:
            print("Test Accuracy improved from {:.3f} -> {:.3f}".format(best_test_ba, test_acc))
            best_test_ba = test_acc
            torch.save(model.state_dict(), model_save_path)
            early_stopping_count = 0
            y_proba_test = y_proba  

        else:
            early_stopping_count += 1
            print("Test Loss did not improved from {:.3f}.. Counter {}/{}".format(best_test_ba, early_stopping_count, args['EARLY_STOPPING_PATIENCE']))
            if early_stopping_count > args['EARLY_STOPPING_PATIENCE']:
                print("Early Stopped ..")
                break

    return y_proba_test