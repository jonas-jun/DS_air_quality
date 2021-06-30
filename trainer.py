from sklearn.metrics import f1_score
import torch
import os


def evaluate(loader, model, criterion, opt):
    model.to(opt.device)
    model.eval()

    with torch.no_grad():
        loss, total, acc = 0, 0, 0
        y_pred, y_true = list(), list()

        for batch in loader:
            inputs = batch['inputs'].float().to(opt.device)
            labels = batch['labels'].long().to(opt.device)
            outputs = model(inputs)
            curloss = criterion(outputs, labels)
            loss += curloss.item()
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            acc += torch.sum(preds==labels).item()
            total += inputs.size(0)
    
    F1 = round(f1_score(y_true, y_pred, average='macro'), 4)*100
    return loss/total, acc/total, F1, y_true, y_pred

def trainer(train_loader, val_loader, model, criterion, optimizer, scheduler, opt):
    model.to(opt.device)
    max_val_acc, max_val_f1, max_val_epoch, global_step = 0, 0, 0, 0

    for i_epoch in range(1, opt.num_epochs+1):
        n_correct, n_total, loss_total = 0, 0, 0
        model.train()
    
        for batch in train_loader:
            global_step += 1
            inputs = batch['inputs'].float().to(opt.device)
            labels = batch['labels'].long().to(opt.device)

            model.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            n_correct += (torch.argmax(outputs, -1)==labels).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)

            if global_step % opt.log_steps == 0:
                train_loss = loss_total / n_total
                train_acc = n_correct / n_total
                print('  global step: {:,} | train loss: {:.3f} | train acc: {:.2f}'\
                    .format(global_step, train_loss, train_acc*100))
            
        val_loss, val_acc, val_f1, _, _ = evaluate(val_loader, model, criterion, opt)

        if i_epoch >= 3:
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                path = 'state_dict/{}_epoch_{}_val_acc_{}%'.format(opt.model_name, i_epoch, round(val_acc*100, 2))
                path = os.path.join(opt.dir, path) # for colab
                torch.save(model.state_dict(), path)
                print('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= opt.patience:
                print('>> EARLY STOP')
                break
        print('Epoch: {:02d} | Val Loss: {:.3f} | Val Acc: {:.2f}%'.format(i_epoch, val_loss, val_acc*100))
    print('Best Val Acc: {:.2f}% at {:02d} epoch'.format(max_val_acc*100, max_val_epoch))
    best_path = 'state_dict/BEST_{}_val_acc_{}%'.format(opt.model_name, round(max_val_acc*100, 2))
    best_path = os.path.join(opt.dir, best_path)
    torch.save(model.state_dict(), best_path)
    print('>> saved best state dict: {}'.format(best_path))
    return max_val_acc, max_val_epoch, best_path

# for insert mode